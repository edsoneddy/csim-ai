"""Structural equivalence checks for mutations. We never execute code or
test cases to validate a mutation (section 5 of the project brief) --
correctness has to come from the transform's own preconditions plus a
static equivalence check here.
"""
from __future__ import annotations

import ast


def parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove docstring-position Expr(Constant(str)) nodes in place."""
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            val = body[0].value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                body.pop(0)
    return tree


def cosmetic_equivalent(original: str, transformed: str) -> bool:
    """True if both parse to the same AST once docstrings are stripped from
    both. Comments/whitespace are never part of `ast` in the first place,
    so this is the entire equivalence surface for L1 mutations."""
    try:
        a = _strip_docstrings(ast.parse(original))
        b = _strip_docstrings(ast.parse(transformed))
    except SyntaxError:
        return False
    return ast.dump(a) == ast.dump(b)


# NOTE: there is deliberately no generic "identifier_equivalent" here. A
# spelling-based canonicalization (relabel every Name/arg to ID<n> by
# first-occurrence order) is NOT scope-aware, and this dataset reuses the
# same name across unrelated scopes constantly (e.g. a module-level
# `cases` and an unrelated function parameter also called `cases`) -- a
# spelling-only check conflates them and flags correct renames as broken.
# L2's real check is training.mutate.l2_rename._verify_roundtrip, which
# re-derives scope groups on the mutated code instead of trusting spelling.


def _sort_statement_lists(node: ast.AST) -> ast.AST:
    """Recursively (post-order, so nested blocks are already canonical
    before they're used as a sort key) sort every statement-list field by
    each statement's own dump, so two trees that differ only in the order
    of statements within some block(s) compare equal."""
    for child in ast.iter_child_nodes(node):
        _sort_statement_lists(child)
    for field in ("body", "orelse", "finalbody"):
        stmts = getattr(node, field, None)
        if isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt):
            stmts.sort(key=ast.dump)
    return node


def order_equivalent(original: str, transformed: str) -> bool:
    """True if the transformed source has, at every block, the same
    statements as the original, only possibly reordered. NOTE: this is a
    necessary but not sufficient check for L3 -- reordering is
    order-agnostic by construction, so this alone cannot tell a safe
    reordering from an unsafe one (both produce the same sorted/canonical
    form). It only guards against a transform bug that changes content
    (drops/duplicates/edits a statement) rather than just its position.
    The actual safety property -- that the reordered statements have no
    data dependency or side effect between them -- has to hold before the
    reordering is applied; see *_mutually_independent below, which
    re-derives it independently via `ast` rather than trusting
    l3_reorder.py's own libcst-based candidate selection."""
    try:
        a = _sort_statement_lists(ast.parse(original))
        b = _sort_statement_lists(ast.parse(transformed))
    except SyntaxError:
        return False
    return ast.dump(a) == ast.dump(b)


def _has_side_effect(stmt: ast.stmt) -> bool:
    """Conservative: anything that could plausibly do I/O, mutate shared
    state, or otherwise make execution order observable. Call covers
    ordinary function/method calls (input(), print(), list.append(), a
    user function...); Await/Yield cover the coroutine/generator forms."""
    return any(isinstance(n, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom)) for n in ast.walk(stmt))


def _write_names(stmt: ast.stmt) -> set[str] | None:
    """Names this statement binds, or None if it's not one of the simple
    shapes this checker understands (in which case the caller must treat
    it as unsafe to reorder, not silently permit it)."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        return {stmt.targets[0].id}
    return None


def assignments_mutually_independent(sources: list[str]) -> bool:
    """Ground truth for whether a list of statement source snippets (each
    exactly one statement) can be freely reordered relative to each
    other: every statement must be call-free (no observable side effect)
    and their write-name-sets must be pairwise disjoint from each other's
    read AND write sets, so there is no data dependency in either
    direction and no two statements clobber the same name."""
    infos: list[tuple[set[str], set[str]]] = []
    for src in sources:
        try:
            body = ast.parse(src).body
        except SyntaxError:
            return False
        if len(body) != 1:
            return False
        stmt = body[0]
        if _has_side_effect(stmt):
            return False
        writes = _write_names(stmt)
        if writes is None:
            return False
        reads = {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        infos.append((writes, reads))

    for i, (writes_i, _) in enumerate(infos):
        for j, (writes_j, reads_j) in enumerate(infos):
            if i == j:
                continue
            if writes_i & reads_j or writes_i & writes_j:
                return False
    return True


def _import_bound_names(stmt: ast.stmt) -> set[str] | None:
    """Names an import statement binds, or None if it's not a plain
    import/import-from (e.g. `from x import *`, which the caller must
    treat as unsafe -- a star import's bound names aren't statically
    knowable, so they can't be checked for collisions)."""
    if isinstance(stmt, ast.Import):
        return {(alias.asname or alias.name.split(".")[0]) for alias in stmt.names}
    if isinstance(stmt, ast.ImportFrom):
        if any(alias.name == "*" for alias in stmt.names):
            return None
        return {(alias.asname or alias.name) for alias in stmt.names}
    return None


def imports_mutually_independent(sources: list[str]) -> bool:
    """Ground truth for whether a list of import-statement source
    snippets can be freely reordered: none may be a star import, and no
    two may bind the same name (which would make the reordering change
    which module a later collision-shadowed name actually refers to)."""
    all_names: list[set[str]] = []
    for src in sources:
        try:
            body = ast.parse(src).body
        except SyntaxError:
            return False
        if len(body) != 1:
            return False
        names = _import_bound_names(body[0])
        if names is None:
            return False
        all_names.append(names)
    seen: set[str] = set()
    for names in all_names:
        if seen & names:
            return False
        seen |= names
    return True
