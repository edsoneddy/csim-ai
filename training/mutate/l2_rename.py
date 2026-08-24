"""L2: identifier renaming with scope analysis.

Uses libcst's ScopeProvider to find, for each locally-bound name (function
parameters and variables actually assigned somewhere in that exact scope
-- never imports, builtins, attributes, keyword-argument names, or a name
touched by a global/nonlocal declaration anywhere in the file), every Name
occurrence that belongs to that one binding, and renames them all
together to a fresh identifier that collides with nothing else in the
file. Safe by construction: substituting a bound name for an unused one,
consistently everywhere it's bound or read, never changes what the code
does -- verified by _verify_roundtrip before a mutation is ever reported
as applied (see its docstring for why a plain canonicalize-and-compare
check on the mutated code isn't sufficient here).
"""
from __future__ import annotations

import ast
import builtins as _builtins
import keyword
import random

import libcst as cst
from libcst.metadata import ClassScope, BuiltinScope, ImportAssignment, MetadataWrapper, ScopeProvider

from .base import MutationResult

_EXCLUDED_NAMES = {"self", "cls"}
_NAME_POOL = [
    "val", "num", "item", "tmp", "res", "data", "idx", "arr", "cnt", "acc",
    "buf", "out", "elem", "aux", "total", "curr", "prev", "entry", "value",
    "count", "result", "index", "temp", "amount", "piece", "chunk", "state",
]


def _collect_global_nonlocal_names(module: cst.Module) -> set[str]:
    names: set[str] = set()

    class _Visitor(cst.CSTVisitor):
        def visit_Global(self, node: cst.Global) -> None:
            names.update(item.name.value for item in node.names)

        def visit_Nonlocal(self, node: cst.Nonlocal) -> None:
            names.update(item.name.value for item in node.names)

    module.visit(_Visitor())
    return names


def _collect_keyword_argument_names(module: cst.Module) -> set[str]:
    """Names used as `f(name=...)` call-site keywords anywhere in the
    file. libcst's ScopeProvider doesn't track these at all (correctly --
    they aren't variable references), which means a parameter can look
    like a safe rename target by scope analysis alone while a call site
    still refers to it by its old name as a keyword argument. Renaming a
    parameter without also excluding this would silently break that call,
    so any name appearing as a keyword here is never renamed."""
    names: set[str] = set()

    class _Visitor(cst.CSTVisitor):
        def visit_Arg(self, node: cst.Arg) -> None:
            if node.keyword is not None:
                names.add(node.keyword.value)

    module.visit(_Visitor())
    return names


def _all_identifiers(module: cst.Module) -> set[str]:
    names: set[str] = set()

    class _Visitor(cst.CSTVisitor):
        def visit_Name(self, node: cst.Name) -> None:
            names.add(node.value)

    module.visit(_Visitor())
    return names | set(keyword.kwlist) | set(dir(_builtins))


def _find_renameable_groups(wrapper: MetadataWrapper) -> dict[tuple[int, str], list[cst.Name]]:
    scopes = wrapper.resolve(ScopeProvider)
    unsafe_names = _collect_global_nonlocal_names(wrapper.module) | _collect_keyword_argument_names(wrapper.module)

    groups: dict[tuple[int, str], list[cst.Name]] = {}
    scope_of: dict[tuple[int, str], object] = {}
    for node, scope in scopes.items():
        if not isinstance(node, cst.Name) or scope is None:
            continue
        key = (id(scope), node.value)
        groups.setdefault(key, []).append(node)
        scope_of[key] = scope

    renameable: dict[tuple[int, str], list[cst.Name]] = {}
    for key, nodes in groups.items():
        _, name = key
        if name in _EXCLUDED_NAMES or keyword.iskeyword(name) or name.startswith("__"):
            continue
        if name in unsafe_names:
            continue
        scope = scope_of[key]
        if isinstance(scope, (ClassScope, BuiltinScope)):
            continue
        local_assignments = [a for a in scope.assignments if a.name == name]
        if not local_assignments or any(isinstance(a, ImportAssignment) for a in local_assignments):
            continue
        renameable[key] = nodes
    return renameable


def _fresh_name(used: set[str], rng: random.Random) -> str:
    for base in rng.sample(_NAME_POOL, k=len(_NAME_POOL)):
        if base not in used:
            return base
    i = 0
    while f"v{i}" in used:
        i += 1
    return f"v{i}"


class _Renamer(cst.CSTTransformer):
    def __init__(self, mapping_by_id: dict[int, str]):
        self.mapping_by_id = mapping_by_id

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
        new_name = self.mapping_by_id.get(id(original_node))
        return updated_node.with_changes(value=new_name) if new_name else updated_node


def _verify_roundtrip(original_code: str, mutated_code: str, pairs: list[tuple[str, str]]) -> bool:
    """Independent check: re-derive renameable groups from scratch on the
    *mutated* code, revert each (old, new) pair using the freshly-derived
    group for `new`, and require the result to be exactly the original
    AST.

    A naive "canonicalize identifiers by spelling and compare" check (an
    earlier version of this) is NOT sufficient: this dataset reuses the
    same name across different scopes constantly (e.g. a module-level
    `cases` and an unrelated function parameter also called `cases`), and
    spelling-based canonicalization conflates them, flagging correct
    renames as broken. Re-deriving scope groups on the mutated code and
    reverting is scope-aware, so it doesn't have that false-positive, and
    it still catches real bugs: a missed occurrence means the group for
    `new` won't cover everything that needs reverting, and a collision
    with something else means the freshly-derived group for `new` won't
    match the original group 1:1 -- either way the final dump differs."""
    try:
        mutated_wrapper = MetadataWrapper(cst.parse_module(mutated_code))
    except cst.ParserSyntaxError:
        return False
    mutated_groups = _find_renameable_groups(mutated_wrapper)

    revert_mapping: dict[int, str] = {}
    for old_name, new_name in pairs:
        nodes = next((ns for (_, name), ns in mutated_groups.items() if name == new_name), None)
        if nodes is None:
            return False
        for node in nodes:
            revert_mapping[id(node)] = old_name

    reverted_code = mutated_wrapper.module.visit(_Renamer(revert_mapping)).code
    try:
        return ast.dump(ast.parse(reverted_code)) == ast.dump(ast.parse(original_code))
    except SyntaxError:
        return False


def apply_random(code: str, rng: random.Random) -> MutationResult:
    """Renames a random non-empty subset of the eligible local bindings.
    Self-verifies via round-trip before reporting success; on failure
    returns applied=False (the caller just tries the next variant)."""
    try:
        wrapper = MetadataWrapper(cst.parse_module(code))
    except cst.ParserSyntaxError:
        return MutationResult("L2", "unparseable", code, False)

    groups = _find_renameable_groups(wrapper)
    if not groups:
        return MutationResult("L2", "noop", code, False)

    keys = list(groups.keys())
    rng.shuffle(keys)
    chosen = keys[: rng.randint(1, len(keys))]

    used = _all_identifiers(wrapper.module)
    mapping_by_id: dict[int, str] = {}
    pairs: list[tuple[str, str]] = []
    for sid, name in chosen:
        new_name = _fresh_name(used, rng)
        used.add(new_name)
        for node in groups[(sid, name)]:
            mapping_by_id[id(node)] = new_name
        pairs.append((name, new_name))

    new_code = wrapper.module.visit(_Renamer(mapping_by_id)).code
    if new_code == code:
        return MutationResult("L2", "noop", code, False)
    if not _verify_roundtrip(code, new_code, pairs):
        return MutationResult("L2", "rejected:roundtrip_mismatch", code, False)

    rule = "rename:" + ",".join(f"{o}->{n}" for o, n in pairs)
    return MutationResult("L2", rule, new_code, True)
