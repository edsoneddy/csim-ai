"""L4: control-flow equivalences (for-range <-> while, if/else <-> ternary).

Unlike L1-L3, these genuinely restructure code, so there is no single
generic equivalence check possible after the fact (see the for/while
docstring below for why). Correctness instead rests on: (1) narrow,
statically-checkable preconditions before a rewrite is ever attempted,
and (2) a rule-specific structural check on the *mutated* code, via
stdlib `ast`, that the rewritten shape actually matches what the rule
claims to have produced (catches implementation bugs -- wrong operator,
missing increment, mismatched body -- not a substitute for the
preconditions being sound in the first place).

Not handled in this version: range() with a step argument (the
while-loop's comparison direction would depend on the runtime sign of an
arbitrary expression), comprehension <-> loop (deferred, see README).
"""
from __future__ import annotations

import ast
import random
import textwrap

import libcst as cst
from libcst.metadata import MetadataWrapper, ParentNodeProvider

from .base import MutationResult
from .l2_rename import _all_identifiers, _fresh_name


# ---------------------------------------------------------------------------
# for-range -> while
# ---------------------------------------------------------------------------


def _range_bounds(iter_expr: cst.BaseExpression) -> tuple[str, str] | None:
    """(start_code, stop_code) for `range(stop)` / `range(start, stop)`
    with only positional args; None for anything else (3-arg step form,
    *args/**kwargs, or not a call to range at all)."""
    if not (isinstance(iter_expr, cst.Call) and isinstance(iter_expr.func, cst.Name) and iter_expr.func.value == "range"):
        return None
    args = iter_expr.args
    if any(a.star or a.keyword is not None for a in args):
        return None
    render = lambda node: cst.Module([]).code_for_node(node)
    if len(args) == 1:
        return "0", render(args[0].value)
    if len(args) == 2:
        return render(args[0].value), render(args[1].value)
    return None


def _has_own_continue(body: cst.BaseSuite) -> bool:
    """True if `body` contains a `continue` that refers to *this* loop --
    i.e. not one belonging to a nested loop or function, which the
    visitor prunes into (returning False from visit_X stops descent)."""
    found = False

    class _Visitor(cst.CSTVisitor):
        def visit_Continue(self, node: cst.Continue) -> None:
            nonlocal found
            found = True

        def visit_For(self, node: cst.For) -> bool:
            return False

        def visit_While(self, node: cst.While) -> bool:
            return False

        def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
            return False

        def visit_Lambda(self, node: cst.Lambda) -> bool:
            return False

    body.visit(_Visitor())
    return found


def _loop_var_reassigned(body: cst.BaseSuite, name: str) -> bool:
    """Conservative: True if `name` is written anywhere in body (regular
    assign, augmented/annotated assign, a nested for/comprehension
    reusing the same name, a `with ... as name`, a walrus, or
    global/nonlocal) -- any of these would make the while-loop's
    `i += 1` diverge from what the for-loop actually did with i."""
    found = False

    class _Visitor(cst.CSTVisitor):
        def visit_AssignTarget(self, node: cst.AssignTarget) -> None:
            self._check(node.target)

        def visit_AugAssign(self, node: cst.AugAssign) -> None:
            self._check(node.target)

        def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
            self._check(node.target)

        def visit_For(self, node: cst.For) -> None:
            self._check(node.target)

        def visit_CompFor(self, node: cst.CompFor) -> None:
            self._check(node.target)

        def visit_AsName(self, node: cst.AsName) -> None:
            self._check(node.name)

        def visit_NamedExpr(self, node: cst.NamedExpr) -> None:
            self._check(node.target)

        def visit_Global(self, node: cst.Global) -> None:
            nonlocal found
            if any(item.name.value == name for item in node.names):
                found = True

        def visit_Nonlocal(self, node: cst.Nonlocal) -> None:
            nonlocal found
            if any(item.name.value == name for item in node.names):
                found = True

        def _check(self, target: cst.BaseAssignTargetExpression | cst.BaseExpression) -> None:
            nonlocal found
            if isinstance(target, cst.Name) and target.value == name:
                found = True

    body.visit(_Visitor())
    return found


def _for_replacement(for_node: cst.For, bound_name: str, start_code: str, stop_code: str) -> list[cst.BaseStatement]:
    i_name = for_node.target.value
    init_stmt = cst.parse_statement(f"{i_name} = {start_code}\n")
    bound_stmt = cst.parse_statement(f"{bound_name} = {stop_code}\n")
    increment_stmt = cst.parse_statement(f"{i_name} += 1\n")
    new_body = for_node.body.with_changes(body=list(for_node.body.body) + [increment_stmt])
    while_stmt = cst.parse_statement(f"while {i_name} < {bound_name}:\n    pass\n").with_changes(body=new_body)
    return [init_stmt, bound_stmt, while_stmt]


class _ForCandidate:
    __slots__ = ("for_id", "bound_name", "start_code", "stop_code", "i_name", "orig_body_dumps")

    def __init__(self, for_id, bound_name, start_code, stop_code, i_name, orig_body_dumps):
        self.for_id = for_id
        self.bound_name = bound_name
        self.start_code = start_code
        self.stop_code = stop_code
        self.i_name = i_name
        self.orig_body_dumps = orig_body_dumps


def _scan_for_loops(module: cst.Module, rng: random.Random) -> list[_ForCandidate]:
    used = _all_identifiers(module)
    candidates: list[_ForCandidate] = []

    class _Visitor(cst.CSTVisitor):
        def visit_For(self, node: cst.For) -> None:
            if node.orelse is not None or not isinstance(node.target, cst.Name):
                return
            if not isinstance(node.body, cst.IndentedBlock):
                # single-line `for i in range(n): body` has a
                # SimpleStatementSuite body instead, whose .body holds
                # BaseSmallStatement items rather than SimpleStatementLine
                # -- appending the increment the way we do below would
                # nest a SimpleStatementLine where one isn't valid (found
                # by testing, not by inspection). Rare style, skip it.
                return
            bounds = _range_bounds(node.iter)
            if bounds is None:
                return
            if _has_own_continue(node.body) or _loop_var_reassigned(node.body, node.target.value):
                return
            # independent ast-side precondition: the body must itself be
            # parseable in isolation (always true for a valid CST body,
            # this doubles as a sanity net) and yield one ast.stmt per
            # line for the post-hoc dump comparison below.
            try:
                dedented = textwrap.dedent(cst.Module([]).code_for_node(node.body))
                body_dumps = [ast.dump(s) for s in ast.parse(dedented).body]
            except SyntaxError:
                return
            start_code, stop_code = bounds
            bound_name = _fresh_name(used | {node.target.value}, rng)
            candidates.append(_ForCandidate(id(node), bound_name, start_code, stop_code, node.target.value, body_dumps))

    module.visit(_Visitor())
    return candidates


class _ForReplacer(cst.CSTTransformer):
    def __init__(self, candidate: _ForCandidate):
        self.candidate = candidate

    def leave_For(self, original_node: cst.For, updated_node: cst.For):
        if id(original_node) != self.candidate.for_id:
            return updated_node
        replacement = _for_replacement(updated_node, self.candidate.bound_name, self.candidate.start_code, self.candidate.stop_code)
        return cst.FlattenSentinel(replacement)


def _verify_for_to_while(mutated_code: str, candidate: _ForCandidate) -> bool:
    """Independent (ast, not libcst) structural check: somewhere in the
    mutated code there must be a `while i < bound:` whose body is exactly
    [original for-body statements..., i += 1] -- bound_name is a freshly
    generated identifier guaranteed not to appear anywhere in the
    original source, so a match on it is unambiguous."""
    try:
        tree = ast.parse(mutated_code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.While):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == candidate.i_name
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Lt)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Name)
            and test.comparators[0].id == candidate.bound_name
        ):
            continue
        if len(node.body) != len(candidate.orig_body_dumps) + 1:
            continue
        *body_stmts, last = node.body
        if not (
            isinstance(last, ast.AugAssign)
            and isinstance(last.target, ast.Name)
            and last.target.id == candidate.i_name
            and isinstance(last.op, ast.Add)
            and isinstance(last.value, ast.Constant)
            and last.value.value == 1
        ):
            continue
        if [ast.dump(s) for s in body_stmts] == candidate.orig_body_dumps:
            return True
    return False


def for_to_while(code: str, rng: random.Random) -> MutationResult:
    try:
        module = cst.parse_module(code)
    except cst.ParserSyntaxError:
        return MutationResult("L4", "unparseable", code, False)

    candidates = _scan_for_loops(module, rng)
    if not candidates:
        return MutationResult("L4", "noop", code, False)

    target = rng.choice(candidates)
    new_code = module.visit(_ForReplacer(target)).code
    if new_code == code:
        return MutationResult("L4", "noop", code, False)
    if not _verify_for_to_while(new_code, target):
        return MutationResult("L4", "rejected:for_while_mismatch", code, False)

    return MutationResult("L4", "for_to_while", new_code, True)


# ---------------------------------------------------------------------------
# if/else -> ternary
# ---------------------------------------------------------------------------


def _single_assign(suite: cst.BaseSuite) -> cst.Assign | None:
    if not isinstance(suite, cst.IndentedBlock) or len(suite.body) != 1:
        return None
    line = suite.body[0]
    if not (isinstance(line, cst.SimpleStatementLine) and len(line.body) == 1):
        return None
    stmt = line.body[0]
    if isinstance(stmt, cst.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0].target, cst.Name):
        return stmt
    return None


class _IfCandidate:
    __slots__ = ("if_id", "target_name", "cond_code", "a_code", "b_code", "cond_dump", "a_dump", "b_dump", "is_elif_position")

    def __init__(self, if_id, target_name, cond_code, a_code, b_code, cond_dump, a_dump, b_dump, is_elif_position):
        self.if_id = if_id
        self.target_name = target_name
        self.cond_code = cond_code
        self.a_code = a_code
        self.b_code = b_code
        self.cond_dump = cond_dump
        self.a_dump = a_dump
        self.b_dump = b_dump
        self.is_elif_position = is_elif_position


def _scan_if_else(module: cst.Module) -> list[_IfCandidate]:
    # Needed for is_elif_position below: an `elif` branch is itself an If
    # node sitting directly in its parent If's `orelse` slot (not wrapped
    # in an Else). Replacing such a node with a bare statement -- valid
    # when an If sits in an ordinary body list -- produces a malformed
    # tree here, since the parent's orelse slot expects If | Else | None.
    # See _IfReplacer for the fix this metadata enables.
    wrapper = MetadataWrapper(module, unsafe_skip_copy=True)
    parents = wrapper.resolve(ParentNodeProvider)

    candidates: list[_IfCandidate] = []
    render = lambda node: cst.Module([]).code_for_node(node)

    class _Visitor(cst.CSTVisitor):
        def visit_If(self, node: cst.If) -> None:
            # If.orelse is Else for a plain `else:`, but for `elif ...:`
            # it's a *nested If* directly (not wrapped in Else) -- and
            # If also happens to have a `.body` attribute, so checking
            # the type here isn't optional: without it, node.orelse.body
            # silently resolves to the elif's own then-branch, discarding
            # its condition entirely (found by testing, not by inspection).
            if not isinstance(node.orelse, cst.Else):
                return
            then_assign = _single_assign(node.body)
            else_assign = _single_assign(node.orelse.body)
            if then_assign is None or else_assign is None:
                return
            then_target = then_assign.targets[0].target.value
            else_target = else_assign.targets[0].target.value
            if then_target != else_target:
                return
            try:
                cond_dump = ast.dump(ast.parse(render(node.test), mode="eval").body)
                a_dump = ast.dump(ast.parse(render(then_assign.value), mode="eval").body)
                b_dump = ast.dump(ast.parse(render(else_assign.value), mode="eval").body)
            except SyntaxError:
                return
            parent = parents.get(node)
            is_elif_position = isinstance(parent, cst.If) and parent.orelse is node
            candidates.append(
                _IfCandidate(
                    id(node), then_target, render(node.test), render(then_assign.value), render(else_assign.value),
                    cond_dump, a_dump, b_dump, is_elif_position,
                )
            )

    wrapper.module.visit(_Visitor())
    return candidates


class _IfReplacer(cst.CSTTransformer):
    def __init__(self, candidate: _IfCandidate):
        self.candidate = candidate

    def leave_If(self, original_node: cst.If, updated_node: cst.If):
        if id(original_node) != self.candidate.if_id:
            return updated_node
        c = self.candidate
        assign_stmt = cst.parse_statement(f"{c.target_name} = {c.a_code} if {c.cond_code} else {c.b_code}\n")
        if c.is_elif_position:
            # this If is itself the elif branch of some other If -- the
            # parent's orelse slot needs an Else (or If, or None), not a
            # bare statement, to stay structurally valid.
            return cst.Else(body=cst.IndentedBlock(body=[assign_stmt]))
        return assign_stmt


def _verify_if_to_ternary(mutated_code: str, candidate: _IfCandidate) -> bool:
    try:
        tree = ast.parse(mutated_code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == candidate.target_name
            and isinstance(node.value, ast.IfExp)
        ):
            continue
        ifexp = node.value
        if (
            ast.dump(ifexp.test) == candidate.cond_dump
            and ast.dump(ifexp.body) == candidate.a_dump
            and ast.dump(ifexp.orelse) == candidate.b_dump
        ):
            return True
    return False


def if_else_to_ternary(code: str, rng: random.Random) -> MutationResult:
    try:
        module = cst.parse_module(code)
    except cst.ParserSyntaxError:
        return MutationResult("L4", "unparseable", code, False)

    candidates = _scan_if_else(module)
    if not candidates:
        return MutationResult("L4", "noop", code, False)

    target = rng.choice(candidates)
    new_code = module.visit(_IfReplacer(target)).code
    if new_code == code:
        return MutationResult("L4", "noop", code, False)
    if not _verify_if_to_ternary(new_code, target):
        return MutationResult("L4", "rejected:ternary_mismatch", code, False)

    return MutationResult("L4", "if_else_to_ternary", new_code, True)


# ---------------------------------------------------------------------------


def apply_random(code: str, rng: random.Random) -> MutationResult:
    rules = [for_to_while, if_else_to_ternary]
    rng.shuffle(rules)
    for rule in rules:
        result = rule(code, rng)
        if result.applied:
            return result
    return MutationResult("L4", "noop", code, False)
