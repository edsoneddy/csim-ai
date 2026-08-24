"""L3: reorder independent statements (imports, side-effect-free simple
assignments).

Two families are handled, each restricted to a shape narrow enough that
independence is decidable purely by inspecting the statements, without
executing anything: contiguous runs of plain import statements with no
bound-name collisions, and contiguous runs of single-target assignments
that are call-free (no observable side effect whose order could matter --
input(), print(), any function/method call all count) and have
pairwise-disjoint read/write name sets. Both properties are re-derived
independently in validate.py via stdlib `ast` before a mutation is ever
reported as applied, rather than trusted from the libcst-based scan here
(see l2_rename's module docstring for why that independence mattered in
practice -- the same lesson applies).

Not handled in this version: reordering function/class defs, and
assignments with tuple/attribute/subscript targets or augmented
assignment -- narrower shapes were enough to get real coverage without
the extra risk of getting definition-time evaluation order wrong.
"""
from __future__ import annotations

import random
from typing import Callable, Sequence

import libcst as cst

from . import validate
from .base import MutationResult


def _stmt_code(stmt: cst.SimpleStatementLine) -> str:
    return cst.Module(body=[stmt]).code


def _is_plain_import_line(stmt: cst.BaseStatement) -> bool:
    return (
        isinstance(stmt, cst.SimpleStatementLine)
        and len(stmt.body) == 1
        and isinstance(stmt.body[0], (cst.Import, cst.ImportFrom))
    )


def _is_simple_assign_line(stmt: cst.BaseStatement) -> bool:
    if not (isinstance(stmt, cst.SimpleStatementLine) and len(stmt.body) == 1):
        return False
    small = stmt.body[0]
    return (
        isinstance(small, cst.Assign)
        and len(small.targets) == 1
        and isinstance(small.targets[0].target, cst.Name)
    )


def _maximal_runs(stmts: Sequence[cst.BaseStatement], predicate: Callable[[cst.BaseStatement], bool]) -> list[tuple[int, int]]:
    """Index ranges [start, end) of maximal consecutive runs (length >= 2)
    of statements satisfying predicate."""
    runs = []
    start = None
    padded = list(stmts) + [None]
    for i, stmt in enumerate(padded):
        if stmt is not None and predicate(stmt):
            if start is None:
                start = i
        else:
            if start is not None and i - start >= 2:
                runs.append((start, i))
            start = None
    return runs


class _Candidate:
    __slots__ = ("block_id", "kind", "start", "end")

    def __init__(self, block_id: int, kind: str, start: int, end: int):
        self.block_id = block_id
        self.kind = kind
        self.start = start
        self.end = end


def _scan(module: cst.Module) -> list[_Candidate]:
    candidates: list[_Candidate] = []

    def _windows(node_id: int, kind: str, body, run_start: int, run_end: int, checker) -> None:
        # A run of shape-matching statements can still contain a subset
        # that isn't mutually independent (e.g. `a = 1; b = 2; c = a + b`
        # -- all three are shape-matching assigns, but only the first two
        # are independent of each other). Try every sub-window rather than
        # only the full maximal run, so one dependent/impure line doesn't
        # sink an otherwise-safe pair sitting right next to it.
        for i in range(run_start, run_end):
            for j in range(i + 2, run_end + 1):
                snippets = [_stmt_code(body[k]) for k in range(i, j)]
                if checker(snippets):
                    candidates.append(_Candidate(node_id, kind, i, j))

    class _Visitor(cst.CSTVisitor):
        def _check_body(self, node: cst.Module | cst.IndentedBlock) -> None:
            body = node.body
            for start, end in _maximal_runs(body, _is_plain_import_line):
                _windows(id(node), "import", body, start, end, validate.imports_mutually_independent)
            for start, end in _maximal_runs(body, _is_simple_assign_line):
                _windows(id(node), "assign", body, start, end, validate.assignments_mutually_independent)

        def visit_Module(self, node: cst.Module) -> None:
            self._check_body(node)

        def visit_IndentedBlock(self, node: cst.IndentedBlock) -> None:
            self._check_body(node)

    module.visit(_Visitor())
    return candidates


class _Reorderer(cst.CSTTransformer):
    def __init__(self, target: _Candidate, permutation: list[int]):
        self.target = target
        self.permutation = permutation

    def _maybe_reorder(self, original_node, updated_node):
        if id(original_node) != self.target.block_id:
            return updated_node
        body = list(updated_node.body)
        window = body[self.target.start : self.target.end]
        reordered = [window[i] for i in self.permutation]
        new_body = body[: self.target.start] + reordered + body[self.target.end :]
        return updated_node.with_changes(body=new_body)

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module):
        return self._maybe_reorder(original_node, updated_node)

    def leave_IndentedBlock(self, original_node: cst.IndentedBlock, updated_node: cst.IndentedBlock):
        return self._maybe_reorder(original_node, updated_node)


def apply_random(code: str, rng: random.Random) -> MutationResult:
    """Picks one random eligible run and applies a random non-identity
    permutation to it. Self-verifies (order_equivalent) before reporting
    success; on failure returns applied=False (caller tries again)."""
    try:
        module = cst.parse_module(code)
    except cst.ParserSyntaxError:
        return MutationResult("L3", "unparseable", code, False)

    candidates = _scan(module)
    if not candidates:
        return MutationResult("L3", "noop", code, False)

    target = rng.choice(candidates)
    n = target.end - target.start
    perm = list(range(n))
    while perm == list(range(n)):
        rng.shuffle(perm)

    new_code = module.visit(_Reorderer(target, perm)).code
    if new_code == code:
        return MutationResult("L3", "noop", code, False)
    if not validate.order_equivalent(code, new_code):
        return MutationResult("L3", "rejected:order_mismatch", code, False)

    return MutationResult("L3", f"reorder_{target.kind}:{n}", new_code, True)
