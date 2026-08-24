"""L1: cosmetic-only mutations (comments, docstrings, indentation width).

Every rule here only touches trivia (comments/whitespace) or removes a
bare string-literal statement (a no-op at runtime) -- never anything that
changes what `ast.parse` sees beyond that. validate.cosmetic_equivalent is
the ground truth these are checked against downstream.
"""
from __future__ import annotations

import random

import libcst as cst

from .base import MutationResult


class _StripDocstrings(cst.CSTTransformer):
    """Removes the docstring-position statement from Module/FunctionDef/
    ClassDef bodies. Skips a def/class whose docstring is its *only*
    statement (would leave an empty suite -- invalid Python) and skips
    one-line suites (e.g. a docstring-only one-liner def, which is a
    SimpleStatementSuite rather than an IndentedBlock) since they're rare
    and not worth the extra shape-handling. Module bodies are a plain
    statement list and can be empty, so those are always stripped."""

    def leave_Module(self, original_node, updated_node):
        stmts = updated_node.body
        if not stmts or not self._is_docstring_stmt(stmts[0]):
            return updated_node
        return updated_node.with_changes(body=stmts[1:])

    def _strip_suite_owner(self, node: cst.FunctionDef | cst.ClassDef):
        body = node.body
        if not isinstance(body, cst.IndentedBlock):
            return node
        stmts = body.body
        if not stmts or not self._is_docstring_stmt(stmts[0]) or len(stmts) == 1:
            return node
        return node.with_changes(body=body.with_changes(body=stmts[1:]))

    def leave_FunctionDef(self, original_node, updated_node):
        return self._strip_suite_owner(updated_node)

    def leave_ClassDef(self, original_node, updated_node):
        return self._strip_suite_owner(updated_node)

    @staticmethod
    def _is_docstring_stmt(stmt: cst.BaseStatement) -> bool:
        return (
            isinstance(stmt, cst.SimpleStatementLine)
            and len(stmt.body) == 1
            and isinstance(stmt.body[0], cst.Expr)
            and isinstance(stmt.body[0].value, (cst.SimpleString, cst.ConcatenatedString))
        )


class _StripComments(cst.CSTTransformer):
    def leave_TrailingWhitespace(self, original_node, updated_node):
        if updated_node.comment is not None:
            return updated_node.with_changes(comment=None)
        return updated_node

    def leave_EmptyLine(self, original_node, updated_node):
        if updated_node.comment is not None:
            return updated_node.with_changes(comment=None)
        return updated_node


class _Reindent(cst.CSTTransformer):
    def __init__(self, indent: str):
        self.indent = indent

    def leave_IndentedBlock(self, original_node, updated_node):
        return updated_node.with_changes(indent=self.indent)


def strip_docstrings(code: str) -> MutationResult:
    tree = cst.parse_module(code)
    new_tree = tree.visit(_StripDocstrings())
    new_code = new_tree.code
    return MutationResult("L1", "strip_docstrings", new_code, new_code != code)


def strip_comments(code: str) -> MutationResult:
    tree = cst.parse_module(code)
    new_tree = tree.visit(_StripComments())
    new_code = new_tree.code
    return MutationResult("L1", "strip_comments", new_code, new_code != code)


def reindent(code: str, rng: random.Random) -> MutationResult:
    width = rng.choice([2, 3, 8])
    tree = cst.parse_module(code)
    new_tree = tree.visit(_Reindent(" " * width))
    new_code = new_tree.code
    return MutationResult("L1", f"reindent_{width}", new_code, new_code != code)


def apply_random(code: str, rng: random.Random) -> MutationResult:
    """Apply a random non-empty subset of the L1 rules, in a random order."""
    rules = [strip_docstrings, strip_comments, lambda c: reindent(c, rng)]
    chosen = rng.sample(rules, k=rng.randint(1, len(rules)))
    cur = code
    fired = []
    for rule in chosen:
        result = rule(cur)
        if result.applied and result.code:
            cur = result.code
            fired.append(result.rule)
    return MutationResult("L1", "+".join(fired) if fired else "noop", cur, bool(fired))
