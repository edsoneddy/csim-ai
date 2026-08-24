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
