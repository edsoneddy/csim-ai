"""L5: inline simple single-expression functions.

Only the narrowest, safely-decidable case: a function whose entire body
is one `return EXPR` statement, with plain positional-or-keyword
parameters (no defaults, no *args/**kwargs/keyword-only, no decorators,
not recursive), called with exactly matching positional arguments (no
keywords, no star-unpacking). Each parameter is substituted into a copy
of EXPR at the call site.

Two hazards make this narrower than a naive find-and-replace:

1. **Duplicate evaluation.** A parameter used more than once in EXPR can
   only be inlined if its argument is a bare Name or literal Constant --
   duplicating a Name/Constant reference has no observable effect, but
   duplicating a Call or other expression would evaluate it multiple
   times instead of once (and could reorder side effects). Each call site
   is checked independently, so the same function can be safely inlined
   at one call site and skipped at another depending on what's passed.
2. **Shadowing.** If EXPR contains a nested lambda or comprehension that
   rebinds a parameter name as its own loop/lambda variable (e.g.
   `def f(x): return [x for x in range(x)]` -- the `x` before `for`
   refers to the comprehension's own x, not the parameter), a blind
   textual substitution would incorrectly rewrite that unrelated bound
   name too. Any such collision excludes the whole function.

Not handled in this version: multi-statement function bodies, extracting
a block into a new function (the reverse direction), container-type swaps
(list<->deque, dict<->defaultdict) and accumulator-pattern rewrites --
each would need its own careful precondition design the same way
for_to_while/if_else_to_ternary did in L4, and the roadmap's >=50k target
for Fase 1 is already well past without them.
"""
from __future__ import annotations

import ast
import random

import libcst as cst

from .base import MutationResult


class _InlinableFunction:
    __slots__ = ("name", "params", "body_code", "body_ast")

    def __init__(self, name: str, params: list[str], body_code: str, body_ast: ast.expr):
        self.name = name
        self.params = params
        self.body_code = body_code
        self.body_ast = body_ast


def _shadows_any_param(body_ast: ast.expr, params: set[str]) -> bool:
    for node in ast.walk(body_ast):
        if isinstance(node, ast.Lambda):
            bound = {a.arg for a in node.args.args + node.args.posonlyargs + node.args.kwonlyargs}
            if bound & params:
                return True
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for gen in node.generators:
                bound = {n.id for n in ast.walk(gen.target) if isinstance(n, ast.Name)}
                if bound & params:
                    return True
    return False


def _find_inlinable_functions(module: cst.Module) -> dict[str, _InlinableFunction]:
    render = lambda node: cst.Module([]).code_for_node(node)
    found: dict[str, _InlinableFunction] = {}

    class _Visitor(cst.CSTVisitor):
        def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
            if node.decorators:
                return
            p = node.params
            if p.star_arg is not cst.MaybeSentinel.DEFAULT or p.star_kwarg is not None:
                return
            if p.posonly_params or p.kwonly_params:
                return
            if any(param.default is not None for param in p.params):
                return
            if not isinstance(node.body, cst.IndentedBlock) or len(node.body.body) != 1:
                return
            line = node.body.body[0]
            if not (isinstance(line, cst.SimpleStatementLine) and len(line.body) == 1):
                return
            stmt = line.body[0]
            if not (isinstance(stmt, cst.Return) and stmt.value is not None):
                return
            param_names = [param.name.value for param in p.params]
            body_code = render(stmt.value)
            try:
                body_ast = ast.parse(body_code, mode="eval").body
            except SyntaxError:
                return
            # recursive functions can't be inlined (nothing to unroll to)
            if any(isinstance(n, ast.Name) and n.id == node.name.value for n in ast.walk(body_ast)):
                return
            if _shadows_any_param(body_ast, set(param_names)):
                return
            found[node.name.value] = _InlinableFunction(node.name.value, param_names, body_code, body_ast)

    module.visit(_Visitor())
    return found


def _name_use_count(body_ast: ast.expr, name: str) -> int:
    return sum(1 for n in ast.walk(body_ast) if isinstance(n, ast.Name) and n.id == name)


def _substitute(body_ast: ast.expr, mapping: dict[str, ast.expr]) -> ast.expr:
    """Returns a copy of body_ast with each Name(id=param) replaced by
    the corresponding argument subtree from `mapping`."""
    import copy

    class _Sub(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name):
            return mapping.get(node.id, node)

    return _Sub().visit(copy.deepcopy(body_ast))


class _CallCandidate:
    __slots__ = ("call_id", "func_name", "expected_dump")

    def __init__(self, call_id: int, func_name: str, expected_dump: str):
        self.call_id = call_id
        self.func_name = func_name
        self.expected_dump = expected_dump


def _scan_call_sites(module: cst.Module, functions: dict[str, _InlinableFunction]) -> list[_CallCandidate]:
    render = lambda node: cst.Module([]).code_for_node(node)
    candidates: list[_CallCandidate] = []

    class _Visitor(cst.CSTVisitor):
        def visit_Call(self, node: cst.Call) -> None:
            if not isinstance(node.func, cst.Name):
                return
            fn = functions.get(node.func.value)
            if fn is None:
                return
            args = node.args
            if len(args) != len(fn.params) or any(a.star or a.keyword is not None for a in args):
                return
            try:
                arg_asts = [ast.parse(render(a.value), mode="eval").body for a in args]
            except SyntaxError:
                return

            mapping: dict[str, ast.expr] = {}
            for param, arg_node in zip(fn.params, arg_asts):
                use_count = _name_use_count(fn.body_ast, param)
                is_duplicable = isinstance(arg_node, (ast.Name, ast.Constant))
                if use_count > 1 and not is_duplicable:
                    return  # this call site can't safely inline this function
                mapping[param] = arg_node

            substituted = _substitute(fn.body_ast, mapping)
            expected_dump = ast.dump(substituted)
            candidates.append(_CallCandidate(id(node), fn.name, expected_dump))

    module.visit(_Visitor())
    return candidates


def _render_inlined_call(fn: _InlinableFunction, args: list[cst.Arg]) -> str:
    # Renders the substituted call by re-deriving it from source text:
    # reuse the function body's own code with each parameter Name
    # textually replaced by its parenthesized argument code. Simpler and
    # less error-prone than reconstructing a CST expression tree by hand;
    # _verify_inline (independent, ast-based) is what actually guarantees
    # correctness, not this rendering step.
    render = lambda node: cst.Module([]).code_for_node(node)
    mapping = {param: f"({render(arg.value)})" for param, arg in zip(fn.params, args)}
    return f"({_replace_identifiers(fn.body_code, mapping)})"


def _replace_identifiers(code: str, mapping: dict[str, str]) -> str:
    """Token-aware, single-pass simultaneous replace: every NAME token
    matching a key in `mapping` is substituted. This has to happen in one
    pass over the *original* code, not one `str.replace`-style pass per
    parameter -- a sequential per-parameter substitution can re-scan text
    an earlier substitution just inserted and corrupt an unrelated
    identifier that happens to share another parameter's name (e.g.
    `agregar_digitos(num, eliminar_digitos(str(n)))` inlining `def
    agregar_digitos(n, digitos): return str(n) + digitos` -- the second
    argument's own unrelated `n` got clobbered by the substitution meant
    for the first parameter, when done as two sequential passes; found on
    the real dataset, not by inspection)."""
    import io
    import tokenize

    out = []
    last_end = (1, 0)
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    lines = code.splitlines(keepends=True) or [code]
    for tok in tokens:
        if tok.type == tokenize.NAME and tok.string in mapping:
            out.append(_slice_between(lines, last_end, tok.start))
            out.append(mapping[tok.string])
            last_end = tok.end
    end_of_text = (len(lines), len(lines[-1])) if lines else (1, 0)
    out.append(_slice_between(lines, last_end, end_of_text))
    return "".join(out)


def _slice_between(lines: list[str], start: tuple[int, int], end: tuple[int, int]) -> str:
    if start[0] == end[0]:
        return lines[start[0] - 1][start[1] : end[1]] if start[0] - 1 < len(lines) else ""
    parts = [lines[start[0] - 1][start[1] :]]
    parts.extend(lines[start[0] : end[0] - 1])
    if end[0] - 1 < len(lines):
        parts.append(lines[end[0] - 1][: end[1]])
    return "".join(parts)


def _verify_inline(mutated_code: str, candidate: _CallCandidate) -> bool:
    try:
        tree = ast.parse(mutated_code)
    except SyntaxError:
        return False
    return any(ast.dump(node) == candidate.expected_dump for node in ast.walk(tree) if isinstance(node, ast.expr))


def apply_random(code: str, rng: random.Random) -> MutationResult:
    try:
        module = cst.parse_module(code)
    except cst.ParserSyntaxError:
        return MutationResult("L5", "unparseable", code, False)

    functions = _find_inlinable_functions(module)
    if not functions:
        return MutationResult("L5", "noop", code, False)
    candidates = _scan_call_sites(module, functions)
    if not candidates:
        return MutationResult("L5", "noop", code, False)

    target = rng.choice(candidates)
    fn = functions[target.func_name]

    class _Replacer(cst.CSTTransformer):
        def leave_Call(self, original_node: cst.Call, updated_node: cst.Call):
            if id(original_node) != target.call_id:
                return updated_node
            rendered = _render_inlined_call(fn, list(updated_node.args))
            return cst.parse_expression(rendered)

    new_code = module.visit(_Replacer()).code
    if new_code == code:
        return MutationResult("L5", "noop", code, False)
    if not _verify_inline(new_code, target):
        return MutationResult("L5", "rejected:inline_mismatch", code, False)

    return MutationResult("L5", f"inline:{target.func_name}", new_code, True)
