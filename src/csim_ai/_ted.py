"""csim TED (tree-edit-distance) signal -- requires the `ast` extra
(`pip install csim-ai[ast]`). Imported lazily by `Scorer` so the base
install doesn't need `csim`.

`preprocess`/`similarity` are split so a caller comparing one file
against many others (e.g. the CLI's `report`/`group`, all-pairs over a
directory) can preprocess each file once and reuse the tree, instead of
re-parsing it for every pair it appears in.
"""
from __future__ import annotations

import contextlib
import io
from typing import Any

LANG = "python_3_13"
TED_ALGORITHM = "apted"


def preprocess(code: str) -> Any | None:
    # csim.utils.preprocess_code does *not* raise on a syntax error --
    # its ANTLR grammar prints "Syntax error ..." to stderr and returns a
    # degenerate near-empty tree instead, which would otherwise silently
    # produce a garbage similarity score. Capture stderr and treat any
    # output as a failed parse (same fix as training/scorer/build_features.py).
    from csim.utils import preprocess_code

    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf):
            tree = preprocess_code("x", code, LANG)
            if buf.getvalue():
                return None
            return tree
    except Exception:
        return None


def similarity(tree_a: Any, tree_b: Any) -> float | None:
    from csim.utils import get_similarity_coefficient

    try:
        return get_similarity_coefficient(tree_a, tree_b, TED_ALGORITHM)
    except Exception:
        return None


def ted_score(code_a: str, code_b: str) -> float | None:
    tree_a = preprocess(code_a)
    tree_b = preprocess(code_b)
    if tree_a is None or tree_b is None:
        return None
    return similarity(tree_a, tree_b)
