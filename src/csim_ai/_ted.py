"""csim TED (tree-edit-distance) signal -- requires the `ast` extra
(`pip install csim-ai[ast]`). Imported lazily by `Scorer` so the base
install doesn't need `csim`.
"""
from __future__ import annotations

import contextlib
import io

LANG = "python_3_13"
TED_ALGORITHM = "apted"


def ted_score(code_a: str, code_b: str) -> float | None:
    # csim.utils.preprocess_code does *not* raise on a syntax error --
    # its ANTLR grammar prints "Syntax error ..." to stderr and returns a
    # degenerate near-empty tree instead, which would otherwise silently
    # produce a garbage similarity score. Capture stderr and treat any
    # output as a failed parse (same fix as training/scorer/build_features.py).
    from csim.utils import get_similarity_coefficient, preprocess_code

    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf):
            proc_a = preprocess_code("a", code_a, LANG)
            proc_b = preprocess_code("b", code_b, LANG)
            if buf.getvalue():
                return None
            return get_similarity_coefficient(proc_a, proc_b, TED_ALGORITHM)
    except Exception:
        return None
