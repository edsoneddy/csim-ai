"""Tests the Hugging Face Hub auto-download path (`Scorer()` with no
model_path). Requires network access to huggingface.co -- unlike
test_inference.py, this doesn't skip based on local artifacts, since the
whole point is to not need any.
"""
from __future__ import annotations

from csim_ai import Scorer

DUPLICATE_A = "def add(a, b):\n    return a + b\n"
DUPLICATE_B = "def add(x, y):\n    # add two numbers\n    return x + y\n"
UNRELATED_A = "def add(a, b):\n    return a + b\n"
UNRELATED_B = "import sys\nfor line in sys.stdin:\n    print(line.strip()[::-1])\n"


def test_auto_download_full_scoring():
    scorer = Scorer(use_fusion=True)
    dup = scorer.score(DUPLICATE_A, DUPLICATE_B)
    unrelated = scorer.score(UNRELATED_A, UNRELATED_B)

    assert dup["fusion"] > unrelated["fusion"]
    assert dup["fusion"] > 0.5
    assert unrelated["fusion"] < 0.5
