"""Fase 5 inference smoke tests. Skips when the gitignored local
artifacts (ONNX export, fusion model) aren't present -- they're
regenerated via `training/export_onnx.py` and
`training/scorer/train_fusion.py`, not part of a fresh clone.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from csim_ai import Scorer

ROOT = Path(__file__).parent.parent
ONNX_MODEL_DIR = ROOT / "training" / "artifacts" / "onnx_model"
FUSION_MODEL_PATH = ROOT / "training" / "scorer" / "artifacts" / "fusion_model_v1.joblib"

pytestmark = pytest.mark.skipif(
    not (ONNX_MODEL_DIR / "model.onnx").exists(),
    reason="ONNX model not exported locally -- run training/export_onnx.py first",
)

DUPLICATE_A = "def add(a, b):\n    return a + b\n"
DUPLICATE_B = "def add(x, y):\n    # add two numbers\n    return x + y\n"
UNRELATED_A = "def add(a, b):\n    return a + b\n"
UNRELATED_B = "import sys\nfor line in sys.stdin:\n    print(line.strip()[::-1])\n"


def test_biencoder_only_scoring():
    scorer = Scorer(ONNX_MODEL_DIR)
    dup = scorer.score(DUPLICATE_A, DUPLICATE_B)
    unrelated = scorer.score(UNRELATED_A, UNRELATED_B)

    assert dup["fusion"] is None  # no fusion_model_path given
    assert dup["biencoder_cosine"] > unrelated["biencoder_cosine"]
    assert dup["biencoder_cosine"] > 0.8


@pytest.mark.skipif(not FUSION_MODEL_PATH.exists(), reason="fusion model not trained locally")
def test_fusion_scoring():
    scorer = Scorer(ONNX_MODEL_DIR, fusion_model_path=FUSION_MODEL_PATH)
    dup = scorer.score(DUPLICATE_A, DUPLICATE_B)
    unrelated = scorer.score(UNRELATED_A, UNRELATED_B)

    assert dup["fusion"] is not None
    assert dup["fusion"] > unrelated["fusion"]
    assert dup["fusion"] > 0.5
    assert unrelated["fusion"] < 0.5
