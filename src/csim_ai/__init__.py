"""csim-ai: neural-augmented Python code plagiarism detection for
programming judges.

Base install (`pip install csim-ai`) gives bi-encoder cosine similarity
only, via onnxruntime + tokenizers (no torch). `pip install
csim-ai[ast,scorer]` adds the csim TED signal and the GBDT fusion of
both -- the full hybrid scorer from Fase 4.

Model weights aren't bundled (the ONNX export is ~500MB) -- point
`Scorer` at a local export from `training/export_onnx.py` via
`model_path`.
"""
from __future__ import annotations

from pathlib import Path

from ._onnx_encoder import OnnxEncoder

__version__ = "0.0.1"

__all__ = ["Scorer", "__version__"]


class Scorer:
    def __init__(self, model_path: str | Path, fusion_model_path: str | Path | None = None):
        self._encoder = OnnxEncoder(model_path)
        self._fusion = None
        if fusion_model_path is not None:
            from ._fusion import FusionModel

            self._fusion = FusionModel(fusion_model_path)

    def score(self, code_a: str, code_b: str) -> dict:
        biencoder_cosine = self._encoder.cosine_similarity(code_a, code_b)

        csim_ted = None
        try:
            from ._ted import ted_score

            csim_ted = ted_score(code_a, code_b)
        except ImportError:
            pass

        fusion = None
        if self._fusion is not None and csim_ted is not None:
            fusion = self._fusion.predict(biencoder_cosine, csim_ted)

        return {"biencoder_cosine": biencoder_cosine, "csim_ted": csim_ted, "fusion": fusion}
