"""csim-ai: neural-augmented Python code plagiarism detection for
programming judges.

Base install (`pip install csim-ai`) gives bi-encoder cosine similarity
only, via onnxruntime + tokenizers + huggingface_hub (no torch).
`pip install csim-ai[ast,scorer]` adds the csim TED signal and the GBDT
fusion of both -- the full hybrid scorer from Fase 4.

Model weights aren't bundled in the package (the ONNX export is
~500MB) -- `Scorer()` with no arguments downloads the pre-trained
bi-encoder from Hugging Face Hub (`edson-eddy/csim-ai`) on first use and
caches it there after; pass `model_path` to use a local export instead
(from `training/export_onnx.py` or `csim-ai setup --export-from`).

Recommended flow: `pip install csim-ai[ast,scorer]` then `csim-ai
setup` once (downloads both the bi-encoder and the fusion model) --
after that, `Scorer()` with no arguments auto-detects the cached fusion
model and gives the full hybrid score with no further flags/arguments,
same as the CLI's `report`/`group` without `--use-fusion`. Pass
`use_fusion=True` to force-download the fusion model on the spot instead
of requiring `setup` first.
"""
from __future__ import annotations

from pathlib import Path

from ._onnx_encoder import OnnxEncoder

__version__ = "0.0.1"

__all__ = ["Scorer", "__version__"]


class Scorer:
    def __init__(
        self,
        model_path: str | Path | None = None,
        fusion_model_path: str | Path | None = None,
        use_fusion: bool = False,
    ):
        if model_path is None:
            from ._hub import download_model

            model_path = download_model()
        self._encoder = OnnxEncoder(model_path)

        if fusion_model_path is None:
            if use_fusion:
                from ._hub import download_fusion_model

                fusion_model_path = download_fusion_model()
            else:
                # Auto-use the fusion model if a prior `csim-ai setup` (or
                # an earlier use_fusion=True call) already cached it --
                # matches it being available without forcing a network
                # request just to check.
                from ._hub import cached_fusion_model

                fusion_model_path = cached_fusion_model()

        self._fusion = None
        if fusion_model_path is not None:
            try:
                from ._fusion import FusionModel

                self._fusion = FusionModel(fusion_model_path)
            except ImportError:
                pass  # scikit-learn not installed -- degrade to bi-encoder-only

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
