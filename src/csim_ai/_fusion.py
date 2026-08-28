"""GBDT fusion of bi-encoder cosine + csim TED -- requires the `scorer`
extra (`pip install csim-ai[scorer]`). Imported lazily by `Scorer` so
the base install doesn't need `scikit-learn`/`joblib`.
"""
from __future__ import annotations

from pathlib import Path


class FusionModel:
    def __init__(self, model_path: str | Path):
        import joblib

        self.model = joblib.load(model_path)

    def predict(self, biencoder_cosine: float, csim_ted: float) -> float:
        proba = self.model.predict_proba([[biencoder_cosine, csim_ted]])
        return float(proba[0, 1])
