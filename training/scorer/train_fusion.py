#!/usr/bin/env python3
"""Fase 4: fit the fusion scorer -- a GBDT over two features
(bi-encoder cosine similarity, csim TED score) -- on the train split's
features from `training.scorer.build_features`.

Two features and ~265k labeled rows means low overfitting risk; no
hyperparameter search, just sklearn's HistGradientBoostingClassifier
defaults.

Usage:
    python -m training.scorer.train_fusion
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            X.append([row["biencoder_cosine"], row["csim_ted"]])
            y.append(row["label"])
    return np.array(X), np.array(y)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir", type=Path, default=Path(__file__).parent / "artifacts"
    )
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "artifacts" / "fusion_model_v1.joblib"
    )
    args = parser.parse_args()

    X_train, y_train = load_features(args.features_dir / "features_train_v1.jsonl")
    print(f"train rows: {len(y_train)} ({y_train.sum()} positive, {len(y_train) - y_train.sum()} negative)")

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    print(f"train accuracy: {train_acc:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
