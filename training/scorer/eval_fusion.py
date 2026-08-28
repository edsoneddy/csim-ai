#!/usr/bin/env python3
"""Fase 4: final comparison eval -- Decision 2. Scores the test split
with the fitted fusion model and tabulates it against its own two input
features alone (csim TED, bi-encoder cosine) plus the already-computed
Dolos, zero-shot, and fine-tuned-alone numbers from earlier scripts.

Usage:
    python -m training.scorer.eval_fusion
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from ..eval.metrics import auprc, auroc, fpr_at_recall


def load_test_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def metrics_for(scores: list[float], labels: list[int]) -> dict:
    return {
        "auroc": round(auroc(scores, labels), 4),
        "auprc": round(auprc(scores, labels), 4),
        "fpr_at_recall95": round(fpr_at_recall(scores, labels), 4),
    }


def per_level_report(rows: list[dict], score_key: str) -> dict:
    neg_scores = [r[score_key] for r in rows if r["level"] == "negative"]
    results = {}
    by_level = defaultdict(list)
    for r in rows:
        if r["level"] != "negative":
            by_level[r["level"]].append(r[score_key])
    for lvl, pos_scores in by_level.items():
        scores = pos_scores + neg_scores
        labels = [1] * len(pos_scores) + [0] * len(neg_scores)
        results[lvl] = metrics_for(scores, labels)
    mean_l4_l6 = sum(results[lvl]["auroc"] for lvl in ("L4", "L5", "L6") if lvl in results) / 3
    return {"levels": results, "mean_l4_l6_auroc": round(mean_l4_l6, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir", type=Path, default=Path(__file__).parent / "artifacts"
    )
    parser.add_argument(
        "--model", type=Path, default=Path(__file__).parent / "artifacts" / "fusion_model_v1.joblib"
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "artifacts")
    args = parser.parse_args()

    rows = load_test_rows(args.features_dir / "features_test_v1.jsonl")
    model = joblib.load(args.model)
    X = np.array([[r["biencoder_cosine"], r["csim_ted"]] for r in rows])
    fusion_probs = model.predict_proba(X)[:, 1]
    for r, p in zip(rows, fusion_probs):
        r["fusion_score"] = float(p)

    report = {
        "fusion": per_level_report(rows, "fusion_score"),
        "biencoder_cosine_alone": per_level_report(rows, "biencoder_cosine"),
        "csim_ted_alone": per_level_report(rows, "csim_ted"),
    }

    eval_artifacts = Path(__file__).parent.parent / "eval" / "artifacts"
    dolos_path = eval_artifacts / "dolos_baseline_v1.json"
    final_eval_path = eval_artifacts / "final_test_eval_v1.json"
    if dolos_path.exists():
        report["dolos"] = json.loads(dolos_path.read_text(encoding="utf-8"))
    if final_eval_path.exists():
        final_eval = json.loads(final_eval_path.read_text(encoding="utf-8"))
        report["zero_shot_biencoder"] = final_eval["zero_shot"]
        report["fine_tuned_biencoder"] = final_eval["fine_tuned"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "eval_fusion_v1.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'model':<24}{'mean L4-L6 AUROC':>18}")
    for name in ("fusion", "biencoder_cosine_alone", "csim_ted_alone", "dolos", "zero_shot_biencoder", "fine_tuned_biencoder"):
        if name in report and "mean_l4_l6_auroc" in report[name]:
            print(f"{name:<24}{report[name]['mean_l4_l6_auroc']:>18}")
    print(f"\nout: {out_path}")


if __name__ == "__main__":
    main()
