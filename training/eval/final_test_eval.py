#!/usr/bin/env python3
"""Fase 3: final evaluation on the untouched TEST split -- the actual
Decision 1 call (does the fine-tuned bi-encoder beat zero-shot UniXcoder
on L4-L6?).

Same protocol as Fase 2's zero-shot baseline and Fase 3's periodic dev
eval (same shared negative pool from the csim baseline, same per-level
synthetic positives), just switched to the test split, which nothing has
touched until now. Evaluates two sets of weights through the same
encoder class (mean pooling + L2 normalize, from train_biencoder.BiEncoder)
so the only difference between the two rows is the weights: pretrained
microsoft/unixcoder-base vs. the fine-tuned checkpoint.

csim is not included here: installing torch bumped numpy past csim's
`==1.26.4` pin in this venv, so there's no in-venv csim score for L1-L6
yet (see Fase 2 caveats in the README) -- unresolved until the packaging
conflict is sorted out in Fase 5.

Usage:
    python -m training.eval.final_test_eval \
        --dataset-dir /path/to/dataset \
        --manifest training/data/artifacts/manifest_v1.jsonl \
        --checkpoint training/artifacts/best_checkpoint
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from ..train_biencoder import MODEL_ID, BiEncoder
from .metrics import auprc, auroc, fpr_at_recall


def load_test_eval_data(manifest_path: Path, artifacts_dir: Path, eval_artifacts_dir: Path):
    manifest = {}
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            manifest[(row["problem_id"], row["submission_id"])] = row["path"]

    excluded = set()
    with (artifacts_dir / "hard_negative_exclusions_v1.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            excluded.add((row["problem_id"], row["submission_a"], row["submission_b"]))

    negatives = []
    with (eval_artifacts_dir / "csim_baseline_pairs_v1.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != "test" or row["is_dup_pair"] != "0":
                continue
            key = (row["problem_id"], row["submission_a"], row["submission_b"])
            if key not in excluded:
                negatives.append(key)

    positives_by_level = {}
    for lvl in ["L1", "L2", "L3", "L4", "L5", "L6"]:
        path = artifacts_dir / f"synthetic_{lvl.lower()}_v1.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        positives_by_level[lvl] = [r for r in rows if r["split"] == "test"]

    return manifest, negatives, positives_by_level


def run_eval(encoder: BiEncoder, dataset_dir: Path, manifest, negatives, positives_by_level) -> dict:
    needed = set()
    for pid, sa, sb in negatives:
        needed.add((pid, sa))
        needed.add((pid, sb))
    for rows in positives_by_level.values():
        for row in rows:
            needed.add((row["problem_id"], row["submission_id"]))
    ids = sorted(needed)
    texts = [(dataset_dir / manifest[key]).read_text(encoding="utf-8", errors="ignore") for key in ids]
    index = {key: i for i, key in enumerate(ids)}

    orig_embs = encoder.encode_eval(texts)
    neg_scores = [float(np.dot(orig_embs[index[(pid, sa)]], orig_embs[index[(pid, sb)]])) for pid, sa, sb in negatives]

    results = {}
    for lvl, rows in positives_by_level.items():
        if not rows:
            continue
        mutated_texts = [r["mutated_code"] for r in rows]
        mut_embs = encoder.encode_eval(mutated_texts)
        pos_scores = [
            float(np.dot(orig_embs[index[(r["problem_id"], r["submission_id"])]], mut_embs[i]))
            for i, r in enumerate(rows)
        ]
        scores = pos_scores + neg_scores
        labels = [1] * len(pos_scores) + [0] * len(neg_scores)
        results[lvl] = {
            "n_positive": len(pos_scores),
            "n_negative": len(neg_scores),
            "auroc": round(auroc(scores, labels), 4),
            "auprc": round(auprc(scores, labels), 4),
            "fpr_at_recall95": round(fpr_at_recall(scores, labels), 4),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=os.environ.get("CSIM_AI_DATASET_DIR"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(__file__).parent.parent / "artifacts" / "best_checkpoint"
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path(__file__).parent.parent / "data" / "artifacts"
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "artifacts")
    args = parser.parse_args()

    if not args.dataset_dir:
        parser.error("--dataset-dir or CSIM_AI_DATASET_DIR is required")
    dataset_dir = Path(args.dataset_dir).resolve()

    manifest, negatives, positives_by_level = load_test_eval_data(
        args.manifest, args.artifacts_dir, Path(__file__).parent / "artifacts"
    )
    print(f"test negatives: {len(negatives)}")
    for lvl, rows in positives_by_level.items():
        print(f"{lvl}: {len(rows)} test positive pairs")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    report = {}
    for name, model_path in [("zero_shot", MODEL_ID), ("fine_tuned", str(args.checkpoint))]:
        print(f"\n=== {name} ({model_path}) ===")
        encoder = BiEncoder(device, model_path=model_path)
        encoder.eval()
        results = run_eval(encoder, dataset_dir, manifest, negatives, positives_by_level)
        mean_l4_l6 = sum(results[lvl]["auroc"] for lvl in ("L4", "L5", "L6") if lvl in results) / 3
        report[name] = {"levels": results, "mean_l4_l6_auroc": round(mean_l4_l6, 4)}
        print(json.dumps(report[name], indent=2))
        del encoder
        if device == "cuda":
            torch.cuda.empty_cache()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "final_test_eval_v1.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nout: {out_path}")

    delta = report["fine_tuned"]["mean_l4_l6_auroc"] - report["zero_shot"]["mean_l4_l6_auroc"]
    print(f"\nmean L4-L6 AUROC: zero-shot={report['zero_shot']['mean_l4_l6_auroc']} "
          f"fine-tuned={report['fine_tuned']['mean_l4_l6_auroc']} (delta={delta:+.4f})")


if __name__ == "__main__":
    main()
