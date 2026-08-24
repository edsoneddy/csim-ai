#!/usr/bin/env python3
"""Fase 0 csim (TED/APTED) baseline: exhaustive within-problem pairwise
similarity.

We have no plagiarism labels yet, so this does not compute AUROC/AUPRC --
that needs either the L1-L6 synthetic generator (Fase 1) or an external
labeled set (ConPlag/IR-Plag, eval-only). What this script does now:

1. Surfaces pseudo-positive candidates for manual review (section 5):
   same-problem, non-exact-duplicate pairs with csim score >= 0.95.
2. Quantifies the exact-duplicate contamination risk in the hard-negative
   pool: the score distribution for duplicate vs. non-duplicate pairs.

Usage:
    python training/eval/csim_baseline.py \
        --dataset-dir /path/to/dataset \
        --manifest training/data/artifacts/manifest_v1.jsonl \
        --splits training/data/artifacts/problem_splits_v1.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from csim.utils import get_similarity_coefficient, preprocess_code

PSEUDO_POSITIVE_THRESHOLD = 0.95
LANG = "python_3_13"
TED_ALGORITHM = "apted"


def load_manifest(path: Path) -> dict[str, list[dict]]:
    by_problem: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_problem[row["problem_id"]].append(row)
    return by_problem


def load_splits(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {pid: info["split"] for pid, info in data["problems"].items()}


def preprocess_problem(dataset_dir: Path, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Preprocess every submission in a problem once. Returns (ok_rows, failed_rows)."""
    ok, failed = [], []
    for row in rows:
        try:
            content = (dataset_dir / row["path"]).read_text(encoding="utf-8", errors="ignore")
            processed = preprocess_code(row["submission_id"], content, LANG)
        except Exception as e:  # csim's ANTLR grammar can reject valid ast.parse() code
            failed.append({**row, "error": str(e)})
            continue
        ok.append({**row, "_processed": processed})
    return ok, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=os.environ.get("CSIM_AI_DATASET_DIR")
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument(
        "--out-dir", type=Path, default=Path(__file__).parent / "artifacts"
    )
    args = parser.parse_args()

    if not args.dataset_dir:
        parser.error("--dataset-dir or CSIM_AI_DATASET_DIR is required")
    dataset_dir = Path(args.dataset_dir).resolve()

    by_problem = load_manifest(args.manifest)
    split_of = load_splits(args.splits)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.out_dir / "csim_baseline_pairs_v1.csv"
    pseudo_pos_path = args.out_dir / "csim_pseudo_positives_v1.jsonl"
    report_path = args.out_dir / "csim_baseline_report_v1.json"

    n_dup_pairs = n_nondup_pairs = 0
    sum_dup = sum_nondup = 0.0
    dup_scores, nondup_scores = [], []
    n_pseudo_positive = 0
    n_preprocess_failures = 0
    problems_with_failures = 0

    with pairs_path.open("w", newline="", encoding="utf-8") as pf, pseudo_pos_path.open(
        "w", encoding="utf-8"
    ) as ppf:
        writer = csv.writer(pf)
        writer.writerow(["problem_id", "split", "submission_a", "submission_b", "csim_score", "is_dup_pair"])

        for problem_id, rows in sorted(by_problem.items()):
            if len(rows) < 2:
                continue
            ok, failed = preprocess_problem(dataset_dir, rows)
            if failed:
                n_preprocess_failures += len(failed)
                problems_with_failures += 1
            if len(ok) < 2:
                continue

            split = split_of.get(problem_id, "unknown")
            for a, b in combinations(ok, 2):
                score = get_similarity_coefficient(a["_processed"], b["_processed"], TED_ALGORITHM)
                is_dup = bool(a["dup_group_id"]) and a["dup_group_id"] == b["dup_group_id"]

                writer.writerow([problem_id, split, a["submission_id"], b["submission_id"], round(score, 4), int(is_dup)])

                if is_dup:
                    n_dup_pairs += 1
                    sum_dup += score
                    dup_scores.append(score)
                else:
                    n_nondup_pairs += 1
                    sum_nondup += score
                    nondup_scores.append(score)
                    if score >= PSEUDO_POSITIVE_THRESHOLD:
                        n_pseudo_positive += 1
                        ppf.write(
                            json.dumps(
                                {
                                    "problem_id": problem_id,
                                    "split": split,
                                    "submission_a": a["submission_id"],
                                    "submission_b": b["submission_id"],
                                    "csim_score": round(score, 4),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

    def pct(data: list[float], p: float) -> float:
        if not data:
            return float("nan")
        s = sorted(data)
        idx = min(len(s) - 1, int(len(s) * p))
        return round(s[idx], 4)

    report = {
        "ted_algorithm": TED_ALGORITHM,
        "pseudo_positive_threshold": PSEUDO_POSITIVE_THRESHOLD,
        "n_preprocess_failures": n_preprocess_failures,
        "problems_with_preprocess_failures": problems_with_failures,
        "duplicate_pairs": {
            "n": n_dup_pairs,
            "mean_score": round(sum_dup / n_dup_pairs, 4) if n_dup_pairs else None,
            "p05": pct(dup_scores, 0.05),
            "median": pct(dup_scores, 0.5),
        },
        "non_duplicate_pairs": {
            "n": n_nondup_pairs,
            "mean_score": round(sum_nondup / n_nondup_pairs, 4) if n_nondup_pairs else None,
            "median": pct(nondup_scores, 0.5),
            "p95": pct(nondup_scores, 0.95),
            "p99": pct(nondup_scores, 0.99),
        },
        "pseudo_positive_candidates": {
            "n": n_pseudo_positive,
            "pct_of_non_duplicate_pairs": round(100 * n_pseudo_positive / n_nondup_pairs, 4)
            if n_nondup_pairs
            else None,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\npairs csv:        {pairs_path}")
    print(f"pseudo-positives:  {pseudo_pos_path}")
    print(f"report:            {report_path}")


if __name__ == "__main__":
    main()
