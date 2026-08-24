#!/usr/bin/env python3
"""Fase 0: canonical hard-negative exclusion rule.

A same-problem pair is unsafe to sample as a hard negative if csim marks it
near-identical: either an exact raw-text duplicate, or a non-duplicate pair
whose csim (TED/APTED) score is >= NEAR_DUP_THRESHOLD. Either case likely
means one submission derived from the other, so treating the pair as a
negative would poison training (section 5 of the project brief: "excluir de
los negativos los pares que csim ya marca como casi identicos").

Reads the pairs already scored by training/eval/csim_baseline.py and writes
the single canonical exclusion list the training sampler (Fase 3) should
consult before drawing a same-problem negative pair.

Usage:
    python training/data/make_hard_negative_exclusions.py \
        --pairs training/eval/artifacts/csim_baseline_pairs_v1.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

NEAR_DUP_THRESHOLD = 0.95  # matches PSEUDO_POSITIVE_THRESHOLD in csim_baseline.py


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "hard_negative_exclusions_v1.jsonl",
    )
    args = parser.parse_args()

    n_hash_dup = n_near_dup = 0
    with args.pairs.open(newline="", encoding="utf-8") as pf, args.out.open(
        "w", encoding="utf-8"
    ) as of:
        reader = csv.DictReader(pf)
        for row in reader:
            score = float(row["csim_score"])
            is_hash_dup = row["is_dup_pair"] == "1"
            if not (is_hash_dup or score >= NEAR_DUP_THRESHOLD):
                continue
            reason = "exact_hash_dup" if is_hash_dup else "csim_near_dup"
            n_hash_dup += is_hash_dup
            n_near_dup += not is_hash_dup
            of.write(
                json.dumps(
                    {
                        "problem_id": row["problem_id"],
                        "submission_a": row["submission_a"],
                        "submission_b": row["submission_b"],
                        "reason": reason,
                        "csim_score": round(score, 4),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"exact_hash_dup: {n_hash_dup}")
    print(f"csim_near_dup (score >= {NEAR_DUP_THRESHOLD}, not a hash dup): {n_near_dup}")
    print(f"total excluded pairs: {n_hash_dup + n_near_dup}")
    print(f"out: {args.out}")


if __name__ == "__main__":
    main()
