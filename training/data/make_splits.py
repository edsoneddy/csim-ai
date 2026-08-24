#!/usr/bin/env python3
"""Fase 0 frozen train/dev/test split, by problem.

Stratifies by submissions-per-problem bucket so dev/test aren't dominated by
thin problems, then splits each bucket independently with a seeded,
per-bucket RNG (so adding/removing a bucket never reshuffles the others).

Usage:
    python training/data/make_splits.py \
        --manifest training/data/artifacts/manifest_v1.jsonl \
        --config training/configs/data.yaml
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml


def load_counts(manifest_path: Path) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            counts[row["problem_id"]] += 1
    return counts


def bucket_of(count: int, edges: list[int]) -> int:
    return bisect.bisect_right(edges, count) - 1


def split_bucket(
    problem_ids: list[str], ratios: dict[str, float], seed: int, bucket_idx: int
) -> dict[str, str]:
    rng = random.Random(f"{seed}:{bucket_idx}")
    ids = sorted(problem_ids)  # sort first: shuffle is reproducible regardless of manifest order
    rng.shuffle(ids)
    n = len(ids)
    n_train = round(n * ratios["train"])
    n_dev = round(n * ratios["dev"])
    train, dev, test = ids[:n_train], ids[n_train : n_train + n_dev], ids[n_train + n_dev :]
    return {pid: "train" for pid in train} | {pid: "dev" for pid in dev} | {
        pid: "test" for pid in test
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "problem_splits_v1.json",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = config["seed"]
    ratios = config["split_ratios"]
    edges = config["submission_count_buckets"]

    counts = load_counts(args.manifest)
    buckets: dict[int, list[str]] = defaultdict(list)
    for pid, c in counts.items():
        buckets[bucket_of(c, edges)].append(pid)

    assignment: dict[str, str] = {}
    for bucket_idx, pids in buckets.items():
        assignment.update(split_bucket(pids, ratios, seed, bucket_idx))

    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()

    summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n_problems": 0, "n_submissions": 0, "n_hard_negative_eligible": 0}
    )
    for pid, split in assignment.items():
        summary[split]["n_problems"] += 1
        summary[split]["n_submissions"] += counts[pid]
        if counts[pid] >= 2:
            summary[split]["n_hard_negative_eligible"] += 1

    out = {
        "seed": seed,
        "ratios": ratios,
        "submission_count_buckets": edges,
        "manifest_sha256": manifest_hash,
        "split_summary": summary,
        "problems": {
            pid: {
                "split": split,
                "submission_count": counts[pid],
                # a problem needs >=2 submissions to source same-problem hard negatives
                "eligible_for_hard_negatives": counts[pid] >= 2,
            }
            for pid, split in sorted(assignment.items())
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"split_summary": summary}, indent=2, default=dict))
    print(f"\nsplits: {args.out}")


if __name__ == "__main__":
    main()
