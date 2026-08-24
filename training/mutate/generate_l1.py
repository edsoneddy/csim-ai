#!/usr/bin/env python3
"""Fase 1: generate L1 (cosmetic) synthetic positive pairs.

For each submission, tries K random L1 mutations (comments/docstrings/
indentation) and keeps the ones that actually changed the code and pass
the structural equivalence check (validate.cosmetic_equivalent) -- never
executes anything, per section 5 of the project brief.

Usage (run as a module so the relative imports resolve):
    python -m training.mutate.generate_l1 \
        --dataset-dir /path/to/dataset \
        --manifest training/data/artifacts/manifest_v1.jsonl \
        --splits training/data/artifacts/problem_splits_v1.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import libcst

from .l1_cosmetic import apply_random
from .validate import cosmetic_equivalent, parses

VARIANTS_PER_SUBMISSION = 3
SEED = 42


def load_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_splits(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {pid: info["split"] for pid, info in data["problems"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=os.environ.get("CSIM_AI_DATASET_DIR"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent.parent / "data" / "artifacts")
    args = parser.parse_args()

    if not args.dataset_dir:
        parser.error("--dataset-dir or CSIM_AI_DATASET_DIR is required")
    dataset_dir = Path(args.dataset_dir).resolve()

    manifest = load_manifest(args.manifest)
    split_of = load_splits(args.splits)
    rng = random.Random(SEED)

    out_path = args.out_dir / "synthetic_l1_v1.jsonl"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_submissions = 0
    n_generated = 0
    n_kept = 0
    n_invalid = 0  # parses fails or not cosmetic_equivalent (should be ~0, would indicate a bug)
    n_libcst_parse_failures = 0  # libcst's tokenizer is stricter than ast.parse (e.g. mixed tabs/spaces)

    with out_path.open("w", encoding="utf-8") as out:
        for row in manifest:
            n_submissions += 1
            code = (dataset_dir / row["path"]).read_text(encoding="utf-8", errors="ignore")
            try:
                libcst.parse_module(code)
            except libcst.ParserSyntaxError:
                n_libcst_parse_failures += 1
                continue

            seen = {code}
            for _ in range(VARIANTS_PER_SUBMISSION):
                result = apply_random(code, rng)
                n_generated += 1
                if not result.applied or result.code in seen:
                    continue
                seen.add(result.code)
                if not parses(result.code) or not cosmetic_equivalent(code, result.code):
                    n_invalid += 1
                    continue
                n_kept += 1
                out.write(
                    json.dumps(
                        {
                            "problem_id": row["problem_id"],
                            "submission_id": row["submission_id"],
                            "split": split_of.get(row["problem_id"], "unknown"),
                            "level": "L1",
                            "rule": result.rule,
                            "mutated_code": result.code,
                            "validated_by_execution": False,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    report = {
        "n_submissions": n_submissions,
        "n_libcst_parse_failures": n_libcst_parse_failures,
        "variants_attempted": n_generated,
        "n_kept": n_kept,
        "n_invalid": n_invalid,  # bug signal if > 0
        "kept_per_submission": round(n_kept / n_submissions, 3),
    }
    print(json.dumps(report, indent=2))
    print(f"\nout: {out_path}")


if __name__ == "__main__":
    main()
