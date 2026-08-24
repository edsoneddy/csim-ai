#!/usr/bin/env python3
"""Fase 0 dataset audit.

Scans a dataset laid out as <dataset_dir>/<problem_id>/<submission_id>.py,
hashes each submission to detect exact duplicates within a problem, checks
that it parses as Python, and emits a manifest + summary report.

Usage:
    python training/data/audit.py --dataset-dir /path/to/dataset
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path


def iter_submissions(dataset_dir: Path):
    for problem_dir in sorted(dataset_dir.iterdir()):
        if not problem_dir.is_dir():
            continue
        problem_id = problem_dir.name
        for path in sorted(problem_dir.glob("*.py")):
            yield problem_id, path.stem, path


def audit(dataset_dir: Path) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    hash_groups: dict[str, list[int]] = {}

    for problem_id, submission_id, path in iter_submissions(dataset_dir):
        src = path.read_text(encoding="utf-8", errors="ignore")
        sha = hashlib.sha256(src.strip().encode("utf-8")).hexdigest()
        try:
            ast.parse(src)
            parseable = True
        except SyntaxError:
            parseable = False

        row = {
            "problem_id": problem_id,
            "submission_id": submission_id,
            "path": str(path.relative_to(dataset_dir)),
            "line_count": src.count("\n") + (1 if src and not src.endswith("\n") else 0),
            "sha256": sha,
            "is_parseable": parseable,
        }
        rows.append(row)
        hash_groups.setdefault(f"{problem_id}:{sha}", []).append(len(rows) - 1)

    # exact-duplicate groups: same content, same problem, different submission_id
    for key, indices in hash_groups.items():
        group_id = key if len(indices) > 1 else None
        for i in indices:
            rows[i]["dup_group_id"] = group_id
            rows[i]["dup_group_size"] = len(indices)

    return rows, summarize(rows)


def summarize(rows: list[dict]) -> dict:
    by_problem: dict[str, int] = {}
    for r in rows:
        by_problem[r["problem_id"]] = by_problem.get(r["problem_id"], 0) + 1
    counts = sorted(by_problem.values())
    lines = sorted(r["line_count"] for r in rows)

    def pct(data: list[int], p: float) -> int:
        idx = min(len(data) - 1, int(len(data) * p))
        return data[idx]

    dup_rows = [r for r in rows if r["dup_group_id"]]
    dup_groups = {r["dup_group_id"] for r in dup_rows}
    problems_with_dups = {r["problem_id"] for r in dup_rows}
    n_extra_copies = len(dup_rows) - len(dup_groups)  # copies beyond the first per group
    parse_failures = [r for r in rows if not r["is_parseable"]]

    return {
        "n_problems": len(by_problem),
        "n_submissions": len(rows),
        "submissions_per_problem": {
            "min": counts[0],
            "p25": pct(counts, 0.25),
            "median": pct(counts, 0.5),
            "mean": round(sum(counts) / len(counts), 2),
            "p75": pct(counts, 0.75),
            "max": counts[-1],
        },
        "lines_per_submission": {
            "min": lines[0],
            "p25": pct(lines, 0.25),
            "median": pct(lines, 0.5),
            "mean": round(sum(lines) / len(lines), 2),
            "p75": pct(lines, 0.75),
            "p95": pct(lines, 0.95),
            "max": lines[-1],
        },
        "problems_with_lt2_submissions": sum(1 for c in counts if c < 2),
        "exact_duplicates": {
            "n_extra_copies": n_extra_copies,
            "n_duplicate_groups": len(dup_groups),
            "problems_with_duplicates": len(problems_with_dups),
            "pct_of_submissions": round(100 * n_extra_copies / len(rows), 2),
        },
        "parse_failures": len(parse_failures),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=os.environ.get("CSIM_AI_DATASET_DIR")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path(__file__).parent / "artifacts"
    )
    args = parser.parse_args()

    if not args.dataset_dir:
        parser.error("--dataset-dir or CSIM_AI_DATASET_DIR is required")
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        parser.error(f"not a directory: {dataset_dir}")

    rows, report = audit(dataset_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest_v1.jsonl"
    report_path = args.out_dir / "audit_report_v1.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nmanifest: {manifest_path}")
    print(f"report:   {report_path}")


if __name__ == "__main__":
    main()
