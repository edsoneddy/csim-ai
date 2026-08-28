#!/usr/bin/env python3
"""Fase 4: Dolos baseline, same protocol as the other Fase 2/3 eval
scripts -- test-split negatives (csim baseline, non-duplicate,
same-problem pairs) and test-split L1-L6 positives (original vs
mutated_code), scored with Dolos (winnowing-based token similarity)
instead of an embedding cosine similarity, then AUROC/AUPRC/FPR@recall95
per level.

Dolos compares files *within one assignment* (its own model), so this
runs one Dolos analysis per test-split problem_id -- writing each
problem's original submissions plus every level's mutated_code out as
throwaway files, parsing `pairs.csv` from the CSV report for exactly the
pair scores this protocol needs, and discarding everything else Dolos
computes (kgrams, fragments, etc).

Requires Node.js >=22 (Dolos's native tree-sitter parser build fails
under Node 20 -- `node-gyp`'s own dependencies need it, even though
Dolos's package.json only declares >=18). Set up once:

    nvm install 22
    export PATH="$(nvm_path)/versions/node/v22.*/bin:$PATH"
    npm install --prefix training/eval/dolos_tool @dodona/dolos

Usage:
    python -m training.eval.dolos_baseline \
        --dataset-dir /path/to/dataset \
        --manifest training/data/artifacts/manifest_v1.jsonl \
        --node-bin /home/user/.nvm/versions/node/v22.23.2/bin/node
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from .metrics import auprc, auroc, fpr_at_recall

DOLOS_CLI = Path(__file__).parent / "dolos_tool" / "node_modules" / "@dodona" / "dolos" / "dist" / "cli.js"


def load_manifest(path: Path) -> dict[tuple[str, str], str]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[(row["problem_id"], row["submission_id"])] = row["path"]
    return out


def load_test_negatives(artifacts_dir: Path, eval_artifacts_dir: Path) -> list[tuple[str, str, str]]:
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
    return negatives


def load_test_positives(artifacts_dir: Path) -> dict[str, list[dict]]:
    positives_by_level = {}
    for lvl in ["L1", "L2", "L3", "L4", "L5", "L6"]:
        path = artifacts_dir / f"synthetic_{lvl.lower()}_v1.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        positives_by_level[lvl] = [r for r in rows if r["split"] == "test"]
    return positives_by_level


def run_dolos(problem_dir: Path, node_bin: str, out_dir: Path) -> list[dict]:
    # -o's target dir must not already exist -- dolos creates it itself
    # and refuses otherwise.
    py_files = sorted(p.name for p in problem_dir.glob("*.py"))
    result = subprocess.run(
        [node_bin, str(DOLOS_CLI), "run", "-l", "python", "-f", "csv", "-o", str(out_dir), *py_files],
        cwd=problem_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dolos failed in {problem_dir}:\n{result.stdout}\n{result.stderr}")
    with (out_dir / "pairs.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=os.environ.get("CSIM_AI_DATASET_DIR"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path(__file__).parent.parent / "data" / "artifacts"
    )
    parser.add_argument(
        "--node-bin", default=shutil.which("node"),
        help="Path to a Node.js >=22 binary (system PATH's `node` is not enough if it's older).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "artifacts")
    args = parser.parse_args()

    if not args.dataset_dir:
        parser.error("--dataset-dir or CSIM_AI_DATASET_DIR is required")
    if not args.node_bin:
        parser.error("--node-bin is required (no `node` found on PATH)")
    dataset_dir = Path(args.dataset_dir).resolve()
    eval_artifacts_dir = Path(__file__).parent / "artifacts"

    manifest = load_manifest(args.manifest)
    negatives = load_test_negatives(args.artifacts_dir, eval_artifacts_dir)
    positives_by_level = load_test_positives(args.artifacts_dir)
    print(f"test negatives: {len(negatives)}")
    for lvl, rows in positives_by_level.items():
        print(f"{lvl}: {len(rows)} test positive pairs")

    negatives_by_problem: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for pid, sa, sb in negatives:
        negatives_by_problem[pid].append((sa, sb))

    positives_by_problem: dict[str, list[tuple[str, int, dict]]] = defaultdict(list)
    for lvl, rows in positives_by_level.items():
        by_problem_idx: dict[str, int] = defaultdict(int)
        for row in rows:
            pid = row["problem_id"]
            idx = by_problem_idx[pid]
            by_problem_idx[pid] += 1
            positives_by_problem[pid].append((lvl, idx, row))

    problem_ids = sorted(set(negatives_by_problem) | set(positives_by_problem))
    print(f"test problems: {len(problem_ids)}")

    pos_scores_by_level: dict[str, list[float]] = defaultdict(list)
    neg_scores: list[float] = []
    missing_pairs = 0

    for i, pid in enumerate(problem_ids):
        originals_needed = {sa for sa, _ in negatives_by_problem[pid]} | {sb for _, sb in negatives_by_problem[pid]}
        originals_needed |= {row["submission_id"] for _, _, row in positives_by_problem[pid]}

        with tempfile.TemporaryDirectory() as tmp:
            problem_dir = Path(tmp)
            for sub_id in originals_needed:
                text = (dataset_dir / manifest[(pid, sub_id)]).read_text(encoding="utf-8", errors="ignore")
                (problem_dir / f"orig__{sub_id}.py").write_text(text, encoding="utf-8")
            for lvl, idx, row in positives_by_problem[pid]:
                (problem_dir / f"pos__{lvl}__{row['submission_id']}__{idx}.py").write_text(
                    row["mutated_code"], encoding="utf-8"
                )

            out_dir = Path(tmp + "_report")
            pairs = run_dolos(problem_dir, args.node_bin, out_dir)
            score_by_names: dict[frozenset, float] = {}
            for row in pairs:
                key = frozenset((row["leftFilePath"], row["rightFilePath"]))
                score_by_names[key] = float(row["similarity"])
            shutil.rmtree(out_dir, ignore_errors=True)

        for sa, sb in negatives_by_problem[pid]:
            key = frozenset((f"orig__{sa}.py", f"orig__{sb}.py"))
            if key not in score_by_names:
                missing_pairs += 1
                continue
            neg_scores.append(score_by_names[key])

        for lvl, idx, row in positives_by_problem[pid]:
            key = frozenset((f"orig__{row['submission_id']}.py", f"pos__{lvl}__{row['submission_id']}__{idx}.py"))
            if key not in score_by_names:
                missing_pairs += 1
                continue
            pos_scores_by_level[lvl].append(score_by_names[key])

        if (i + 1) % 10 == 0 or (i + 1) == len(problem_ids):
            print(f"  {i + 1}/{len(problem_ids)} problems done")

    if missing_pairs:
        print(f"WARNING: {missing_pairs} expected pairs missing from dolos output")

    results = {}
    for lvl, pos_scores in pos_scores_by_level.items():
        if not pos_scores:
            continue
        scores = pos_scores + neg_scores
        labels = [1] * len(pos_scores) + [0] * len(neg_scores)
        results[lvl] = {
            "n_positive": len(pos_scores),
            "n_negative": len(neg_scores),
            "auroc": round(auroc(scores, labels), 4),
            "auprc": round(auprc(scores, labels), 4),
            "fpr_at_recall95": round(fpr_at_recall(scores, labels), 4),
        }
        print(json.dumps({"level": lvl, **results[lvl]}))

    mean_l4_l6 = sum(results[lvl]["auroc"] for lvl in ("L4", "L5", "L6") if lvl in results) / 3
    report = {"levels": results, "mean_l4_l6_auroc": round(mean_l4_l6, 4)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "dolos_baseline_v1.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nmean L4-L6 AUROC: {report['mean_l4_l6_auroc']}")
    print(f"out: {out_path}")


if __name__ == "__main__":
    main()
