#!/usr/bin/env python3
"""Fase 4: feature computation for the fusion scorer -- bi-encoder cosine
similarity + csim TED score, for every negative and L1-L6 positive pair
in train/dev/test.

Negatives reuse the `csim_score` already computed in Fase 0's
`csim_baseline_pairs_v1.csv` (every split, every same-problem
non-duplicate pair) -- no need to recompute TED there. Positives (L1-L6)
need a fresh TED score between each original and its `mutated_code`,
computed here with the same `csim.utils.preprocess_code` +
`get_similarity_coefficient` (apted, python_3_13) settings as Fase 0,
parallelized across CPU cores since it's ~0.1-0.2s/pair single-threaded.

TED runs first, before the bi-encoder/CUDA context is touched --
multiprocessing forks after CUDA init can hang or crash.

Usage:
    python -m training.scorer.build_features \
        --dataset-dir /path/to/dataset \
        --manifest training/data/artifacts/manifest_v1.jsonl \
        --checkpoint training/artifacts/best_checkpoint
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

LANG = "python_3_13"
TED_ALGORITHM = "apted"


def load_manifest(path: Path) -> dict[tuple[str, str], str]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[(row["problem_id"], row["submission_id"])] = row["path"]
    return out


def load_split_negatives(split: str, artifacts_dir: Path, eval_artifacts_dir: Path) -> list[dict]:
    excluded = set()
    with (artifacts_dir / "hard_negative_exclusions_v1.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            excluded.add((row["problem_id"], row["submission_a"], row["submission_b"]))

    out = []
    with (eval_artifacts_dir / "csim_baseline_pairs_v1.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split or row["is_dup_pair"] != "0":
                continue
            key = (row["problem_id"], row["submission_a"], row["submission_b"])
            if key in excluded:
                continue
            out.append({
                "problem_id": row["problem_id"],
                "submission_a": row["submission_a"],
                "submission_b": row["submission_b"],
                "csim_ted": float(row["csim_score"]),
            })
    return out


def load_split_positives(split: str, artifacts_dir: Path) -> list[dict]:
    out = []
    for lvl in ["L1", "L2", "L3", "L4", "L5", "L6"]:
        path = artifacts_dir / f"synthetic_{lvl.lower()}_v1.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["split"] == split:
                    out.append({
                        "problem_id": row["problem_id"],
                        "submission_id": row["submission_id"],
                        "level": lvl,
                        "mutated_code": row["mutated_code"],
                    })
    return out


def _ted_worker(pair: tuple[str, str]) -> float | None:
    # preprocess_code does *not* raise on a syntax error -- its ANTLR
    # grammar prints "Syntax error ..." to stderr and returns a
    # degenerate near-empty tree instead, which would otherwise silently
    # produce a garbage similarity score. Capture stderr and treat any
    # output as a failed parse, matching what csim_baseline.py (Fase 0)
    # assumed would be an exception.
    import contextlib
    import io

    from csim.utils import get_similarity_coefficient, preprocess_code

    orig_text, mut_text = pair
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf):
            proc_a = preprocess_code("a", orig_text, LANG)
            proc_b = preprocess_code("b", mut_text, LANG)
            if buf.getvalue():
                return None
            return get_similarity_coefficient(proc_a, proc_b, TED_ALGORITHM)
    except Exception:
        return None


def compute_positive_ted(positives: list[dict], manifest, dataset_dir: Path) -> list[float | None]:
    pairs = []
    for row in positives:
        orig_text = (dataset_dir / manifest[(row["problem_id"], row["submission_id"])]).read_text(
            encoding="utf-8", errors="ignore"
        )
        pairs.append((orig_text, row["mutated_code"]))

    n_workers = os.cpu_count() or 4
    chunksize = max(1, len(pairs) // (n_workers * 4))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(_ted_worker, pairs, chunksize=chunksize))


def compute_biencoder_cosine(negatives, positives, manifest, dataset_dir, checkpoint):
    from ..train_biencoder import BiEncoder

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = BiEncoder(device, model_path=str(checkpoint))
    encoder.eval()

    ids_needed = set()
    for row in negatives:
        ids_needed.add((row["problem_id"], row["submission_a"]))
        ids_needed.add((row["problem_id"], row["submission_b"]))
    for row in positives:
        ids_needed.add((row["problem_id"], row["submission_id"]))
    ids = sorted(ids_needed)
    orig_texts = [(dataset_dir / manifest[key]).read_text(encoding="utf-8", errors="ignore") for key in ids]
    index = {key: i for i, key in enumerate(ids)}
    orig_embs = encoder.encode_eval(orig_texts)

    neg_cosine = []
    for row in negatives:
        ia = index[(row["problem_id"], row["submission_a"])]
        ib = index[(row["problem_id"], row["submission_b"])]
        neg_cosine.append(float(np.dot(orig_embs[ia], orig_embs[ib])))

    mut_texts = [row["mutated_code"] for row in positives]
    mut_embs = encoder.encode_eval(mut_texts) if mut_texts else np.zeros((0, orig_embs.shape[1]))
    pos_cosine = []
    for i, row in enumerate(positives):
        io = index[(row["problem_id"], row["submission_id"])]
        pos_cosine.append(float(np.dot(orig_embs[io], mut_embs[i])))

    return neg_cosine, pos_cosine


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
    eval_artifacts_dir = Path(__file__).parent.parent / "eval" / "artifacts"
    manifest = load_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev", "test"):
        print(f"=== {split} ===")
        negatives = load_split_negatives(split, args.artifacts_dir, eval_artifacts_dir)
        positives = load_split_positives(split, args.artifacts_dir)
        print(f"negatives: {len(negatives)}, positives: {len(positives)}")

        print("computing TED for positives (parallel, CPU)...")
        pos_ted = compute_positive_ted(positives, manifest, dataset_dir)
        n_failed = sum(1 for t in pos_ted if t is None)
        if n_failed:
            print(f"WARNING: {n_failed}/{len(positives)} positive TED computations failed (preprocess_code error)")

        print("computing bi-encoder cosine (GPU)...")
        neg_cosine, pos_cosine = compute_biencoder_cosine(negatives, positives, manifest, dataset_dir, args.checkpoint)

        out_path = args.out_dir / f"features_{split}_v1.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row, cosine in zip(negatives, neg_cosine):
                f.write(json.dumps({
                    "problem_id": row["problem_id"],
                    "level": "negative",
                    "label": 0,
                    "biencoder_cosine": cosine,
                    "csim_ted": row["csim_ted"],
                }) + "\n")
            for row, cosine, ted in zip(positives, pos_cosine, pos_ted):
                if ted is None:
                    continue
                f.write(json.dumps({
                    "problem_id": row["problem_id"],
                    "level": row["level"],
                    "label": 1,
                    "biencoder_cosine": cosine,
                    "csim_ted": ted,
                }) + "\n")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
