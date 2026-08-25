#!/usr/bin/env python3
"""Fase 2: zero-shot baseline -- cosine similarity from pretrained code
encoders, no fine-tuning. Confirms (or not) the embedding-collapse
reported in the literature (section 3 of the project brief) on our own
dataset, per level L1-L6, using the same protocol and metrics as the
csim/ConPlag baselines from Fase 0: AUROC, AUPRC, FPR@recall95.

Models: microsoft/unixcoder-base (MIT), nomic-ai/CodeRankEmbed (MIT).
CodeSage-v2-small was dropped -- its remote code targets a transformers
version old enough that Python 3.14 / transformers 5.x can't run it
without patching multiple removed internal APIs one at a time (Conv1D
moved, then all_tied_weights_keys missing, then get_head_mask missing);
not worth chasing for a zero-shot reference point. See README.

CodeRankEmbed is a retrieval model (query text -> code document); our
task is code-vs-code similarity, not query-to-document retrieval, so
both sides are embedded as plain documents (no "search_query:" prefix)
-- the model card's only defined prompt doesn't fit this use case.

Positives: (original, L{n} mutated) pairs from the synthetic generator
(Fase 1), dev split only.
Negatives: same-problem pairs from the csim baseline (Fase 0), dev
split, excluding exact-hash duplicates and near-duplicate pairs already
flagged unsafe (hard_negative_exclusions_v1.jsonl) -- one shared pool,
reused across all levels and both models, since it doesn't depend on
any encoder.

Usage:
    python -m training.eval.zero_shot_baseline \
        --dataset-dir /path/to/dataset \
        --manifest training/data/artifacts/manifest_v1.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from .metrics import auprc, auroc, fpr_at_recall


def load_manifest(path: Path) -> dict[tuple[str, str], str]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[(row["problem_id"], row["submission_id"])] = row["path"]
    return out


def load_negative_pairs(pairs_csv: Path, exclusions_path: Path) -> list[tuple[str, str, str]]:
    excluded = set()
    with exclusions_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            excluded.add((row["problem_id"], row["submission_a"], row["submission_b"]))

    negatives = []
    with pairs_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != "dev" or row["is_dup_pair"] != "0":
                continue
            key = (row["problem_id"], row["submission_a"], row["submission_b"])
            if key in excluded:
                continue
            negatives.append(key)
    return negatives


def load_positive_rows(artifacts_dir: Path, level: str) -> list[dict]:
    path = artifacts_dir / f"synthetic_{level.lower()}_v1.jsonl"
    with path.open(encoding="utf-8") as f:
        rows = (json.loads(line) for line in f)
        return [row for row in rows if row["split"] == "dev"]


def get_encoder(name: str):
    if name == "unixcoder":
        return UniXcoderEncoder()
    if name == "coderank":
        return CodeRankEncoder()
    raise ValueError(name)


class UniXcoderEncoder:
    name = "unixcoder"
    model_id = "microsoft/unixcoder-base"

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        torch = self.torch
        embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self.tok(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(self.device)
                out = self.model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                summed = (out.last_hidden_state * mask).sum(1)
                counts = mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(summed / counts, dim=-1)
                embs.append(pooled.cpu().numpy())
        return np.concatenate(embs, axis=0)


class CodeRankEncoder:
    name = "coderank"
    model_id = "nomic-ai/CodeRankEmbed"

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_id, trust_remote_code=True)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=os.environ.get("CSIM_AI_DATASET_DIR"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path(__file__).parent.parent / "data" / "artifacts"
    )
    parser.add_argument("--models", nargs="+", default=["unixcoder", "coderank"])
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "artifacts")
    args = parser.parse_args()

    if not args.dataset_dir:
        parser.error("--dataset-dir or CSIM_AI_DATASET_DIR is required")
    dataset_dir = Path(args.dataset_dir).resolve()

    manifest = load_manifest(args.manifest)
    negatives = load_negative_pairs(
        Path(__file__).parent / "artifacts" / "csim_baseline_pairs_v1.csv",
        args.artifacts_dir / "hard_negative_exclusions_v1.jsonl",
    )
    print(f"negative pairs (dev, clean): {len(negatives)}")

    levels = ["L1", "L2", "L3", "L4", "L5", "L6"]
    positives_by_level = {lvl: load_positive_rows(args.artifacts_dir, lvl) for lvl in levels}
    for lvl, rows in positives_by_level.items():
        print(f"{lvl}: {len(rows)} dev positive pairs")

    # Build the set of (problem_id, submission_id) whose *original* source
    # we need, plus every unique mutated_code string (keyed by identity,
    # not dedup'd -- each row's mutated_code is already unique).
    needed_originals: set[tuple[str, str]] = set()
    for pid, sa, sb in negatives:
        needed_originals.add((pid, sa))
        needed_originals.add((pid, sb))
    for rows in positives_by_level.values():
        for row in rows:
            needed_originals.add((row["problem_id"], row["submission_id"]))

    original_ids = sorted(needed_originals)
    original_texts = [
        (dataset_dir / manifest[key]).read_text(encoding="utf-8", errors="ignore") for key in original_ids
    ]
    id_to_index = {key: i for i, key in enumerate(original_ids)}
    print(f"unique original submissions to embed: {len(original_ids)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_reports = []

    for model_name in args.models:
        print(f"\n=== {model_name} ===")
        encoder = get_encoder(model_name)

        orig_embs = encoder.encode(original_texts)

        neg_scores = []
        for pid, sa, sb in negatives:
            ia, ib = id_to_index[(pid, sa)], id_to_index[(pid, sb)]
            neg_scores.append(float(np.dot(orig_embs[ia], orig_embs[ib])))

        for lvl in levels:
            rows = positives_by_level[lvl]
            if not rows:
                continue
            mutated_texts = [row["mutated_code"] for row in rows]
            mut_embs = encoder.encode(mutated_texts)
            orig_indices = [id_to_index[(row["problem_id"], row["submission_id"])] for row in rows]
            pos_scores = [
                float(np.dot(orig_embs[orig_indices[i]], mut_embs[i])) for i in range(len(rows))
            ]

            scores = pos_scores + neg_scores
            labels = [1] * len(pos_scores) + [0] * len(neg_scores)
            report = {
                "model": model_name,
                "level": lvl,
                "n_positive": len(pos_scores),
                "n_negative": len(neg_scores),
                "auroc": round(auroc(scores, labels), 4),
                "auprc": round(auprc(scores, labels), 4),
                "fpr_at_recall95": round(fpr_at_recall(scores, labels), 4),
                "mean_positive_score": round(sum(pos_scores) / len(pos_scores), 4),
                "mean_negative_score": round(sum(neg_scores) / len(neg_scores), 4),
            }
            print(json.dumps(report))
            all_reports.append(report)

    out_path = args.out_dir / "zero_shot_baseline_v1.json"
    out_path.write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
    print(f"\nout: {out_path}")


if __name__ == "__main__":
    main()
