#!/usr/bin/env python3
"""Fase 3: contrastive fine-tune of the bi-encoder (Etapa A).

Backbone: microsoft/unixcoder-base (MIT) -- CodeSage-v2-small, the
brief's original pick, doesn't run under this Python 3.14 / transformers
5.x environment (see Fase 2 in the README); UniXcoder already has a
measured zero-shot baseline here, so it's the backbone we fine-tune to
keep the before/after comparison apples to apples.

Loss: symmetric InfoNCE over in-batch (anchor, positive) pairs. Batches
are built to include 2 submissions per problem when possible
(training.data.contrastive_batches.ContrastiveData), so in-batch
negatives include both easy negatives (different problem) and hard
negatives (same problem, different submission) without any special-cased
negative-mining logic in the loss itself.

Evaluation: same protocol as Fase 2's zero-shot baseline (same dev-split
positives per level, same shared negative pool from the csim baseline),
computed periodically against the live in-training weights so training
can be stopped as soon as dev AUROC (averaged over L4-L6, the levels
Decision 1 in the roadmap cares about) stops improving.

Usage:
    python -m training.train_biencoder \
        --dataset-dir /path/to/dataset \
        --manifest training/data/artifacts/manifest_v1.jsonl \
        --splits training/data/artifacts/problem_splits_v1.json \
        --config training/configs/biencoder.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModel, AutoTokenizer

from .data.contrastive_batches import ContrastiveData
from .eval.metrics import auprc, auroc, fpr_at_recall

MODEL_ID = "microsoft/unixcoder-base"


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return F.normalize(summed / counts, dim=-1)


class BiEncoder:
    def __init__(self, device: str):
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID).to(device)
        self.device = device

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def encode_train(self, texts: list[str], max_length: int = 512) -> torch.Tensor:
        inputs = self.tok(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(self.device)
        with torch.autocast(device_type="cuda" if self.device == "cuda" else "cpu", dtype=torch.bfloat16):
            out = self.model(**inputs)
            pooled = mean_pool(out.last_hidden_state, inputs["attention_mask"])
        return pooled.float()

    @torch.no_grad()
    def encode_eval(self, texts: list[str], batch_size: int = 64, max_length: int = 512) -> np.ndarray:
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tok(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(self.device)
            with torch.autocast(device_type="cuda" if self.device == "cuda" else "cpu", dtype=torch.bfloat16):
                out = self.model(**inputs)
                pooled = mean_pool(out.last_hidden_state, inputs["attention_mask"])
            embs.append(pooled.float().cpu().numpy())
        return np.concatenate(embs, axis=0)


def info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor, temperature: float) -> torch.Tensor:
    sim = anchor @ positive.T / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


# ---------------------------------------------------------------------------
# Dev evaluation (same protocol/data as training.eval.zero_shot_baseline)
# ---------------------------------------------------------------------------


def load_dev_eval_data(manifest_path: Path, artifacts_dir: Path, eval_artifacts_dir: Path):
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
            if row["split"] != "dev" or row["is_dup_pair"] != "0":
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
        positives_by_level[lvl] = [r for r in rows if r["split"] == "dev"]

    return manifest, negatives, positives_by_level


def run_dev_eval(encoder: BiEncoder, dataset_dir: Path, manifest, negatives, positives_by_level):
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

    encoder.eval()
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
            "auroc": round(auroc(scores, labels), 4),
            "auprc": round(auprc(scores, labels), 4),
            "fpr_at_recall95": round(fpr_at_recall(scores, labels), 4),
        }
    encoder.train()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=os.environ.get("CSIM_AI_DATASET_DIR"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "artifacts")
    args = parser.parse_args()

    if not args.dataset_dir:
        parser.error("--dataset-dir or CSIM_AI_DATASET_DIR is required")
    dataset_dir = Path(args.dataset_dir).resolve()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    data_dir = Path(__file__).parent / "data" / "artifacts"
    eval_dir = Path(__file__).parent / "eval" / "artifacts"
    train_data = ContrastiveData(dataset_dir, args.manifest, args.splits, data_dir, split="train")
    print(f"train problems: {len(train_data.problem_ids)}")

    manifest, dev_negatives, dev_positives = load_dev_eval_data(args.manifest, data_dir, eval_dir)
    print(f"dev negatives: {len(dev_negatives)}, dev positives by level: {[(k, len(v)) for k, v in dev_positives.items()]}")

    encoder = BiEncoder(device)
    encoder.train()
    optimizer = torch.optim.AdamW(encoder.model.parameters(), lr=cfg["lr"])
    n_steps = cfg["max_steps"] // cfg.get("grad_accum_steps", 1) * cfg.get("grad_accum_steps", 1)
    warmup_steps = cfg.get("warmup_steps", 100)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup_steps)
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "train_log_v1.jsonl"
    best_ckpt_dir = args.out_dir / "best_checkpoint"
    best_score = -1.0
    history = []

    t0 = time.time()
    for step in range(cfg["max_steps"]):
        batch = train_data.sample_batch(rng, cfg["n_problems_per_batch"])
        anchors = [a for a, _ in batch]
        positives = [p for _, p in batch]

        anchor_emb = encoder.encode_train(anchors, max_length=cfg["max_length"])
        positive_emb = encoder.encode_train(positives, max_length=cfg["max_length"])
        loss = info_nce_loss(anchor_emb, positive_emb, cfg["temperature"])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % cfg.get("log_every", 20) == 0:
            elapsed = time.time() - t0
            print(f"step {step + 1}/{cfg['max_steps']} loss={loss.item():.4f} n_pairs={len(batch)} elapsed={elapsed:.0f}s")

        if (step + 1) % cfg["eval_every"] == 0 or (step + 1) == cfg["max_steps"]:
            results = run_dev_eval(encoder, dataset_dir, manifest, dev_negatives, dev_positives)
            mean_l4_l6 = sum(results[lvl]["auroc"] for lvl in ("L4", "L5", "L6") if lvl in results) / 3
            entry = {"step": step + 1, "loss": round(loss.item(), 4), "dev": results, "mean_l4_l6_auroc": round(mean_l4_l6, 4)}
            history.append(entry)
            print(json.dumps(entry))
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

            if mean_l4_l6 > best_score:
                best_score = mean_l4_l6
                best_ckpt_dir.mkdir(parents=True, exist_ok=True)
                encoder.model.save_pretrained(best_ckpt_dir)
                encoder.tok.save_pretrained(best_ckpt_dir)
                print(f"  new best mean L4-L6 AUROC: {best_score:.4f} -- saved checkpoint")

    print(f"\ndone. best mean L4-L6 dev AUROC: {best_score:.4f}")
    print(f"checkpoint: {best_ckpt_dir}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
