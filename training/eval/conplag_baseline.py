#!/usr/bin/env python3
"""Fase 0: csim baseline on the external ConPlag benchmark (evaluation
only, never used for training -- section 5 of the project brief).

ConPlag: 911 labeled Java solution pairs from 21 Codeforces problems
(251 plagiarized, 660 not), Slobodkin & Sadovnikov 2023
(arXiv:2303.10763), CC-BY-4.0, https://zenodo.org/records/7332790. Two
versions: "raw" (version_1) and "template-free" (version_2, contest
boilerplate manually stripped).

Reproduces the paper's own evaluation protocol so csim lands in the same
table as their reported JPlag/MOSS/SIM/Dolos/Sherlock/BPlag numbers
(their Table I): tune a similarity threshold on the 230-pair train split
to maximize F-beta (beta=1.5, recall weighted higher -- false accusations
are the real cost), then report precision/recall/F1.5 on the untouched
681-pair test split. Also reports AUROC, AUPRC and FPR@recall=95% on the
test split, since those are the threshold-free metrics section 7 of the
brief requires for our own model later.

Usage:
    python training/eval/conplag_baseline.py --version 1
    python training/eval/conplag_baseline.py --version 2
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import csim

LANG = "java_24"
TED_ALGORITHM = "apted"
BETA = 1.5
CONPLAG_DIR = Path(__file__).parent / "external" / "conplag" / "extracted"


def load_labels() -> list[dict]:
    with (CONPLAG_DIR / "versions" / "labels.csv").open() as f:
        return list(csv.DictReader(f))


def load_split_keys(name: str) -> set[str]:
    path = CONPLAG_DIR / "versions" / f"{name}_pairs.csv"
    return {line.strip() for line in path.open() if line.strip()}


def score_all_pairs(version: int, labels: list[dict]) -> dict[str, tuple[float, int]]:
    """Returns {pair_key: (csim_score, label)} for every pair, skipping parse failures."""
    base = CONPLAG_DIR / "versions" / f"version_{version}"
    scored = {}
    for row in labels:
        key = f"{row['sub1']}_{row['sub2']}"
        d = base / key
        a = (d / f"{row['sub1']}.java").read_text(encoding="utf-8", errors="ignore")
        b = (d / f"{row['sub2']}.java").read_text(encoding="utf-8", errors="ignore")
        score = csim.Compare(row["sub1"], a, row["sub2"], b, lang=LANG, ted_algorithm=TED_ALGORITHM)
        if score is None:
            continue
        scored[key] = (score, int(row["verdict"]))
    return scored


def auroc(scored: list[tuple[float, int]]) -> float:
    """Mann-Whitney U / rank-sum AUROC, average ranks on ties."""
    n_pos = sum(y for _, y in scored)
    n_neg = len(scored) - n_pos
    combined = sorted(scored, key=lambda t: t[0])
    ranks: dict[int, float] = {}
    i, rank = 0, 1
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2
        for k in range(i, j):
            ranks[k] = avg_rank
        rank += j - i
        i = j
    rank_sum_pos = sum(ranks[k] for k, (_, y) in enumerate(combined) if y == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auprc(scored: list[tuple[float, int]]) -> float:
    """Step-function average precision (sklearn-compatible, no interpolation)."""
    n_pos = sum(y for _, y in scored)
    ordered = sorted(scored, key=lambda t: -t[0])
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    i = 0
    while i < len(ordered):
        j = i
        score = ordered[i][0]
        while j < len(ordered) and ordered[j][0] == score:
            tp += ordered[j][1] == 1
            fp += ordered[j][1] == 0
            j += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return ap


def fpr_at_recall(scored: list[tuple[float, int]], target_recall: float = 0.95) -> float:
    n_pos = sum(y for _, y in scored)
    n_neg = len(scored) - n_pos
    tp = fp = 0
    for _, y in sorted(scored, key=lambda t: -t[0]):
        tp += y == 1
        fp += y == 0
        if tp / n_pos >= target_recall:
            return fp / n_neg
    return 1.0  # never reached target recall


def precision_recall_fbeta(scored: list[tuple[float, int]], threshold: float, beta: float) -> tuple[float, float, float]:
    tp = sum(1 for s, y in scored if s >= threshold and y == 1)
    fp = sum(1 for s, y in scored if s >= threshold and y == 0)
    fn = sum(1 for s, y in scored if s < threshold and y == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    b2 = beta**2
    fbeta = (1 + b2) * precision * recall / (b2 * precision + recall) if (precision or recall) else 0.0
    return precision, recall, fbeta


def best_threshold(train_scored: list[tuple[float, int]], beta: float) -> float:
    """Grid search over every observed score, matching the paper's threshold sweep."""
    candidates = sorted({s for s, _ in train_scored})
    best_t, best_f = candidates[0], -1.0
    for t in candidates:
        _, _, f = precision_recall_fbeta(train_scored, t, beta)
        if f > best_f:
            best_t, best_f = t, f
    return best_t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, choices=[1, 2], required=True)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "artifacts")
    args = parser.parse_args()

    labels = load_labels()
    all_scored = score_all_pairs(args.version, labels)
    train_keys = load_split_keys("train")
    test_keys = load_split_keys("test")

    train_scored = [all_scored[k] for k in train_keys if k in all_scored]
    test_scored = [all_scored[k] for k in test_keys if k in all_scored]

    threshold = best_threshold(train_scored, BETA)
    precision, recall, fbeta = precision_recall_fbeta(test_scored, threshold, BETA)

    report = {
        "dataset": "ConPlag",
        "dataset_version": "raw" if args.version == 1 else "template-free",
        "method": "csim (TED/APTED)",
        "lang_grammar": LANG,
        "n_pairs_scored": len(all_scored),
        "n_parse_failures": len(labels) - len(all_scored),
        "threshold_tuning": {
            "train_n": len(train_scored),
            "tuned_threshold": round(threshold, 4),
            "beta": BETA,
        },
        "test_set": {
            "n": len(test_scored),
            "n_positive": sum(y for _, y in test_scored),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1.5": round(fbeta, 4),
            "auroc": round(auroc(test_scored), 4),
            "auprc": round(auprc(test_scored), 4),
            "fpr_at_recall95": round(fpr_at_recall(test_scored), 4),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"conplag_csim_v{args.version}_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nreport: {out_path}")


if __name__ == "__main__":
    main()
