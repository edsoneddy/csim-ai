"""Shared ranking metrics for pairwise-similarity baselines and eval runs
(Fase 0 csim/ConPlag, Fase 2 zero-shot, Fase 3+ fine-tuned). Pure Python,
no numpy/torch dependency, so it's usable from any eval script regardless
of what else that script has installed.
"""
from __future__ import annotations


def auroc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney U / rank-sum AUROC, average ranks on ties."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    combined = sorted(zip(scores, labels), key=lambda t: t[0])
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


def auprc(scores: list[float], labels: list[int]) -> float:
    """Step-function average precision (sklearn-compatible, no interpolation)."""
    n_pos = sum(labels)
    ordered = sorted(zip(scores, labels), key=lambda t: -t[0])
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    i = 0
    while i < len(ordered):
        j = i
        s = ordered[i][0]
        while j < len(ordered) and ordered[j][0] == s:
            tp += ordered[j][1] == 1
            fp += ordered[j][1] == 0
            j += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return ap


def fpr_at_recall(scores: list[float], labels: list[int], target_recall: float = 0.95) -> float:
    """The product metric (section 7 of the brief): fraction of clean
    pairs flagged to catch `target_recall` of the plagiarized ones."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    tp = fp = 0
    for _, y in sorted(zip(scores, labels), key=lambda t: -t[0]):
        tp += y == 1
        fp += y == 0
        if tp / n_pos >= target_recall:
            return fp / n_neg
    return 1.0
