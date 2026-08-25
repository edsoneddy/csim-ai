"""Batch construction for InfoNCE contrastive training (Fase 3).

Each batch draws several problems and, when possible, 2 submissions per
problem, so in-batch negatives naturally include both easy negatives
(different problems) and hard negatives (same problem, different
submission) -- the loss function doesn't need special-case logic for
hard negatives, the batch composition does that work. Same-problem
submission pairs flagged unsafe in hard_negative_exclusions_v1.jsonl
(Fase 0: exact/near-duplicate, i.e. plausibly real unlabeled plagiarism)
are never placed together, since treating them as a negative would be
wrong.

For each selected (problem, submission) slot, one of that submission's
L1-L6 synthetic mutations (matching `split`) is sampled uniformly at
random as its positive -- so every training step sees a random mutation
level rather than a fixed curriculum.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


class ContrastiveData:
    def __init__(
        self,
        dataset_dir: Path,
        manifest_path: Path,
        splits_path: Path,
        synthetic_dir: Path,
        split: str = "train",
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split

        manifest: dict[tuple[str, str], str] = {}
        with Path(manifest_path).open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                manifest[(row["problem_id"], row["submission_id"])] = row["path"]
        self.manifest = manifest

        excluded: set[tuple[str, frozenset]] = set()
        excl_path = Path(synthetic_dir) / "hard_negative_exclusions_v1.jsonl"
        if excl_path.exists():
            with excl_path.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    excluded.add((row["problem_id"], frozenset((row["submission_a"], row["submission_b"]))))
        self.excluded = excluded

        positives_by_sub: dict[tuple[str, str], list[str]] = {}
        for level in range(1, 7):
            path = Path(synthetic_dir) / f"synthetic_l{level}_v1.jsonl"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    if row["split"] != split:
                        continue
                    key = (row["problem_id"], row["submission_id"])
                    positives_by_sub.setdefault(key, []).append(row["mutated_code"])
        self.positives_by_sub = positives_by_sub

        by_problem: dict[str, list[str]] = {}
        for pid, sid in positives_by_sub:
            by_problem.setdefault(pid, []).append(sid)
        self.by_problem = by_problem
        self.problem_ids = list(by_problem.keys())

    def read_code(self, problem_id: str, submission_id: str) -> str:
        path = self.dataset_dir / self.manifest[(problem_id, submission_id)]
        return path.read_text(encoding="utf-8", errors="ignore")

    def sample_batch(self, rng: random.Random, n_problems: int) -> list[tuple[str, str]]:
        """Up to 2 (anchor_code, positive_code) pairs per sampled problem."""
        pairs: list[tuple[str, str]] = []
        problems = rng.sample(self.problem_ids, min(n_problems, len(self.problem_ids)))
        for pid in problems:
            subs = self.by_problem[pid]
            chosen = rng.sample(subs, min(2, len(subs)))
            if len(chosen) == 2 and (pid, frozenset(chosen)) in self.excluded:
                # the random pair is unsafe -- look for any other partner
                # for chosen[0] before falling back to just one submission
                for candidate in subs:
                    if candidate not in chosen and (pid, frozenset((chosen[0], candidate))) not in self.excluded:
                        chosen[1] = candidate
                        break
                else:
                    chosen = chosen[:1]
            for sid in chosen:
                anchor = self.read_code(pid, sid)
                positive = rng.choice(self.positives_by_sub[(pid, sid)])
                pairs.append((anchor, positive))
        return pairs
