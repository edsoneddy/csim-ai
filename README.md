# csim-ai

Neural-augmented Python code plagiarism detection for programming judges.
Successor to [csim](https://github.com/edsoneddy/csim) (ANTLR4 parse-tree
normalization + Tree Edit Distance), adding a contrastively fine-tuned
encoder for the L4-L6 structural/semantic plagiarism cases where pure TED
similarity degrades.

**Task**: plagiarism detection (did B derive from A?), not semantic clone
detection (does B solve the same problem as A?). Two independent correct
solutions to the same problem are a negative, not a positive.

Status: **Fase 0 - data audit and split harness.** No model code yet.

## Layout

```
src/csim_ai/       inference package (empty until Fase 5)
training/
  configs/         YAML configs, one seed per experiment
  data/            audit + split scripts, versioned artifacts
  eval/            baseline harnesses (csim, later Dolos/JPlag), versioned artifacts
```

## Fase 0: dataset audit and frozen splits

The dataset is a folder tree external to this repo:
`<dataset_dir>/<problem_id>/<submission_id>.py`. Point at it with
`--dataset-dir` or the `CSIM_AI_DATASET_DIR` env var; it is never copied
into the repo.

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[train]"

./.venv/bin/python training/data/audit.py --dataset-dir /path/to/dataset
./.venv/bin/python training/data/make_splits.py \
  --manifest training/data/artifacts/manifest_v1.jsonl \
  --config training/configs/data.yaml
```

`audit.py` walks the tree, hashes each submission to find exact duplicates
within a problem, and checks that it parses. Output:
`training/data/artifacts/manifest_v1.jsonl` (one row per submission) and
`audit_report_v1.json` (aggregate stats).

`make_splits.py` produces a frozen, problem-level train/dev/test split
(70/15/15, seed 42), stratified by submissions-per-problem bucket so
dev/test aren't dominated by thin problems. Output:
`training/data/artifacts/problem_splits_v1.json`.

### Current dataset snapshot (299 problems, 7,879 submissions)

- Submissions/problem: median 6, mean 26.4, heavily right-skewed (max 296).
  13 problems have only 1 submission (not usable as a source of same-problem
  hard negatives).
- Lines/submission: median 19, p95 52 -- short intro/competitive-programming
  solutions, not large programs.
- **22.4% of submissions are exact (raw-text) duplicates of another
  submission to the same problem** (884 duplicate groups across 220
  problems). No `user_id`/`verdict`/`timestamp` exists to tell resubmission
  from copying, so these are excluded from same-problem hard negatives (see
  below) and kept in the manifest (`dup_group_id`) as a candidate-positive
  pool for manual review.
- 0 syntax/parse failures.

External labeled datasets (ConPlag, IR-Plag, Kaggle sets) are for
evaluation only and never enter training.

## Fase 0: csim baseline and the hard-negative exclusion rule

```bash
./.venv/bin/pip install -e ".[train]"   # pulls in csim from PyPI

./.venv/bin/python training/eval/csim_baseline.py \
  --dataset-dir /path/to/dataset \
  --manifest training/data/artifacts/manifest_v1.jsonl \
  --splits training/data/artifacts/problem_splits_v1.json

./.venv/bin/python training/data/make_hard_negative_exclusions.py \
  --pairs training/eval/artifacts/csim_baseline_pairs_v1.csv
```

`csim_baseline.py` runs csim (TED/APTED) exhaustively over every
same-problem pair (307,791 pairs across 299 problems, ~2.5 min). We have no
plagiarism labels yet, so it doesn't compute AUROC -- it exists to (a)
surface pseudo-positive candidates for manual review and (b) size the
duplicate-contamination risk in same-problem pairs before we sample hard
negatives from them.

Raw-text hashing (in `audit.py`) is a lower bound on duplication: **~1% of
pairs it does *not* flag as duplicate still get a perfect csim score of
1.0** (identical normalized parse tree -- renamed variable, reformatted
whitespace, etc.), and 4.6% score >= 0.95. The rule from section 5 of the
project brief -- *"excluir de los negativos los pares que csim ya marca
como casi identicos"* -- is what closes this gap:

> **A same-problem pair is excluded from hard-negative sampling if it is an
> exact raw-text duplicate OR its csim score >= 0.95.**

`make_hard_negative_exclusions.py` applies this rule to the scored pairs
and writes the canonical list any future sampler (Fase 3) must consult:
`training/data/artifacts/hard_negative_exclusions_v1.jsonl` -- 20,027 pairs
(6,062 exact + 13,965 near-duplicate), 6.5% of all same-problem pairs, each
tagged with `reason` (`exact_hash_dup` / `csim_near_dup`) and `csim_score`.
