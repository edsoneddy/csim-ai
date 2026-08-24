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

## Fase 0: csim vs. published tools on ConPlag (external, eval-only)

[ConPlag](https://zenodo.org/records/7332790) (Slobodkin & Sadovnikov,
[arXiv:2303.10763](https://arxiv.org/abs/2303.10763), CC-BY-4.0): 911
labeled Java solution pairs from 21 Codeforces problems (251 plagiarized,
660 not), in "raw" and "template-free" versions, with a fixed 230/681
train/test split. This is the only external plagiarism benchmark we've
pulled in so far -- **it's Java, ours is Python**, so treat it as a sanity
check on csim's TED signal in general, not a Python-specific number.

```bash
./.venv/bin/python training/eval/external/fetch_conplag.py
./.venv/bin/python training/eval/conplag_baseline.py --version 1   # raw
./.venv/bin/python training/eval/conplag_baseline.py --version 2   # template-free
```

`conplag_baseline.py` reproduces the paper's own protocol so csim lands in
the same table as their reported numbers: tune a similarity threshold on
the 230-pair train split to maximize F1.5 (beta=1.5, recall weighted
higher -- a false accusation is the real cost), report precision/recall/F1.5
on the untouched 681-pair test split, plus AUROC/AUPRC/FPR@recall95 (the
threshold-free metrics section 7 of the brief wants for our own model).

| Tool | Raw P / R / F1.5 | Template-free P / R / F1.5 |
|---|---|---|
| JPlag | 0.66 / 0.83 / **0.77** | 0.75 / 0.83 / **0.80** |
| MOSS | 0.77 / 0.71 / 0.72 | 0.66 / 0.81 / 0.75 |
| SIM | 0.69 / 0.74 / 0.72 | 0.73 / 0.75 / 0.74 |
| Dolos | 0.68 / 0.65 / 0.66 | 0.72 / 0.83 / 0.79 |
| BPlag | 0.45 / 0.61 / 0.55 | 0.52 / 0.87 / 0.72 |
| Sherlock | 0.34 / 0.76 / 0.55 | 0.39 / 0.81 / 0.60 |
| **csim (ours)** | 0.27 / 0.97 / 0.54 | 0.39 / 0.65 / 0.54 |

(JPlag/MOSS/SIM/Dolos/BPlag/Sherlock rows are copied from Table I of the
paper -- same test split, same F1.5, not rerun by us.)

**csim ties the weakest tools in the paper's own comparison** (Sherlock
and BPlag, which the authors themselves flag as low-precision /
oversensitive), and sits well below the token-based tools (JPlag, MOSS,
SIM, Dolos) on both dataset versions. Extra numbers not in the paper:
AUROC 0.65 (raw) / 0.69 (template-free), FPR@recall95 **0.95 / 0.92** --
to catch 95% of plagiarism you'd have to flag ~95%/92% of clean pairs too,
i.e. useless as a standalone product-grade detector at high recall. This
is external, Java, evidence for exactly the gap section 3 of the brief is
about; it is not yet evidence about our Python dataset or about L1-L6
levels specifically, since ConPlag doesn't label by level.

Caveats: csim has only one tunable knob (similarity threshold) vs. 400+
tuned configurations per tool in the paper, so this isn't fully apples to
apples in the other direction. `java_24` grammar was picked over `java_20`
without a rigorous comparison (single-pair spot check: 0.65 vs. 0.50 on
one pair) -- worth revisiting if csim's Java path ever matters, though it
doesn't for the Python-only product.
