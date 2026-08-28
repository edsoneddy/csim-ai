# csim-ai: neural-augmented plagiarism detection for programming judges

## 1. Problem and motivation

Programming judges need to flag plagiarized submissions among many
independent, correct solutions to the same problem. This is **plagiarism
detection**, not clone detection: two students who independently write
correct, idiomatic solutions to the same problem are a negative, not a
positive, even though their code may look similar simply because the
problem constrains the solution space.

[csim](https://github.com/edsoneddy/csim), the predecessor to this
project, detects plagiarism via ANTLR4 parse-tree normalization and Tree
Edit Distance (TED): it strips cosmetic differences (whitespace,
identifier names) and measures structural similarity on what remains.
TED is precise on structural transformations but has a known weak point,
consistent with the wider plagiarism-detection literature: as a rewrite
gets more semantic and less structural -- an LLM rewording code while
preserving behavior, for instance -- a purely structural signal
degrades. csim-ai's premise is that a contrastively fine-tuned neural
encoder can pick up exactly where TED gets weak, and that combining the
two signals should beat either alone.

The project is organized as a sequence of phases, each ending in either
a concrete artifact or, at two points, a **go/no-go decision** gated on a
held-out test-split measurement:

- **Decision 1** (Fase 3): does a contrastively fine-tuned bi-encoder
  beat zero-shot embedding similarity on the plagiarism levels where TED
  is expected to be weak?
- **Decision 2** (Fase 4): does fusing the bi-encoder with TED beat
  [Dolos](https://dolos.ugent.be/) (winnowing-based token similarity), a
  real, independently-built plagiarism detector, on the same data?

Both decisions were run to completion on the actual test split, not
assumed. The rest of this report walks through how, and what came out.

## 2. Dataset and evaluation protocol

**Data.** 299 competitive-programming problems, 7,879 Python
submissions (median 6/problem, heavily right-skewed, max 296), pulled
from a folder tree external to this repo. 22.4% of submissions are exact
raw-text duplicates of another submission to the same problem -- with no
`user_id`/`verdict`/`timestamp` to distinguish resubmission from
copying, these are excluded from negative sampling rather than guessed
at.

**Split.** A single frozen, problem-level 70/15/15 train/dev/test split
(seed 42, stratified by submissions-per-problem so dev/test aren't
dominated by thin problems), fixed once in Fase 0 and never touched
again except to read from it. The test split stayed completely untouched
until the final evaluation of each decision -- no metric computed on it
fed back into any modeling choice.

**Labels.** There is no labeled plagiarism data for this dataset (no
judge logs, no plagiarism flags). Positives are synthetically generated:
six escalating levels (L1-L6) of code transformation applied to real
submissions, described below. Negatives are same-problem,
non-near-duplicate pairs -- genuinely independent solutions -- sampled
from the real submission pool, with a hard-negative exclusion rule
(csim score >= 0.95 or exact duplicate) removing near-duplicate pairs
that would otherwise contaminate the negative pool with disguised
positives.

**Metric.** AUROC (with AUPRC and FPR@recall95 as secondary metrics) on
a shared negative pool against each level's positives, separately per
level and as a mean across L4-L6 -- the three levels the project brief
identifies as the ones a purely structural signal should struggle with,
and therefore the ones the two decisions are gated on.

## 3. Synthetic plagiarism levels (Fase 1)

Since submissions aren't executed (no test cases, no sandboxing
infrastructure), correctness of a synthetic mutation has to be argued
statically, not verified by running it. Each level has a different, and
decreasing, strength of static correctness guarantee:

| Level | Transform | Correctness argument | Validated pairs |
|---|---|---|---|
| L1 | Cosmetic (strip comments/docstrings, reindent) | AST-identical after stripping docstrings | 13,370 |
| L2 | Identifier renaming | Scope-aware round-trip: revert and diff ASTs exactly | 23,467 |
| L3 | Reorder independent statements/imports | Re-derived read/write independence via `ast`, not just "same statements sorted" | 4,387 |
| L4 | Control-flow rewrites (for→while, if/else→ternary) | Narrow static preconditions per rule + structural check | 14,354 |
| L5 | Inline single-return functions | Duplicate-evaluation and shadowing checks per call site | 696 |
| L6 | Free-form LLM rewrite (qwen2.5-coder:7b) | No static guarantee -- manual review only | 400 |

Total: **56,674 validated synthetic positives**, above the project's
50,000 target. L1-L5 (55,274 pairs) carry an automated, mechanically
checked correctness proof; L6 (400 pairs, a deliberately small subset)
is the lowest-confidence tier and was backed by a manual line-by-line
review of a 30-pair random sample -- 30/30 read as behaviorally correct,
including real restructuring, not just cosmetic changes. This is a spot
check, not a proof: 30/400 is a ~7.5% audit, and if L6 turns out to carry
a lot of weight in a future model, that sample should grow before being
trusted further.

Building the L1-L5 generators surfaced several real correctness bugs
along the way -- a rename that broke a call site passing the renamed
parameter as a keyword argument, an equivalence checker that falsely
rejected 43% of valid L2 mutations before being redesigned, an `elif`
handling bug in the L4 control-flow rewriter, a text-substitution bug in
L5 that duplicated evaluation of a shadowed variable -- each caught by
testing against thousands of real submissions rather than by inspection,
and each fixed with a narrower, more explicit precondition rather than a
broader but riskier general-purpose transform. Full detail, including
the exact bug mechanics, is in [DEVELOPMENT.md](DEVELOPMENT.md)'s Fase 1
sections; the point
worth carrying into this report is methodological: every mutation level
that claims correctness backs that claim with a *mechanical* check
(AST diff, independence analysis, round-trip verification), not
inspection or the transform's own internal logic.

## 4. Zero-shot baseline (Fase 2)

Before fine-tuning anything, Fase 2 measured how far off-the-shelf code
embeddings (UniXcoder-base, CodeRankEmbed) already are from solving this
task, per level, on the dev split.

| Level | UniXcoder AUROC | CodeRankEmbed AUROC |
|---|---|---|
| L1 (cosmetic) | 0.989 | 0.992 |
| L2 (rename) | **0.800** | **0.776** |
| L3 (reorder) | 0.9999 | 0.9999 |
| L4 (control flow) | 0.979 | 0.990 |
| L5 (inline) | 0.988 | 0.999 |
| L6 (LLM rewrite) | **0.608** | **0.681** |

The embedding collapse the literature reports is confirmed, and it's
sharper on this dataset than the usual "structural changes are easy,
semantic changes are hard" story predicts: **L2 (pure identifier
renaming) is the second-worst level for both models**, worse than L4
(control-flow rewrites). Off-the-shelf code embeddings here are
apparently more sensitive to identifier spelling than to structural
changes -- the opposite of what a human reviewer would guess, and a
concrete, falsifiable target for fine-tuning rather than a vague "do
better on hard cases."

## 5. Contrastive fine-tuning and Decision 1 (Fase 3)

**Setup.** UniXcoder-base fine-tuned with symmetric InfoNCE loss
(temperature swept over 0.05/0.07/0.1, see below) over in-batch
(anchor, positive) pairs, batches drawn to include two submissions per
problem where possible so in-batch negatives cover both easy
(different-problem) and hard (same-problem) cases without any
special-cased negative mining. Each anchor's positive is a uniformly
random L1-L6 mutation of it, drawn from the train split only. Dev
evaluation ran periodically against the exact zero-shot protocol above,
so training progress was always directly comparable to the Fase 2
numbers, and the best checkpoint by mean L4-L6 dev AUROC was kept.

**Temperature.** A short (600-step) comparison across temperatures 0.05/
0.07/0.1 showed 0.05 leading clearly at the 600-step mark -- but running
it out to the full 1500 steps showed **the early lead didn't hold**:
temperature 0.07's best (step 1350, 0.9815 dev AUROC) ended up ahead of
0.05's best (0.9784). A short partial-training comparison was a
misleading signal for where the full run lands; 0.07 was kept as the
final choice.

**Result -- Decision 1 (test split, untouched until this point):**

| Level | Zero-shot (test) | Fine-tuned (test) |
|---|---|---|
| L1 | 0.989 | 0.998 |
| L2 | 0.836 | 0.999 |
| L3 | 1.000 | 1.000 |
| L4 | 0.995 | 0.9997 |
| L5 | 0.997 | 0.998 |
| L6 | **0.583** | **0.931** |
| **mean L4-L6** | **0.858** | **0.976** |

**The fine-tune clearly beats zero-shot on held-out test data**: mean
L4-L6 AUROC 0.858 -> 0.976 (+0.118), every level moved up or stayed at
ceiling, none regressed. The improvement concentrates exactly where Fase
2 found the weaknesses (L2: 0.800 dev -> 0.999; L6: 0.608 dev -> 0.950
dev), not spread uniformly across levels that were already near-ceiling
-- evidence the fine-tune is targeting the actual weak points rather
than just overfitting a generically higher similarity everywhere. The
test numbers track the dev numbers closely (0.9815 dev vs. 0.976 test,
~0.005 lower) -- a small, healthy generalization gap.

**Decision 1: yes.** The contrastive fine-tune earns its place as a
signal.

## 6. Fusion with structural similarity and Decision 2 (Fase 4)

A fine-tuned encoder beating its own zero-shot baseline doesn't by
itself justify shipping a neural model in a plagiarism detector -- the
real bar is whether it adds anything over a simpler, already-existing
detector. **Dolos**, an independently built, winnowing/token-based
similarity tool, was installed and run on this project's own test split
(not taken from the literature) as that bar.

| Model | mean L4-L6 AUROC (test) |
|---|---|
| **Fusion (bi-encoder cosine + csim TED, GBDT)** | **0.9892** |
| Dolos | 0.9851 |
| Bi-encoder fine-tuned (alone) | 0.9764 |
| csim TED (alone) | 0.9140 |
| Zero-shot bi-encoder | 0.8583 |

Two results here matter more than the headline number:

1. **The bi-encoder alone loses to Dolos** (0.9764 vs. 0.9851). Neither
   of this project's two structural/neural signals beats Dolos
   standalone -- Dolos is a strong baseline on this dataset's mutation
   levels, most of which preserve substantial token-level structure even
   when they change identifiers or control flow.
2. **Fusing the bi-encoder with csim's TED score via a GBDT does beat
   Dolos, by +0.0041** (0.9892 vs. 0.9851) -- a real but thin margin, not
   the +0.03 goalpost noted earlier in planning (a number from an
   external conversation, not verifiable against any artifact in this
   repo). Two things support treating +0.0041 as a real effect rather
   than noise: the fusion score beats *both* of its own input features
   at every level (never worse than the better of the two, as a working
   GBDT fusion should behave), and it substantially reduces
   FPR@recall95 against csim TED alone on L4 (0.0003 vs. 0.538) and L5
   (0.002 vs. 0.011). Per level, Dolos still wins specifically on L6
   (0.975 vs. fusion's 0.968); the fusion's aggregate win comes from
   dominating L4/L5, not from winning everywhere.

**Decision 2: yes, keep the hybrid** -- on the strength of the two
supporting results above, despite the margin being thin. The honest
framing is that the neural signal's contribution here is real but
modest: it doesn't stand on its own against a well-built structural
baseline, and only earns a place in combination with one.

## 7. Packaging (Fase 5)

The trained artifacts (a ~500MB fine-tuned transformer, a small GBDT)
needed a runtime path that doesn't require every downstream user to
install `torch`. The bi-encoder was exported to ONNX (fp32) and wrapped,
along with the TED signal and the fusion model, behind a small Python
API (`csim_ai.Scorer`) and CLI, importable with just `onnxruntime` +
`tokenizers` + `numpy` for the bi-encoder-only path (`csim`/
`scikit-learn` are optional extras for the full hybrid).

Quantization was attempted and rejected on measurement, not assumption:
dynamic int8 quantization measurably changes the embedding *direction*
(cosine ~0.4-0.55 against the fp32 model on simple test cases) --
disqualifying, since cosine similarity is exactly what both the training
objective and the downstream fusion model depend on. fp16 preserves
quality (cosine ~0.99999) but runs ~7x *slower* than fp32 on CPU, since
`onnxruntime`'s CPU execution provider has no efficient native fp16
kernels. The package ships fp32 only; static/calibrated quantization is
left as future work if CPU inference speed becomes a real bottleneck in
production.

## 8. Limitations

- **L5 (696 pairs) and L6 (400 pairs, from a 400-submission subset) are
  small relative to L1-L4** (13k-23k each) -- their per-level metrics
  should be read as indicative, not precise, and L6's correctness rests
  on a 30-pair manual audit, not a mechanical proof.
- **Decision 2's margin is thin** (+0.0041 AUROC over Dolos). It's
  supported by two secondary results (per-level dominance, FPR
  reduction), but it is not a large, unambiguous win, and a different
  random seed or a slightly different fusion feature set could plausibly
  close or reverse it.
- **The bi-encoder alone is not competitive** with either Dolos or the
  fusion model -- its value in this system is conditional on being
  combined with a structural signal, not standalone.
- **No plagiarism labels exist for this dataset.** Every positive is
  synthetic. This is a deliberate, necessary design choice (there is no
  labeled ground truth to train or evaluate against otherwise), but it
  means the reported numbers measure detection of *this project's six
  specific transformation families*, not plagiarism in general --
  transformation strategies outside L1-L6 (e.g., translating through an
  intermediate language, heavier algorithmic restructuring) are
  untested.
- **`csim`/`torch` packaging is not yet a clean install story.** Both
  currently work together in this dev venv despite `csim`'s documented
  `numpy==1.26.4` pin, but this hasn't been resolved into a real
  dependency constraint or tested across environments.
- **No production deployment or load testing.** Latency numbers (Fase 5)
  are measured on a single CPU batch shape, not under realistic judge
  traffic.

## 9. Conclusion

Both go/no-go decisions were answered with real, held-out-test-split
evidence rather than assumption: a contrastively fine-tuned bi-encoder
meaningfully improves on zero-shot code-embedding similarity (Decision
1), and fusing it with a structural TED signal via a small GBDT beats an
independently-built plagiarism detector, Dolos, though by a real rather
than comfortable margin (Decision 2). The resulting hybrid -- bi-encoder
cosine similarity + csim TED, fused with a GBDT -- is packaged as
`csim-ai`, an ONNX-based, torch-free-by-default Python library and CLI.
The main open threads are the thinness of Decision 2's margin, the small
sample sizes at L5/L6, and the packaging/dependency story ahead of any
real distribution -- each called out above rather than smoothed over.
