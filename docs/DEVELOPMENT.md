# csim-ai: development log

Phase-by-phase build log for csim-ai (Fase 0 through Fase 5): dataset
audit, synthetic plagiarism generation, baselines, the contrastive
fine-tune, the fusion scorer, and packaging -- commands, exact numbers,
and the real bugs/dead-ends hit along the way. This is the operational
record for anyone extending the project, not end-user docs.

- For **installing and using csim-ai** (CLI/API), see the main
  [README.md](../README.md).
- For the **narrative writeup** of the whole project (problem statement,
  methodology, both decisions, results, limitations), see
  [REPORT.md](REPORT.md).

## Layout

```
src/csim_ai/       inference package -- ONNX bi-encoder + csim TED + GBDT fusion (Fase 5)
tests/             pytest smoke tests for src/csim_ai
training/
  configs/         YAML configs, one seed per experiment
  data/            audit + split scripts, versioned artifacts
  eval/            baseline harnesses (csim, zero-shot, Dolos), versioned artifacts
    dolos_tool/    local npm install of @dodona/dolos (Node >=22 needed)
  mutate/          L1-L6 synthetic plagiarism generator (Fase 1)
  scorer/          Fase 4 fusion scorer: feature computation, GBDT train/eval
  export_onnx.py   Fase 5: exports training/artifacts/best_checkpoint to ONNX
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

## Fase 1: L1 synthetic positives (cosmetic mutations)

We don't execute code or test cases to validate a mutation (confirmed in
Fase 0), so correctness has to come from the transform itself, not from
testing after the fact. Each L1 rule in `training/mutate/l1_cosmetic.py`
only touches trivia (comments/whitespace) or removes a docstring-position
statement (a no-op at runtime) -- never anything `ast.parse` would see
differently beyond that. `training/mutate/validate.py` is the ground
truth: a mutation is kept only if it still parses and its AST is
identical to the original's once docstrings are stripped from both sides.

```bash
./.venv/bin/pip install -e ".[train]"   # pulls in libcst

./.venv/bin/python -m training.mutate.generate --level L1 \
  --dataset-dir /path/to/dataset \
  --manifest training/data/artifacts/manifest_v1.jsonl \
  --splits training/data/artifacts/problem_splits_v1.json
```

Rules: `strip_docstrings`, `strip_comments`, `reindent` (2/3/8-space
width). Each submission gets up to 3 random variants; duplicates and
no-op variants (e.g. a file with nothing to strip) are dropped before
validation.

Result on the full dataset: **13,370 validated L1 pairs** from 7,879
submissions (1.7/submission), 0 failed the equivalence check, 3 files
libcst's tokenizer couldn't parse (stricter than `ast.parse` on mixed
tabs/spaces, same issue csim hit in Fase 0 -- skipped, not fatal). Output:
`training/data/artifacts/synthetic_l1_v1.jsonl`, one row per validated
mutation: `problem_id`, `submission_id`, `split` (inherited from the
source submission's problem), `level`, `rule`, `mutated_code`,
`validated_by_execution: false`.

## Fase 1: L2 synthetic positives (identifier renaming)

`training/mutate/l2_rename.py` uses libcst's `ScopeProvider` to group
every `Name` occurrence bound to the same local variable/parameter (never
imports, builtins, attribute names, keyword-argument names, or a name
touched by `global`/`nonlocal` anywhere in the file) and renames each
group to a fresh identifier that collides with nothing in the file.

Two real bugs came up building this, both worth remembering:

1. **Renaming a parameter breaks a call site that uses it as a keyword
   argument** (`def f(x): ...` then `f(x=1)` -- renaming `x` to `curr`
   leaves the call passing `x=1` to a parameter now called `curr`).
   libcst's scope analysis correctly does *not* track keyword-argument
   names as variable references, so this doesn't show up as a scope
   conflict -- it has to be excluded separately. Fixed by collecting every
   name used as a call-site keyword anywhere in the file and never
   renaming it.
2. **A spelling-based equivalence check is wrong for this dataset.** The
   first version of the validator canonicalized identifiers by relabeling
   every distinct name-string to `ID<n>` and comparing dumps -- but this
   dataset reuses generic names (`cases`, `n`, `result`...) across
   unrelated scopes constantly, e.g. a module-level `cases` list and an
   unrelated function parameter also called `cases`. Spelling-based
   canonicalization conflates the two and flags a correct rename as
   broken. It rejected **43% of otherwise-valid mutations** on the first
   full run. Fixed by replacing it with a round-trip check
   (`l2_rename._verify_roundtrip`): re-derive scope groups from scratch on
   the *mutated* code, revert using those freshly-derived groups, and
   require an exact AST match against the original. This is scope-aware
   instead of spelling-aware, so it doesn't have that false-positive, and
   it still catches real bugs (a missed occurrence or a name collision
   both show up as a mismatch after reverting).

```bash
./.venv/bin/python -m training.mutate.generate --level L2 \
  --dataset-dir /path/to/dataset \
  --manifest training/data/artifacts/manifest_v1.jsonl \
  --splits training/data/artifacts/problem_splits_v1.json
```

Result: **23,467 validated L2 pairs** from 7,879 submissions
(3.0/submission -- almost every attempt succeeds now that the validator
isn't producing false negatives), 0 rejected by the round-trip check.
Output: `training/data/artifacts/synthetic_l2_v1.jsonl`, same schema as
L1's, `rule` recorded as `rename:old1->new1,old2->new2,...`.

## Fase 1: L3 synthetic positives (reorder independent statements)

`training/mutate/l3_reorder.py` permutes two narrow, statically-decidable
families: contiguous runs of plain imports with no bound-name collisions,
and contiguous runs of single-target assignments that are call-free (no
`input()`/`print()`/any function or method call -- order of side effects
is observable and can't be checked without executing) and have
pairwise-disjoint read/write name sets.

Order-only reordering has a validation trap worth remembering: comparing
"same statements, sorted" between original and mutated (`order_equivalent`
in validate.py) only proves nothing was *added, dropped, or edited* -- it
can't tell a safe reorder from an unsafe one, because both produce the
same sorted form. `a = 1; b = a + 1` sorted is indistinguishable from
`b = a + 1; a = 1` sorted, even though swapping them is a real bug (`b`
would read a stale/undefined `a`). So the actual safety property
(pairwise independence) is re-derived with stdlib `ast`
(`assignments_mutually_independent`, `imports_mutually_independent`) on
the *candidate group itself*, independent of `l3_reorder.py`'s libcst-based
scan, before a reorder is ever applied -- not something `order_equivalent`
can verify after the fact.

A second, non-safety issue showed up while testing: scanning only the
*maximal* run of shape-matching statements missed obvious cases. In
`a = 1; b = 2; c = a + b`, all three lines are shape-matching simple
assignments, but the maximal-run independence check fails for all three
at once (because of `c`), silently hiding the safe `a = 1; b = 2` swap
sitting right there. Fixed by checking every sub-window of a maximal run,
not just the whole thing.

Not handled in this version: function/class def reordering, and
assignments with tuple/attribute/subscript targets or augmented
assignment -- narrower shapes were enough for real coverage without the
extra risk of getting definition-time evaluation order wrong.

```bash
./.venv/bin/python -m training.mutate.generate --level L3 \
  --dataset-dir /path/to/dataset \
  --manifest training/data/artifacts/manifest_v1.jsonl \
  --splits training/data/artifacts/problem_splits_v1.json
```

Result: **4,387 validated L3 pairs** from 7,879 submissions
(0.56/submission -- lower than L1/L2 since it needs >=2 adjacent
independent statements, less common in short scripts: 3,915
`reorder_assign`, 472 `reorder_import`), 0 rejected. Output:
`training/data/artifacts/synthetic_l3_v1.jsonl`, same schema, `rule`
recorded as `reorder_assign:<n>` / `reorder_import:<n>`.

**Running total after L3: 41,224 synthetic positives** (13,370 + 23,467 +
4,387) against the roadmap's >=50k target for Fase 1, with L4-L6 still to
come.

## Fase 1: L4 synthetic positives (control-flow equivalences)

`training/mutate/l4_control_flow.py` handles two rules: `for i in
range(...): BODY` -> `while`, and `if COND: x = A else: x = B` ->
`x = A if COND else B`. Unlike L1-L3, these genuinely restructure code, so
there's no single generic equivalence check possible afterward -- see the
module docstring for why (a naive "same statements, different shape"
comparison can't distinguish a correct for/while translation from a
subtly wrong one). Correctness rests on narrow, statically-checked
preconditions before a rewrite is attempted, plus a rule-specific
structural check on the output via stdlib `ast`.

`for_to_while` only handles `range(stop)` / `range(start, stop)` (the
3-arg step form is excluded -- the while condition's direction would
depend on the runtime sign of an arbitrary expression), requires a plain
`Name` loop target, no `for...else`, no `continue` belonging to *this*
loop (nested loops/functions are fine, they get their own `continue`),
and the loop variable must not be reassigned anywhere in the body. The
range's stop bound is always captured into a fresh temp evaluated once,
matching `range()`'s own eager-evaluation semantics -- inlining it
directly into the while condition would re-evaluate it every iteration,
which is wrong whenever it's not a bare name/constant (e.g.
`range(len(items))` if `items` could change size).

`if_else_to_ternary` requires both branches to be exactly one `Assign` to
the same single `Name` target (this also naturally excludes `elif` --
libcst represents it as a nested `If`, which fails the "exactly one
Assign" shape check on its own, no special-casing needed for that part).

Three real bugs turned up building this, all found by testing against the
real dataset (1,500+ submissions, 3 mutations each) rather than by
inspection alone:

1. **Single-line `for i in range(n): body` has a different tree shape**
   (`SimpleStatementSuite` instead of `IndentedBlock`), and splicing the
   `i += 1` increment the way the multi-line case needs produced a
   malformed tree (a statement nested where only a small-statement
   belongs). Fixed by restricting the rule to multi-line bodies.
2. **libcst represents `elif` as a nested `If` sitting directly in the
   parent's `orelse` slot -- not wrapped in an `Else`.** Since `If` also
   happens to have a `.body` attribute, code that assumed `orelse` is
   always `Else` didn't crash, it silently read the elif's own
   then-branch as if it were the else value, discarding the elif's
   condition entirely.
3. Once that was type-checked away, a related issue remained: when the
   matched `If` *is* such an elif-position node, replacing it with a bare
   assignment statement is invalid there -- the parent's `orelse` slot
   expects `If | Else | None`. Fixed by detecting the position (via
   libcst's `ParentNodeProvider`) and wrapping the replacement in an
   `Else` when needed.

```bash
./.venv/bin/python -m training.mutate.generate --level L4 \
  --dataset-dir /path/to/dataset \
  --manifest training/data/artifacts/manifest_v1.jsonl \
  --splits training/data/artifacts/problem_splits_v1.json
```

Result: **14,354 validated L4 pairs** from 7,879 submissions (14,121
`for_to_while`, 233 `if_else_to_ternary` -- for-range loops are far more
common than single-assign if/else in this dataset), 0 rejected. Output:
`training/data/artifacts/synthetic_l4_v1.jsonl`, same schema, `rule`
recorded as `for_to_while` / `if_else_to_ternary`.

Not handled in this version: `range()` with a step argument,
comprehension <-> loop (deferred -- meaningfully more complex than either
rule here, would need its own careful precondition design the same way
these did).

**Running total after L4: 55,578 synthetic positives** (41,224 + 14,354)
-- already past the roadmap's >=50k target for Fase 1.

## Fase 1: L5 synthetic positives (inline simple functions)

The brief's L5 groups several different patterns (extract/inline
functions, list<->deque, dict<->defaultdict, accumulator-pattern
rewrites) that vary a lot in how hard they are to make safe without
execution. Rather than force shallow coverage of all four,
`training/mutate/l5_inline.py` implements only the narrowest one well:
inlining a function whose entire body is a single `return EXPR`, at a
call site with exactly matching positional arguments. Container swaps,
accumulator rewrites, and extracting a block *into* a new function (the
reverse direction) are deferred -- each would need its own precondition
design the same way for_to_while/if_else_to_ternary did in L4, and the
Fase 1 target is already cleared without them.

Two hazards specific to inlining, beyond the usual "does it still parse":

1. **Duplicate evaluation.** A parameter used more than once in the body
   can only be inlined if its argument is a bare Name or literal Constant
   -- duplicating a Name/Constant has no effect, but duplicating a Call
   would evaluate it multiple times instead of once. Checked per call
   site, so the same function can be inlined at one call and skipped at
   another depending on what's passed there.
2. **Shadowing.** A nested lambda or comprehension inside the body that
   rebinds a parameter name as its own loop/lambda variable (e.g.
   `def f(x): return [x for x in range(x)]`) would have that unrelated
   binding incorrectly rewritten by a blind substitution. Any such
   collision excludes the whole function.

Two more real bugs turned up building the substitution itself (again
found by testing against 2,000+ real submissions, not by inspection):

1. The rendering used Python's `tokenize` module to substitute each
   parameter name in the function body's source text (safer than a plain
   string replace, which would also corrupt a longer identifier like
   `x1` when replacing `x`) -- but the "end of text" position for the
   final slice was computed as `(len(lines), 0)`, which is only correct
   when the text ends in a newline. For a single physical line with no
   trailing newline, that pointed *before* the actual end, silently
   truncating everything after the last substitution.
2. With that fixed, a second bug remained: substituting each parameter
   one at a time, in separate passes over the *already-partially-substituted*
   text, meant a later pass could re-scan and corrupt text a former pass
   had just inserted. Concretely: inlining `def agregar(n, digitos):
   return str(n) + digitos` at `agregar(num, otra(str(n)))` -- the second
   argument has its *own*, unrelated `n`, but the sequential substitution
   clobbered it when substituting the first parameter's replacement text
   afterward. Fixed by substituting all parameters in a single
   tokenizer pass over the original text, so replacement text is never
   re-scanned.

```bash
./.venv/bin/python -m training.mutate.generate --level L5 \
  --dataset-dir /path/to/dataset \
  --manifest training/data/artifacts/manifest_v1.jsonl \
  --splits training/data/artifacts/problem_splits_v1.json
```

Result: **696 validated L5 pairs** from 7,879 submissions
(0.09/submission -- simple single-expression functions are rare in this
dataset, as expected given the narrow scope), 0 rejected. Output:
`training/data/artifacts/synthetic_l5_v1.jsonl`, same schema, `rule`
recorded as `inline:<function_name>`.

**Running total after L5: 56,274 synthetic positives** (55,578 + 696).

## Fase 1: L6 synthetic positives (free-form LLM rewrite)

Unlike L1-L5, there is no static safety argument for L6 at all -- an LLM
can silently change behavior, and per Fase 0 we don't execute code to
check output. Treated accordingly as the lowest-confidence tier: run on
a *subset*, every row tagged `validated_by_execution: false`, and backed
by an actual manual review (not just automated checks) before trusting
it -- exactly what the Fase 1 exit criterion asks for at this level.

Setup: [Ollama](https://ollama.com) running locally (GPU-accelerated on
the RTX 3060 Ti, confirmed via `nvidia-smi` inside WSL), serving
`qwen2.5-coder:7b` (Q4 quantized, ~4.7GB). `training/mutate/l6_llm.py`
prompts it per submission to rewrite the code so it "looks different"
while explicitly preserving stdin/stdout behavior, strips markdown
fences from the response, and keeps the result only if it still parses.

```bash
ollama pull qwen2.5-coder:7b   # one-time, ~4.7GB

./.venv/bin/python -m training.mutate.l6_llm \
  --dataset-dir /path/to/dataset \
  --manifest training/data/artifacts/manifest_v1.jsonl \
  --splits training/data/artifacts/problem_splits_v1.json \
  --n-samples 400
```

Result on a seeded random subset of 400 submissions: **400/400 parsed**
(0 syntax failures, 0 unreachable), ~2-5s/generation after model
warm-up. Output: `training/data/artifacts/synthetic_l6_v1.jsonl`, same
schema as other levels, `rule: "llm_rewrite"`, plus a
`heuristic_io_counts_match` field (does the rewrite have the same
input()/print() call *count* as the original -- informational only, not
a filter, since a legitimate rewrite can merge print calls; 36/400 =
9% didn't match).

**Manual review** (the actual point of this section): a seeded 30-pair
sample (`l6_manual_review_sample.jsonl`, original + rewrite side by
side) was read by hand, line by line -- notes in
`l6_manual_review_notes.md`. **30/30 read as behaviorally correct**,
including real restructuring (not just renaming): moving a `print` from
inside a function to the caller, replacing a manual counter with a
boolean-toggle expression, inlining a confusingly-shadowed nested
function. One narrow caveat found (replacing indexed access with a
plain iteration is only equivalent if the input is well-formed, which
judge input is but isn't *guaranteed* to be) -- judged acceptable, not a
rejection. Both `heuristic_io_counts_match: false` cases in the sample
were confirmed correct on reading: the heuristic missed a call because
the model renamed the reassigned-`input` variable, not because behavior
changed.

This is a manual read, not a proof -- no test cases were run against
either version, since Fase 0 ruled that out. It's the strongest check
available under that constraint, but 30/400 is a ~7.5% audit sample:
treat the batch as spot-checked and consistent, not individually
verified. If L6 pairs turn out to matter a lot for the final model
(Fase 3), revisit with a larger review sample before trusting them at
scale.

## Fase 1 total: 56,674 synthetic positives

13,370 (L1) + 23,467 (L2) + 4,387 (L3) + 14,354 (L4) + 696 (L5) + 400
(L6) -- past the roadmap's >=50k target, with a manually-reviewed L6
sample closing out the Fase 1 exit criterion.

## Fase 2: zero-shot baseline

Cosine similarity from pretrained code encoders, no fine-tuning --
Fase 2's job is just to confirm (or not) the embedding collapse the
literature reports (section 3 of the brief) on *our own* dataset, per
level, before we spend a week fine-tuning anything in Fase 3.

**CodeSage-v2-small was dropped from the comparison.** Its HF repo ships
custom `trust_remote_code` modeling code old enough that it doesn't run
under Python 3.14 / transformers 5.x without patching multiple removed
internal APIs one at a time -- fixed `Conv1D`'s import path, then hit a
missing `all_tied_weights_keys`, then a missing `get_head_mask`, with no
sign of it stopping there. This is the same Python-3.14-vs-ecosystem
friction flagged as a risk back in Fase 0; decided not worth chasing
further for a reference-point baseline. **UniXcoder-base** (MIT, via
`transformers`) and **CodeRankEmbed** (MIT, via `sentence-transformers`,
needs `trust_remote_code=True`) both loaded and ran cleanly.

CodeRankEmbed is a retrieval model (its only defined prompt is
`"Represent this query for searching relevant code: "`, meant for a text
query against a code document) -- our task is code-vs-code similarity,
not query-to-document retrieval, so both sides are embedded as plain
documents, no query prefix. Worth keeping in mind when reading its
numbers: this is an out-of-domain use of the model.

Positives: (original, mutated) pairs from each of L1-L6 (Fase 1), **dev
split only** -- test stays untouched for later phases, even though
nothing is being trained here. Negatives: the same-problem, dev-split,
non-duplicate pairs from the csim baseline (Fase 0), still excluding
everything in `hard_negative_exclusions_v1.jsonl` -- **one shared
negative pool (30,452 pairs) reused across every level and both models**,
since it doesn't depend on the encoder. Metrics match Fase 0's protocol:
AUROC, AUPRC, FPR@recall95.

```bash
./.venv/bin/pip install -e ".[torch]"   # torch, transformers, sentence-transformers

./.venv/bin/python -m training.eval.zero_shot_baseline \
  --dataset-dir /path/to/dataset \
  --manifest training/data/artifacts/manifest_v1.jsonl
```

| Level | UniXcoder AUROC | CodeRankEmbed AUROC | n positive |
|---|---|---|---|
| L1 (cosmetic) | 0.989 | 0.992 | 1,821 |
| L2 (rename) | **0.800** | **0.776** | 3,080 |
| L3 (reorder) | 0.9999 | 0.9999 | 416 |
| L4 (control flow) | 0.979 | 0.990 | 1,849 |
| L5 (inline) | 0.988 | 0.999 | 61 |
| L6 (LLM rewrite) | **0.608** | **0.681** | 53 |

Full numbers (AUPRC, FPR@recall95, mean scores) in
`training/eval/artifacts/zero_shot_baseline_v1.json`.

**The collapse is confirmed, and it's sharper than the shape the
literature reports.** L1/L3 near-perfect and L6 the worst matches
section 3's L1-L3-vs-L4-L6 story, but here **L2 (pure identifier
renaming) is the second-worst level for both models** -- worse than L4
(control-flow rewrites) and L5 (function inlining), which is not the
"easy/hard" ordering a human would guess. Renaming is usually treated as
the trivial case in plagiarism taxonomies; these off-the-shelf embeddings
are apparently more sensitive to identifier text than to structural
changes. That's a concrete, actionable target for the contrastive
fine-tune in Fase 3, not just "do better on the hard levels in general."

L6's FPR@recall95 (0.84 UniXcoder, 0.74 CodeRankEmbed -- in
`zero_shot_baseline_v1.json`) means: to catch 95% of L6-level plagiarism
with either encoder's raw cosine, 74-84% of clean same-problem pairs
would also get flagged. Unusable standalone at that level, same
conclusion Fase 0 reached for csim on ConPlag.

Caveats: L5 (n=61) and L6 (n=53) are small samples inherited from Fase
1's scope decisions there (simple-function inlining is rare; L6 only ran
on a 400-submission subset) -- read those two rows as indicative, not
precise. installing torch bumped numpy past csim's exact `==1.26.4` pin in this
venv (still installed, just version-mismatched -- see the `[ast]`/
`[torch]` conflict noted in `pyproject.toml`), so there's no in-venv
apples-to-apples csim-vs-encoder number for L1-L6 in this run; Fase 0's
csim numbers (section above) are the reference until that packaging
conflict is resolved.

## Fase 3: contrastive fine-tune (Etapa A -- bi-encoder trained to convergence)

Etapa A (bi-encoder). Backbone: **UniXcoder-base**, not CodeSage-v2-small
-- CodeSage doesn't run in this environment (Fase 2), and UniXcoder
already has a measured zero-shot baseline here, so the before/after
comparison is apples to apples.

**Batching**: `training/data/contrastive_batches.py` draws problems and,
when possible, 2 submissions per problem, so in-batch negatives include
both easy negatives (different problem) and hard negatives (same
problem) without any special-cased mining -- the batch composition does
that work, the loss doesn't need to know about it. Same-problem pairs
flagged unsafe in `hard_negative_exclusions_v1.jsonl` (Fase 0) are never
placed together. Each anchor's positive is a uniformly random L1-L6
mutation of it (train split) -- every step sees a random level.

**Loss**: symmetric InfoNCE over the in-batch similarity matrix
(temperature 0.07, untested against alternatives -- see caveats).

**A real memory bug, found by testing, not by inspection:** the first
smoke test threw intermittent CUDA OOM on the 8GB 3060 Ti -- training
still completed (PyTorch's allocator recovers), but each recovery cost
30-100+ seconds, making training look far slower than it actually was.
Root cause: fp32 throughout, and two full forward passes (anchor,
positive) kept alive at once for the backward pass, occasionally spiking
when a batch happened to include a long submission. Fixed with
`torch.autocast(dtype=torch.bfloat16)` on both the training and eval
encode paths, plus reducing `n_problems_per_batch` from 16 to 10 -- this
took step time from ~38s/step (with OOM recovery) down to ~4.4s/step
with zero OOM warnings across a 300-step run.

```bash
./.venv/bin/python -m training.train_biencoder \
  --dataset-dir /path/to/dataset \
  --manifest training/data/artifacts/manifest_v1.jsonl \
  --splits training/data/artifacts/problem_splits_v1.json \
  --config training/configs/biencoder.yaml
```

Dev evaluation runs periodically (`eval_every` in the config) using the
*exact* protocol and negative pool from Fase 2's zero-shot baseline,
against the live in-training weights, so training progress is directly
comparable to the zero-shot numbers -- and the best checkpoint by mean
L4-L6 dev AUROC (the levels Decision 1 cares about) is saved to
`training/artifacts/best_checkpoint/`.

### Result (step 1350/1500 best checkpoint, dev split, temperature 0.07)

| Level | Zero-shot UniXcoder | Fine-tuned (step 1350) |
|---|---|---|
| L1 | 0.989 | 0.998 |
| L2 | 0.800 | 0.999 |
| L3 | 0.9999 | 0.9999 |
| L4 | 0.979 | 0.9995 |
| L5 | 0.988 | 0.995 |
| L6 | **0.608** | **0.950** |
| **mean L4-L6** | **0.859** | **0.981** |

(Per-level numbers above are read off `training/artifacts/train_log_v1.jsonl`'s
step-1350 entry; L6 there is 0.9495, rounded to 0.950 in the table.)

Full per-eval history (steps 150 through 1500):

| Step | mean L4-L6 AUROC |
|---|---|
| 150 | 0.9546 |
| 300 | 0.9666 |
| 450 | 0.9644 |
| 600 | 0.9676 |
| 750 | 0.9675 |
| 900 | 0.9741 |
| 1050 | 0.9770 |
| 1200 | 0.9794 |
| **1350** | **0.9815 (best)** |
| 1500 | 0.9792 |

Run to the full planned 1500 steps (in two parts -- 0-300 in one session,
300-1500 resumed in a later one, see "Resume support" below); monotonic
improvement with normal step-to-step noise (dips at 450 and 1500, both
within ~0.002-0.003 of the surrounding trend) and a clear plateau by the
last ~150 steps -- step 1500 landed *below* step 1350's peak, so this is
a real plateau, not an early stop. `best_checkpoint/` holds the step-1350
weights, not step-1500's.

- Dev split only, as it should be at this stage -- test stays untouched
  for the final Decision 1 call.
- Only one temperature (0.07) tested so far -- the planned sweep
  (0.05/0.07/0.1) is next.
- L2's jump (0.80 -> 0.999) and L6's (0.608 -> 0.950) are the two biggest
  moves and both land on exactly the weaknesses Fase 2 found: raw
  embeddings overly sensitive to identifier spelling (L2) and to
  free-form LLM rewrites (L6). L1/L3/L4/L5 were already near-ceiling
  zero-shot and stayed there or nudged closer to 1.0 -- the fine-tune is
  fixing the two specific weak points, not uniformly inflating every
  score.

**Resume support added**, used to run this continuation.
`training/artifacts/best_checkpoint/` only ever held weights (saved via
`save_pretrained`), so it couldn't resume the exact
optimizer/scheduler/step trajectory from the step-300 checkpoint -- only
warm-start a new one. `train_biencoder.py` now also writes
`training/artifacts/last_checkpoint/` on every eval (weights +
`trainer_state.pt`: optimizer state, scheduler state, step, best score,
and python/numpy/torch/cuda RNG state, always overwritten with the
latest step, independent of `best_checkpoint/` which only updates on
improvement) so a future stop can resume exactly via `--resume-from
training/artifacts/last_checkpoint`. Both directories are gitignored
(large binaries). The step-300 checkpoint used here predated this, so
continuing from it was necessarily a warm start: `--resume-from
training/artifacts/best_checkpoint --start-step 300` -- fresh
optimizer/scheduler state, but step numbering aligned with the existing
`train_log_v1.jsonl` history. Confirmed low-risk in practice: no
discontinuity in the metric trend across the step-300 join (0.9666 ->
0.9644 -> 0.9676, i.e. within the run's normal noise band), consistent
with `warmup_steps=100` already being long saturated by step 300.

**Temperature sweep: complete.** Short 600-step comparison runs for 0.05
and 0.1 (fresh from `microsoft/unixcoder-base`, same seed/lr/warmup/
batching as `biencoder.yaml`), compared against the temperature-0.07
trajectory above at the same four eval points:

| Step | temp 0.05 | temp 0.07 | temp 0.1 |
|---|---|---|---|
| 150 | 0.9557 | 0.9546 | 0.9582 |
| 300 | 0.9670 | 0.9666 | 0.9667 |
| 450 | 0.9693 | 0.9644 | 0.9691 |
| 600 | 0.9769 | 0.9676 | 0.9739 |

temp=0.05 led clearly at the 600-step mark, so per the pre-agreed scope
decision it was resumed (via `--resume-from
training/artifacts/temp_sweep/temp0.05/last_checkpoint`, continuing the
exact trajectory, not a warm start) out to the full 1500 steps.
**Result: the early lead didn't hold.** temp=0.07's best (step 1350,
0.9815) ends up ahead of temp=0.05's best (also step 1350, 0.9784) --
the 600-step comparison was a misleading signal for where the full run
lands. `training/artifacts/best_checkpoint/` (temp=0.07) stays the model
used for the final evaluation below; temp=0.05's full run is kept at
`training/artifacts/temp_sweep/temp0.05/` for the record but not used
further. temp=0.1 was only run to 600 steps (already behind both others
there) and wasn't extended.

### Final evaluation on the test split -- Decision 1

`training/eval/final_test_eval.py` runs the same protocol as Fase 2's
zero-shot baseline and Fase 3's periodic dev eval (same shared negative
pool, same per-level synthetic positives), switched to the **test
split**, untouched until now, for both the pretrained
`microsoft/unixcoder-base` weights and the fine-tuned
`best_checkpoint/`, through the same encoder class so the only variable
is the weights.

```bash
./.venv/bin/python -m training.eval.final_test_eval \
  --dataset-dir /path/to/dataset \
  --manifest training/data/artifacts/manifest_v1.jsonl \
  --checkpoint training/artifacts/best_checkpoint
```

| Level | Zero-shot (test) | Fine-tuned (test) |
|---|---|---|
| L1 | 0.989 | 0.998 |
| L2 | 0.836 | 0.999 |
| L3 | 1.000 | 1.000 |
| L4 | 0.995 | 0.9997 |
| L5 | 0.997 | 0.998 |
| L6 | **0.583** | **0.931** |
| **mean L4-L6** | **0.858** | **0.976** |

Full numbers in `training/eval/artifacts/final_test_eval_v1.json`.

**Decision 1: the fine-tune clearly beats zero-shot on held-out test
data** -- mean L4-L6 AUROC 0.858 -> 0.976 (+0.118), and every level moved
up or stayed at ceiling, none regressed. The test numbers track the dev
numbers closely (dev mean L4-L6 was 0.9815; test is 0.976, ~0.005 lower)
-- a small, healthy generalization gap, not overfitting to the dev split
the training loop was periodically evaluating against.

csim is not part of this comparison (the zero-shot/fine-tuned comparison
is encoder-vs-encoder only) -- see Fase 4 below for csim TED numbers on
this same test split. The `==1.26.4` pin conflict noted in Fase 2 turned
out not to matter in practice: `csim` and `torch` import and run
correctly together in this venv (numpy 2.5.2, past the pin) -- confirmed
in Fase 4 by recomputing a known TED score and matching Fase 0's stored
value exactly, then computing TED at scale (tens of thousands of pairs)
without issue.

**Etapa A (bi-encoder) is done.** Next: Fase 4 (scorer + fusion).

## Fase 4: scorer + fusion (Decision 2)

Two structural/neural similarity signals -- csim's TED score and the
fine-tuned bi-encoder's cosine similarity -- combined into one score via
a GBDT, compared against Dolos (winnowing-based token similarity) as the
external reference point.

### Dolos setup

Dolos (`@dodona/dolos`, npm) needs Node.js >=22 in practice: its native
tree-sitter parser build (`node-gyp rebuild`) fails under Node 20 because
`node-gyp`'s own dependencies (not Dolos's, which only declares
`engines.node >= 18`) need a newer runtime -- confirmed by tracing the
actual `node-gyp` crash, not just the `npm warn EBADENGINE` noise, which
looked non-fatal but wasn't. Installed via a user-local `nvm` (no sudo)
into `training/eval/dolos_tool/` (gitignored `node_modules/`, tracked
`package.json`):

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"
nvm install 22

export PATH="$(dirname "$(nvm which 22)"):$PATH"   # node-gyp needs this on PATH too, not just `node`
npm install --prefix training/eval/dolos_tool @dodona/dolos
```

### Dolos baseline (test split)

`training/eval/dolos_baseline.py` runs one Dolos analysis per test-split
problem (Dolos compares files within one "assignment"), writing each
problem's original submissions plus every level's `mutated_code` out as
throwaway files, and reads the pairwise `similarity` column out of its
CSV report for exactly the negative/positive pairs the rest of this
project's eval protocol already uses (same pool as `final_test_eval.py`).

```bash
./.venv/bin/python -m training.eval.dolos_baseline \
  --dataset-dir /path/to/dataset \
  --manifest training/data/artifacts/manifest_v1.jsonl \
  --node-bin /path/to/nvm/versions/node/v22.x.x/bin/node
```

| Level | Dolos AUROC |
|---|---|
| L1 | 0.9998 |
| L2 | 0.9999 |
| L3 | 0.9973 |
| L4 | 0.9845 |
| L5 | 0.9961 |
| L6 | 0.9747 |
| **mean L4-L6** | **0.9851** |

Full numbers in `training/eval/artifacts/dolos_baseline_v1.json`.
**Dolos alone already beats both the fine-tuned bi-encoder alone (0.9764)
and the literature-quoted reference number (0.864)** -- a clean
winnowing/token-based matcher is a genuinely strong baseline on this
dataset's L1-L6 mutation levels, most of which preserve a lot of
token-level structure even where they change identifiers or control
flow.

### Feature computation

`training/scorer/build_features.py` computes, for every negative and
L1-L6 positive pair in train/dev/test:

- `csim_ted`: reused directly from `csim_score` in
  `csim_baseline_pairs_v1.csv` for negatives (already computed in Fase
  0); computed fresh via `csim.utils.preprocess_code` +
  `get_similarity_coefficient` (`apted`/`python_3_13`, same as Fase 0)
  for positives, parallelized across all 24 cores (`ProcessPoolExecutor`)
  since each pair is ~0.1-0.2s single-threaded -- makes computing TED for
  all ~56,600 L1-L6 positive pairs (train+dev+test) a matter of minutes,
  not hours, so no subsampling was needed even for train.
- `biencoder_cosine`: the fine-tuned `best_checkpoint`'s cosine
  similarity, same as `final_test_eval.py`.

**Bug caught during this step**: `csim.utils.preprocess_code` does *not*
raise on a syntax error -- its ANTLR grammar prints `"Syntax error ..."`
to stderr and returns a degenerate near-empty tree instead. A small
fraction of L1-L6 mutated code (mostly reindentation-related, tabs vs.
spaces) fails to parse this way; naively wrapping the call in
`try/except Exception` (the pattern Fase 0's `csim_baseline.py` also
uses) does not catch this and would silently produce a garbage TED score
from comparing two near-empty trees. Fixed by capturing stderr during
the call and treating any output as a failed parse. Affected rows are
dropped: 71/41,244 train positives (0.17%), 91/8,150 test positives
(1.1%), 0/7,280 dev positives.

```bash
./.venv/bin/python -m training.scorer.build_features \
  --dataset-dir /path/to/dataset \
  --manifest training/data/artifacts/manifest_v1.jsonl \
  --checkpoint training/artifacts/best_checkpoint
```

### Fusion scorer and Decision 2

`training/scorer/train_fusion.py` fits an
`sklearn.ensemble.HistGradientBoostingClassifier` (default
hyperparameters -- two features, ~265k rows, low overfitting risk) on
the two features, on the **train** split. `training/scorer/eval_fusion.py`
scores the **test** split and tabulates everything together:

```bash
./.venv/bin/python -m training.scorer.train_fusion
./.venv/bin/python -m training.scorer.eval_fusion
```

| Model | mean L4-L6 AUROC (test) |
|---|---|
| **Fusion (bi-encoder + csim TED, GBDT)** | **0.9892** |
| Dolos | 0.9851 |
| Bi-encoder fine-tuned (alone) | 0.9764 |
| csim TED (alone) | 0.9140 |
| Zero-shot bi-encoder | 0.8583 |

Full per-level numbers (including AUPRC, FPR@recall95) in
`training/scorer/artifacts/eval_fusion_v1.json`.

**Decision 2: the fusion scorer beats Dolos, but by a real margin, not a
comfortable one.** +0.0041 mean L4-L6 AUROC (0.9892 vs 0.9851) -- nowhere
near the +0.03 margin noted as a target in earlier planning (that number
came from a prior conversation, not any file in this repo, so it's
unverifiable as a hard requirement rather than a rough goalpost). Two
things support treating this as a real win rather than noise: (1) the
**fusion score beats both of its own input features alone at every
level** (dominated by `biencoder_cosine` on L4/L5, by neither on L6,
never worse than the better of the two -- exactly what a working GBDT
fusion should do), and (2) it clearly **reduces FPR@recall95 vs. csim
TED alone** on L4 (0.0003 vs 0.538) and L5 (0.002 vs 0.011), the second
criterion noted for this decision. The bi-encoder *alone* actually loses
to Dolos (0.9764 vs 0.9851) -- the neural path only earns its complexity
here in combination with a structural signal, not standalone. Per-level,
Dolos still wins on L6 specifically (0.975 vs fusion's 0.968); fusion's
aggregate win comes from dominating L4/L5, not from winning everywhere.

**Decision: keep the hybrid (bi-encoder + csim TED + GBDT) as Fase 4's
answer**, on the strength of (1) and (2) above, despite the thin margin
over Dolos alone.

Next: Fase 5 (packaging -- `src/csim_ai` inference path).

## Fase 5: packaging (inference path)

`src/csim_ai` is now a real package: export the fine-tuned bi-encoder to
ONNX, wrap it plus the csim TED signal and the Fase 4 fusion model behind
a small Python API and a CLI. Base install stays light (no torch, per
the plan since Fase 0/2) -- `onnxruntime` + `tokenizers` + `numpy` only.

### ONNX export

```bash
./.venv/bin/pip install -e ".[export]"   # onnx, dev-only, needed to export
./.venv/bin/python -m training.export_onnx --verify
```

Exports the base transformer only (`last_hidden_state`); pooling +
L2-normalize happen in numpy at inference time (`src/csim_ai/_onnx_encoder.py`),
matching `mean_pool()` in `train_biencoder.py` exactly. `--verify` checks
the exported graph against the live PyTorch model on a couple of sample
inputs.

**Exporter note**: `torch.onnx.export(..., dynamo=True)` (the default in
torch >=2.9) needs the `onnxscript` package and isn't otherwise
necessary here -- used the legacy TorchScript exporter (`dynamo=False`)
instead. It emits tracer warnings about boolean conversions in HF's
attention-masking code; confirmed harmless for this encoder-only,
padding-mask use case: max abs diff 3.35e-7, cosine 1.0 against PyTorch,
on both a single input and a padded batch.

**Quantization: tried, rejected.** The plan was int8 dynamic
quantization for lighter/faster CPU inference (judges running this at
scale). Measured instead:

| Precision | Cosine vs. fp32 | CPU latency (batch=8, 512 tok) | Size |
|---|---|---|---|
| fp32 (shipped) | 1.0 | 350ms | 502MB |
| fp16 | ~0.99999 | **2530ms (7x slower)** | 251MB |
| int8 dynamic | **~0.35-0.55 (broken)** | 128ms | 126MB |

int8 dynamic quantization (with and without the recommended
`quant_pre_process` shape-inference pass) changes the embedding
*direction*, not just its precision -- disqualifying, since cosine
similarity is exactly the training objective and the only thing the
downstream scorer reads. fp16 preserves quality but `onnxruntime`'s CPU
execution provider has no efficient native fp16 kernels, so it's slower
than fp32, not faster -- fp16 only pays off on GPU. **Shipping fp32
only**; revisiting quantization later (static/calibrated int8, more
engineering effort) only if CPU inference speed becomes a real
bottleneck in practice.

### Model hosting

The trained weights (bi-encoder ONNX export + the Fase 4 fusion model)
are hosted on Hugging Face Hub, **not bundled in the package**:
[edson-eddy/csim-ai](https://huggingface.co/edson-eddy/csim-ai) (public
model repo, `model.onnx` + tokenizer files + `fusion_model.joblib`).
`huggingface_hub` is a base dependency (no torch -- just
requests/filelock/tqdm-scale deps), so `Scorer()` with no arguments
downloads and caches the bi-encoder on first use with zero setup; a
local `model_path`/`fusion_model_path` still works and skips the
network entirely. This was a deliberate reversal of Fase 5's original
"local path only" decision, made once actually publishing to PyPI raised
the question -- **retraining isn't a reasonable `setup` step** (needs
the external dataset, a GPU, ~2.3h, per Fase 3), so shipping usable
pre-trained weights means hosting them somewhere; Hugging Face Hub was
chosen over GitHub Releases for the official `huggingface_hub` Python
client and because the fine-tune's own base model
(`microsoft/unixcoder-base`) already lives there.

Verified end-to-end against real dataset pairs: an unrelated same-problem
pair scored `fusion=0.00007`, a true L4-mutation positive pair scored
`fusion=0.978` -- both directions correct.

`tests/test_inference.py` and `tests/test_hub_download.py` (pytest,
`pip install -e ".[dev]"`) cover bi-encoder-only, full-fusion, and
Hugging Face Hub auto-download scoring; the first two skip gracefully
when the gitignored local artifacts (`onnx_model/`,
`fusion_model_v1.joblib`) aren't present, since a fresh clone doesn't
have them.

CLI/API usage details (`csim-ai setup/report/group/info`, the `Scorer`
class) live in the main [README.md](../README.md), not duplicated here.

## Fase 6: report

See [REPORT.md](REPORT.md) for the full project narrative -- problem
statement, methodology, both decisions, results, and limitations, pulled
together from the phase-by-phase log above.
