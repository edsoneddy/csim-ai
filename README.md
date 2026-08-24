# csim-ai

Neural-augmented Python code plagiarism detection for programming judges.
Successor to [csim](https://github.com/edsoneddy/csim) (ANTLR4 parse-tree
normalization + Tree Edit Distance), adding a contrastively fine-tuned
encoder for the L4-L6 structural/semantic plagiarism cases where pure TED
similarity degrades.

**Task**: plagiarism detection (did B derive from A?), not semantic clone
detection (does B solve the same problem as A?). Two independent correct
solutions to the same problem are a negative, not a positive.

Status: **Fase 1 - synthetic plagiarism generator (L1-L5 done, >=50k
target reached).** No model code yet.

## Layout

```
src/csim_ai/       inference package (empty until Fase 5)
training/
  configs/         YAML configs, one seed per experiment
  data/            audit + split scripts, versioned artifacts
  eval/            baseline harnesses (csim, later Dolos/JPlag), versioned artifacts
  mutate/          L1-L6 synthetic plagiarism generator (Fase 1)
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

**Running total: 56,274 synthetic positives** (55,578 + 696). L6 (LLM
rewrite) is what's left -- it can't get a strong correctness guarantee
without execution, so it'll ship with `validated_by_execution: false` and
a manually-reviewed sample, not blanket trust.
