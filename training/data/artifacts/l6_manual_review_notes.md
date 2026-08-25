# L6 manual review notes

30 pairs from `l6_manual_review_sample.jsonl` (seeded sample of the 400
L6 rewrites), read line-by-line by hand -- no execution, per the
project's constraint that mutations are validated by reading/parsing,
never by running the code.

## Result

**30/30 read as behaviorally correct.** Rewrites ranged from pure
identifier renaming (the majority) to real restructuring: moving a
print from inside a function to the caller (pair 16), replacing a
counter-in-a-loop with an equivalent boolean-toggle expression (pair
22), replacing manual indexing with slicing + enumerate (pairs 4, 8),
inlining a confusingly-shadowed nested function into a single
expression (pair 12). In every case the renamed/restructured version
was checked against the original statement-by-statement for the same
control flow, the same I/O calls in the same order, and the same
arithmetic.

**One caveat (pair 12, problem 1588):** the model replaced `for j in
range(nn): item = y[j]` with `for item in y:`. These are only
equivalent when `len(y) == nn` -- true for well-formed judge input
(which is what `nn` is supposed to describe), but not identical in
general if the input were malformed. Judged as an acceptable, narrow
risk given the data source, not a rejection.

## `heuristic_io_counts_match` false positives

2 of the 30 sampled pairs (4, 8; same pattern, different submissions of
problem 1123) have `heuristic_io_counts_match: false`, both confirmed
correct on reading. Cause: the original reassigns `input =
sys.stdin.read` and calls it; the model renamed that variable to
`input_data`. The heuristic only counts calls to a function literally
named `input`, so it miscounts this as "input() call removed" when
behavior is unchanged. This confirms the heuristic's role as documented
in `l6_llm.py`: informational, not a filter.

## Methodology note

This is a manual read, not a proof -- no test cases were run against
either version. It's the strongest check available under the project's
"never execute to validate" constraint (Fase 0), and it's what the Fase
1 roadmap asks for specifically at L6. 30 pairs out of 400 generated is
a ~7.5% audit sample; treat the whole L6 batch as "spot-checked and
consistent," not "individually verified."
