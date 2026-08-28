# csim-ai

Neural-augmented Python code plagiarism detection for programming judges.
Successor to [csim](https://github.com/edsoneddy/csim) (ANTLR4 parse-tree
normalization + Tree Edit Distance), adding a contrastively fine-tuned
bi-encoder for the structural/semantic plagiarism cases where pure TED
similarity degrades. Scores are a fusion of both signals via a small
GBDT, verified to beat [Dolos](https://dolos.ugent.be/) on this
project's own test data -- see [docs/REPORT.md](docs/REPORT.md) for the
full methodology and results, [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
for the phase-by-phase build log.

**Task**: plagiarism detection (did B derive from A?), not semantic clone
detection (does B solve the same problem as A?). Two independent correct
solutions to the same problem are a negative, not a positive.

## Install

```bash
pip install csim-ai              # bi-encoder cosine similarity only (onnxruntime, no torch)
pip install csim-ai[ast,scorer]  # + csim TED signal + GBDT fusion -- the full hybrid score
```

Model weights aren't bundled in the package (the ONNX export is
~500MB) -- run `csim-ai setup` once after installing to download and
cache them from Hugging Face Hub
([edson-eddy/csim-ai](https://huggingface.co/edson-eddy/csim-ai)).
After that, both the CLI and the `Scorer` class auto-detect the cache
and give the full hybrid score with no further flags or arguments.

```bash
pip install csim-ai[ast,scorer]
csim-ai setup
# bi-encoder cached at: ~/.cache/huggingface/hub/models--edson-eddy--csim-ai/...
# fusion model cached at: .../fusion_model.joblib
```

## CLI

**CLI shape follows csim's**, not a from-scratch design: a single
`csim-ai` command, an action positional, `--path` pointing at a
directory compared exhaustively -- same pattern as `csim
{report,group,tree,view,info} --path DIR --lang ... --talg ...`, since
this tool has the same predecessor and audience.

```bash
csim-ai report --path submissions/
# b.py is similar to a.py with similarity index: 0.9998 (biencoder_cosine=1.0000, csim_ted=1.0000, fusion=0.9998)

csim-ai group --path submissions/ --threshold 0.9
# Group 1 (Average Similarity: 1.00):
# a.py
# c.py
# Unique Files (similarity below threshold):
# b.py

csim-ai info
# which optional backends (onnxruntime, tokenizers, huggingface_hub, csim, scikit-learn, torch) are available
```

### `report`

Pairwise similarity report over every `.py` file in `--path`, all
combinations.

| Flag | Default | Meaning |
|---|---|---|
| `--path`, `-p` | required | Directory of `.py` files to compare exhaustively. |
| `--model-path` | Hugging Face Hub | Directory with `model.onnx`/`tokenizer.json`. Skips the Hub entirely if given. |
| `--fusion-model` | none | Path to a `fusion_model.joblib`. Skips the Hub entirely if given. |
| `--use-fusion` | off | Force-download the fusion model from HF Hub if it isn't already cached and no `--fusion-model` is given. |

### `group`

Same comparison as `report`, but groups files into connected components
by a similarity threshold instead of listing every pair.

Same flags as `report`, plus:

| Flag | Default | Meaning |
|---|---|---|
| `--threshold`, `-t` | required | Similarity threshold (0.0-1.0) for grouping. |

### `info`

No comparison -- just reports which optional backends are importable
(`onnxruntime`/`tokenizers`/`huggingface_hub` from the base install;
`csim`/`scikit-learn` from `[ast,scorer]`; `torch` from `[export]`).
Takes an optional `--model-path` to also check a directory for
`model.onnx`/`tokenizer.json`.

### `setup`

Not part of `pip install .` -- a separate step because the weights
aren't bundled in the package.

| Flag | Default | Meaning |
|---|---|---|
| (none) | -- | Downloads and caches the bi-encoder + fusion model from Hugging Face Hub. |
| `--export-from CHECKPOINT` | none | Export a local torch checkpoint to ONNX instead of downloading (requires `pip install csim-ai[export]`) -- entirely offline, for your own fine-tuned weights rather than this project's. |
| `--out` | `./onnx_model` | Output directory for `--export-from`. |
| `--opset` | `17` | ONNX opset version for `--export-from`. |
| `--no-verify` | off | Skip the PyTorch-vs-ONNX parity check after `--export-from`. |

For **every** `report`/`group` result, `"similarity index"` is the
fusion score when available, else `biencoder_cosine`. `csim_ted`/
`fusion` come back as `None` (and are dropped from the report line) when
`csim`/`scikit-learn` aren't installed, so a bare `pip install csim-ai`
(no extras, no `setup`) still gives a usable bi-encoder-only score.

## Python API

```python
from csim_ai import Scorer

scorer = Scorer()                     # after `csim-ai setup`: full hybrid, cache auto-detected
scorer = Scorer(use_fusion=True)      # force-downloads the fusion model too if `setup` wasn't run yet
scorer = Scorer(
    "path/to/onnx_model",
    fusion_model_path="path/to/fusion_model.joblib",
)                                      # fully local, no network

scorer.score(code_a, code_b)
# {"biencoder_cosine": 0.987, "csim_ted": 0.83, "fusion": 0.978}
```

`Scorer(model_path=None, fusion_model_path=None, use_fusion=False)`:

- `model_path`: directory with `model.onnx`/`tokenizer.json`. `None`
  (default) downloads from Hugging Face Hub, cached after first call.
- `fusion_model_path`: path to a `fusion_model.joblib`. `None` (default)
  auto-uses a fusion model already cached by a prior `csim-ai setup` or
  `use_fusion=True` call, without triggering a network request to check.
- `use_fusion`: if `True` and no `fusion_model_path` is given,
  force-downloads the fusion model from HF Hub instead of just checking
  the cache.

`scorer.score(code_a: str, code_b: str) -> dict` returns
`{"biencoder_cosine": float, "csim_ted": float | None, "fusion": float | None}`
-- `csim_ted`/`fusion` are `None` when `csim`/`scikit-learn` aren't
installed or no fusion model is available.

## Layout

```
src/csim_ai/       inference package -- ONNX bi-encoder + csim TED + GBDT fusion
tests/             pytest smoke tests for src/csim_ai
training/          dataset prep, synthetic plagiarism generation, training, eval, export tooling
docs/
  REPORT.md        project narrative: problem, methodology, results, limitations
  DEVELOPMENT.md   phase-by-phase build log: commands, exact numbers, bugs hit and fixed
```
