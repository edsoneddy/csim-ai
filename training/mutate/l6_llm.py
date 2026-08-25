#!/usr/bin/env python3
"""L6: free-form semantic rewriting via a local LLM (Qwen2.5-Coder,
served by Ollama).

Unlike L1-L5, this has no static safety argument -- an LLM can silently
change behavior, and per section 5 of the project brief we don't execute
code to check output. So L6 is deliberately treated as lowest-confidence:

1. Run on a *subset*, not the whole dataset -- LLM inference is slow, and
   this is explicitly the level the brief expects the least trust from.
2. Every row is tagged `validated_by_execution: false`, plus a cheap,
   non-blocking heuristic (`heuristic_io_counts_match`: does the rewrite
   have the same number of input()/print() calls as the original) --
   informational only, not a filter, since a legitimate rewrite could
   merge print calls and still be correct.
3. A small sample is written alongside the full output
   (`l6_manual_review_sample.jsonl`, original + rewritten side by side)
   for a human to actually read before this tier is trusted at all --
   the "muestra revisada a mano" the Fase 1 exit criterion asks for.

Requires Ollama running locally with the model pulled:
    ollama pull qwen2.5-coder:7b
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import urllib.request
from pathlib import Path

from . import validate

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:7b"
TEMPERATURE = 0.4

PROMPT_TEMPLATE = """You are rewriting Python source code so it looks different while preserving IDENTICAL behavior: same stdin reads, same stdout output, same results for every input, same edge cases and errors.

Rules:
- You MAY: rename variables/functions, restructure control flow (loops, conditionals), reorganize code, use different but behaviorally equivalent idioms and data structures, change comments/formatting.
- You MUST NOT: change the algorithm's logic, change what is read from input or printed to output, add or remove functionality, add explanations or comments about what you changed.
- Output ONLY the rewritten Python code. No markdown code fences, no explanation, no preamble.

Original code:
{code}

Rewritten code:"""


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:python)?\s*\n(.*)\n```\s*$", text, re.DOTALL)
    return m.group(1) if m else text


def rewrite(code: str, timeout: int = 180) -> str | None:
    payload = json.dumps(
        {
            "model": MODEL,
            "prompt": PROMPT_TEMPLATE.format(code=code),
            "stream": False,
            "options": {"temperature": TEMPERATURE},
        }
    ).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return _strip_markdown_fence(data["response"])
    except Exception as e:
        print(f"  [ollama error: {e}]")
        return None


def _io_counts(code: str) -> tuple[int, int] | None:
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    n_input = n_print = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "input":
                n_input += 1
            elif node.func.id == "print":
                n_print += 1
    return n_input, n_print


def load_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_splits(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {pid: info["split"] for pid, info in data["problems"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=os.environ.get("CSIM_AI_DATASET_DIR"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent.parent / "data" / "artifacts")
    args = parser.parse_args()

    if not args.dataset_dir:
        parser.error("--dataset-dir or CSIM_AI_DATASET_DIR is required")
    dataset_dir = Path(args.dataset_dir).resolve()

    manifest = load_manifest(args.manifest)
    split_of = load_splits(args.splits)
    rng = random.Random(args.seed)
    sample = rng.sample(manifest, min(args.n_samples, len(manifest)))

    out_path = args.out_dir / "synthetic_l6_v1.jsonl"
    review_path = args.out_dir / "l6_manual_review_sample.jsonl"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    review_sample_size = 30
    review_indices = set(rng.sample(range(len(sample)), min(review_sample_size, len(sample))))

    n_attempt = n_kept = n_parse_failed = n_unreachable = n_io_mismatch = 0

    with out_path.open("w", encoding="utf-8") as out, review_path.open("w", encoding="utf-8") as rev:
        for i, row in enumerate(sample):
            code = (dataset_dir / row["path"]).read_text(encoding="utf-8", errors="ignore")
            n_attempt += 1
            print(f"[{n_attempt}/{len(sample)}] {row['path']}")
            rewritten = rewrite(code)
            if rewritten is None:
                n_unreachable += 1
                continue
            if not rewritten.strip() or not validate.parses(rewritten):
                n_parse_failed += 1
                continue

            io_matches = _io_counts(code) == _io_counts(rewritten)
            record = {
                "problem_id": row["problem_id"],
                "submission_id": row["submission_id"],
                "split": split_of.get(row["problem_id"], "unknown"),
                "level": "L6",
                "rule": "llm_rewrite",
                "mutated_code": rewritten,
                "validated_by_execution": False,
                "heuristic_io_counts_match": io_matches,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_kept += 1
            if not io_matches:
                n_io_mismatch += 1
            if i in review_indices:
                rev.write(json.dumps({"original_code": code, **record}, ensure_ascii=False) + "\n")

    report = {
        "n_attempted": n_attempt,
        "n_kept": n_kept,
        "n_parse_failed": n_parse_failed,
        "n_unreachable": n_unreachable,
        "n_io_count_mismatch": n_io_mismatch,
    }
    print(json.dumps(report, indent=2))
    print(f"\nout: {out_path}")
    print(f"review sample: {review_path}")


if __name__ == "__main__":
    main()
