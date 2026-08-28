"""`csim-ai score <file_a> <file_b>` -- CLI entry point (see [project.scripts]
in pyproject.toml)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import Scorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Score two Python files for plagiarism similarity.")
    parser.add_argument("file_a", type=Path)
    parser.add_argument("file_b", type=Path)
    parser.add_argument("--model-path", required=True, help="Directory from training/export_onnx.py (model.onnx + tokenizer.json).")
    parser.add_argument("--fusion-model", default=None, help="Path to fusion_model_v1.joblib (requires the [ast,scorer] extras).")
    args = parser.parse_args()

    scorer = Scorer(args.model_path, fusion_model_path=args.fusion_model)
    code_a = args.file_a.read_text(encoding="utf-8", errors="ignore")
    code_b = args.file_b.read_text(encoding="utf-8", errors="ignore")
    print(json.dumps(scorer.score(code_a, code_b), indent=2))


if __name__ == "__main__":
    main()
