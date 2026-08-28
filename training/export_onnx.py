#!/usr/bin/env python3
"""Fase 5: export the fine-tuned bi-encoder to ONNX (dev-side wrapper
around `csim_ai._export`, defaulting to this repo's
training/artifacts/best_checkpoint -- see that module's docstring for
the export/quantization rationale). Also reachable from an installed
csim-ai package without this repo via `csim-ai setup --export-from`.

Usage:
    python -m training.export_onnx --verify
"""
from __future__ import annotations

import argparse
from pathlib import Path

from csim_ai._export import export


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(__file__).parent / "artifacts" / "best_checkpoint"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path(__file__).parent / "artifacts" / "onnx_model"
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify", action="store_true", help="Check exported model against PyTorch after export.")
    args = parser.parse_args()

    export(args.checkpoint, args.out_dir, opset=args.opset, verify_after=args.verify)


if __name__ == "__main__":
    main()
