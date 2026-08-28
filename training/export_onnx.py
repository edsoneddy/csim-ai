#!/usr/bin/env python3
"""Fase 5: export the fine-tuned bi-encoder to ONNX for the runtime
inference package (`src/csim_ai`), which uses onnxruntime + tokenizers
only -- no torch/transformers.

Exports just the base transformer (output = last_hidden_state);
mean-pooling and L2-normalize happen in numpy at inference time, same as
`mean_pool()` in `train_biencoder.py`.

Uses the legacy TorchScript-based exporter (`dynamo=False`) -- torch's
new default dynamo-based exporter needs the `onnxscript` package, which
isn't otherwise needed here. The legacy exporter emits tracer warnings
about boolean conversions in HF's attention-masking code; verified
harmless for this encoder-only, padding-mask use case by comparing
exported-model output against the live PyTorch model (`--verify` below):
max abs diff ~3e-7, cosine ~1.0, on both single inputs and padded
batches.

Quantization (int8 dynamic) was tried and rejected: it drops embedding
cosine similarity to ~0.4-0.55 against the fp32 model (see README, Fase
5) -- cosine is exactly what the training objective and downstream
scorer both depend on, so this isn't a minor accuracy hit. fp16 preserves
quality (cosine ~0.99999) but is ~7x *slower* on CPU (no efficient fp16
kernels in onnxruntime's CPU execution provider). Ships fp32 only.

Usage:
    python -m training.export_onnx --verify
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class _LastHiddenStateOnly(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(np.float32)
    summed = (last_hidden * mask).sum(1)
    counts = mask.sum(1).clip(min=1e-9)
    pooled = summed / counts
    return pooled / np.linalg.norm(pooled, axis=-1, keepdims=True)


def verify(onnx_path: Path, checkpoint: Path) -> None:
    import onnxruntime as ort

    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).eval()
    sess = ort.InferenceSession(str(onnx_path))

    samples = [
        ["def add(a, b):\n    return a + b\n"],
        [
            "def add(a, b):\n    return a + b\n",
            "import math\ndef f(x):\n    return math.sqrt(x) + 1\n\nprint(f(4))\n",
        ],
    ]
    max_diff = 0.0
    for batch in samples:
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            torch_out = model(**inputs).last_hidden_state.numpy()
        torch_pooled = mean_pool(torch_out, inputs["attention_mask"].numpy())

        onnx_out = sess.run(
            None, {"input_ids": inputs["input_ids"].numpy(), "attention_mask": inputs["attention_mask"].numpy()}
        )[0]
        onnx_pooled = mean_pool(onnx_out, inputs["attention_mask"].numpy())

        diff = float(np.abs(torch_pooled - onnx_pooled).max())
        cos = float((torch_pooled * onnx_pooled).sum(-1).min())
        max_diff = max(max_diff, diff)
        print(f"  batch size {len(batch)}: max abs diff={diff:.2e}, min cosine={cos:.6f}")

    if max_diff > 1e-3:
        raise SystemExit(f"ONNX/PyTorch parity check failed: max abs diff {max_diff:.2e} > 1e-3")
    print("parity OK")


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

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModel.from_pretrained(args.checkpoint).eval()
    wrapped = _LastHiddenStateOnly(model)

    sample = tok(["def f(x):\n    return x\n"], return_tensors="pt")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out_dir / "model.onnx"

    torch.onnx.export(
        wrapped,
        (sample["input_ids"], sample["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"exported: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(args.checkpoint / name, args.out_dir / name)
    print(f"copied tokenizer files to {args.out_dir}")

    if args.verify:
        verify(onnx_path, args.checkpoint)


if __name__ == "__main__":
    main()
