"""ONNX export of a fine-tuned bi-encoder checkpoint -- shared by
`training/export_onnx.py` (dev-side, points at
training/artifacts/best_checkpoint by default) and `csim-ai setup
--export-from` (works from any installed checkpoint, no access to this
repo's training/ tree needed).

Requires the `export` extra (`torch`, `transformers`, `onnx`) -- not a
base-install dependency, imported lazily here.

See docs/DEVELOPMENT.md, Fase 5, for why: legacy TorchScript exporter (not the
dynamo-based default in torch >=2.9, which needs `onnxscript`), fp32
only (int8 dynamic quantization measurably breaks embedding direction;
fp16 is ~7x slower than fp32 on CPU).
"""
from __future__ import annotations

import shutil
from pathlib import Path


def mean_pool(last_hidden, attention_mask):
    import numpy as np

    mask = attention_mask[..., None].astype(np.float32)
    summed = (last_hidden * mask).sum(1)
    counts = mask.sum(1).clip(min=1e-9)
    pooled = summed / counts
    return pooled / np.linalg.norm(pooled, axis=-1, keepdims=True)


def verify(onnx_path: Path, checkpoint: Path) -> float:
    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import AutoModel, AutoTokenizer

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

    return max_diff


def export(checkpoint: Path, out_dir: Path, opset: int = 17, verify_after: bool = False) -> Path:
    import torch
    from transformers import AutoModel, AutoTokenizer

    class _LastHiddenStateOnly(torch.nn.Module):
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    checkpoint = Path(checkpoint)
    out_dir = Path(out_dir)

    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).eval()
    wrapped = _LastHiddenStateOnly(model)

    sample = tok(["def f(x):\n    return x\n"], return_tensors="pt")
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

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
        opset_version=opset,
        dynamo=False,
    )
    print(f"exported: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(checkpoint / name, out_dir / name)
    print(f"copied tokenizer files to {out_dir}")

    if verify_after:
        max_diff = verify(onnx_path, checkpoint)
        if max_diff > 1e-3:
            raise SystemExit(f"ONNX/PyTorch parity check failed: max abs diff {max_diff:.2e} > 1e-3")
        print("parity OK")

    return onnx_path
