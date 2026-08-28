"""ONNX bi-encoder wrapper -- onnxruntime + tokenizers only, no
torch/transformers at runtime. Mean-pool + L2-normalize matches
`mean_pool()` in `training/train_biencoder.py` exactly, since that's
what the exported ONNX graph was trained against.

GPU is opt-in, not automatic: the base `onnxruntime` package (a Fase 5
dependency) is CPU-only. Using CUDA means swapping it out for
`onnxruntime-gpu` yourself (`pip uninstall onnxruntime && pip install
onnxruntime-gpu`) -- the two packages occupy the same import namespace
and can't both be installed, which is why this isn't a normal pip
extra -- plus having the CUDA/cuDNN runtime libraries available (a
system CUDA toolkit, or the `nvidia-cublas-cuXX`/`nvidia-cudnn-cuXX`
pip packages providing them, e.g. already pulled in by `torch`'s CUDA
build if that happens to be installed too). Requesting `device="cuda"`
when none of that is in place doesn't error -- onnxruntime silently
falls back to CPU -- so this checks `session.get_providers()` after the
fact and warns if the requested provider didn't actually take, instead
of leaving that silent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MAX_LENGTH = 512

_PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "auto": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}


def _mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(np.float32)
    summed = (last_hidden * mask).sum(1)
    counts = mask.sum(1).clip(min=1e-9)
    pooled = summed / counts
    return pooled / np.linalg.norm(pooled, axis=-1, keepdims=True)


class OnnxEncoder:
    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        if device not in _PROVIDERS:
            raise ValueError(f"device must be one of {sorted(_PROVIDERS)}, got {device!r}")
        model_dir = Path(model_dir)
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=_PROVIDERS[device])
        if device == "cuda" and "CUDAExecutionProvider" not in self.session.get_providers():
            print(
                "csim_ai: device='cuda' requested but CUDAExecutionProvider isn't active "
                f"(using {self.session.get_providers()} instead) -- install onnxruntime-gpu "
                "and its CUDA/cuDNN runtime libraries to actually use the GPU.",
                file=sys.stderr,
            )
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=MAX_LENGTH)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        # Chunked so a large batch of wildly-different-length files
        # doesn't all get padded to the single longest one at once.
        chunks = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encodings = self.tokenizer.encode_batch(batch)
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            last_hidden = self.session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]
            chunks.append(_mean_pool(last_hidden, attention_mask))
        return np.concatenate(chunks, axis=0)

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        embs = self.encode([text_a, text_b])
        return float(np.dot(embs[0], embs[1]))
