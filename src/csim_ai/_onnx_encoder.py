"""ONNX bi-encoder wrapper -- onnxruntime + tokenizers only, no
torch/transformers at runtime. Mean-pool + L2-normalize matches
`mean_pool()` in `training/train_biencoder.py` exactly, since that's
what the exported ONNX graph was trained against.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MAX_LENGTH = 512


def _mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(np.float32)
    summed = (last_hidden * mask).sum(1)
    counts = mask.sum(1).clip(min=1e-9)
    pooled = summed / counts
    return pooled / np.linalg.norm(pooled, axis=-1, keepdims=True)


class OnnxEncoder:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"))
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
