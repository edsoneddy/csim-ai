"""Hugging Face Hub download for the pre-trained bi-encoder + fusion
model -- lets `Scorer()`/`csim-ai setup --download` work out of the box
without a local checkpoint or the `[export]` extra (torch/transformers).

`huggingface_hub` is a base dependency: no torch, just small utility
deps (requests/filelock/tqdm/...), so it doesn't break the "no torch in
the base install" promise from Fase 5.
"""
from __future__ import annotations

DEFAULT_REPO_ID = "edson-eddy/csim-ai"


def download_model(repo_id: str = DEFAULT_REPO_ID, revision: str | None = None) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=["model.onnx", "tokenizer*.json"])


def download_fusion_model(repo_id: str = DEFAULT_REPO_ID, revision: str | None = None) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename="fusion_model.joblib", revision=revision)


def cached_fusion_model(repo_id: str = DEFAULT_REPO_ID, revision: str | None = None) -> str | None:
    """Like download_fusion_model, but never hits the network -- returns
    None if it isn't already cached locally (e.g. from a prior `csim-ai
    setup` or `use_fusion=True` call), instead of downloading it."""
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id=repo_id, filename="fusion_model.joblib", revision=revision, local_files_only=True)
    except Exception:
        return None
