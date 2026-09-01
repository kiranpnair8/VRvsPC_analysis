"""Optional deterministic REVE embedding cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from feature_extractors import reve_embed_batch


def _array_digest(X: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(X.shape).encode("utf-8"))
    h.update(str(X.dtype).encode("utf-8"))
    h.update(np.ascontiguousarray(X).view(np.uint8))
    return h.hexdigest()


def cache_path(cache_dir, *, model_dir, ch_names, X, tag):
    cache_dir = Path(cache_dir)
    payload = {
        "tag": tag,
        "model_dir": str(Path(model_dir).resolve()),
        "ch_names": list(ch_names),
        "x_digest": _array_digest(X),
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return cache_dir / f"{tag}_{key}.npz"


@torch.no_grad()
def embed_with_cache(
    model,
    pos_bank,
    X,
    ch_names,
    device,
    batch_size,
    cache_dir,
    model_dir,
    tag,
    use_cache=True,
):
    path = None
    if use_cache:
        path = cache_path(cache_dir, model_dir=model_dir, ch_names=ch_names, X=X, tag=tag)
    if path is not None and path.exists():
        return np.load(path)["Z"]

    zs = []
    for i in range(0, X.shape[0], batch_size):
        xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32, device=device)
        zb = reve_embed_batch(model, pos_bank, xb, ch_names, device)
        zs.append(zb.detach().cpu().numpy())
    Z = np.vstack(zs).astype(np.float32)

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, Z=Z)
    return Z
