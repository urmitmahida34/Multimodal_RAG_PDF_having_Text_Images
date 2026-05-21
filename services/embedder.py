"""
CLIP embedding service.

The model is a module-level singleton — loaded once per process and reused
across every Streamlit rerun, eval script, and ingest script.
"""
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import config

_model:     CLIPModel     | None = None
_processor: CLIPProcessor | None = None


def _load() -> tuple[CLIPModel, CLIPProcessor]:
    global _model, _processor
    if _model is None:
        _model     = CLIPModel.from_pretrained(config.CLIP_MODEL)
        _processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
        _model.eval()
    return _model, _processor


# ── Public API ────────────────────────────────────────────────────────────────

def embed_text(text: str) -> np.ndarray:
    """
    Embed a single text string → 512-dim unit vector.
    CLIP's text encoder has a hard 77-token limit; truncation=True
    ensures longer inputs are cut rather than raising an error.
    """
    model, processor = _load()
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
    return feats.squeeze().cpu().numpy()


def embed_image(pil_image: Image.Image) -> np.ndarray:
    """Embed a PIL image → 512-dim unit vector."""
    model, processor = _load()
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
    return feats.squeeze().cpu().numpy()


def embed_texts_batch(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    True batched text embedding — one CLIP forward pass per batch of 64
    instead of one per chunk. Significantly faster for large PDFs.
    """
    if not texts:
        return np.empty((0, config.EMBEDDING_DIM), dtype=np.float32)

    model, processor = _load()
    all_feats: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        all_feats.append(feats.cpu().numpy())

    return np.vstack(all_feats).astype(np.float32)


def embed_images_batch(images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
    """
    True batched image embedding — one CLIP forward pass per batch of 32.
    Images are larger tensors than text so a smaller batch size is used.
    """
    if not images:
        return np.empty((0, config.EMBEDDING_DIM), dtype=np.float32)

    model, processor = _load()
    all_feats: list[np.ndarray] = []

    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        all_feats.append(feats.cpu().numpy())

    return np.vstack(all_feats).astype(np.float32)
