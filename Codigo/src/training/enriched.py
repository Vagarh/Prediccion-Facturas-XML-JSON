# -*- coding: utf-8 -*-
"""Construcción del vector enriquecido (48 numéricas + 3×384 embeddings).

Usado por ``src.training.model_suite`` para el modelo ``xgb_enriched``. Los
embeddings se cachean por (prefijo, split, campo) en ``data/interim`` para no
recomputarlos entre corridas (MiniLM sobre ~700k textos en CPU tarda horas).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import EMBEDDING_DIM
from src.features.embedding_pipeline import TEXT_FIELD_KEYS

logger = logging.getLogger(__name__)

INTERIM_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "interim"
_CHUNK = 8_192


def _embed_field(texts: list[str], cache: Path) -> np.ndarray:
    """Codifica una lista de textos, con caché NPZ en ``cache``."""
    if cache.exists():
        arr = np.load(cache)["emb"]
        if len(arr) == len(texts):
            logger.info("  cache hit %s (%s)", cache.name, arr.shape)
            return arr
        logger.warning("  cache %s desalineada (%d != %d) — recomputando", cache.name, len(arr), len(texts))

    from src.embeddings.service import EmbeddingService

    svc = EmbeddingService()
    parts: list[np.ndarray] = []
    for i in range(0, len(texts), _CHUNK):
        parts.append(svc.encode_batch(texts[i : i + _CHUNK]))
        if (i // _CHUNK) % 10 == 0:
            logger.info("  %s: %d/%d", cache.stem, min(i + _CHUNK, len(texts)), len(texts))
    emb = np.vstack(parts).astype(np.float32) if parts else np.zeros((0, EMBEDDING_DIM), np.float32)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, emb=emb)
    logger.info("  guardado %s (%s)", cache.name, emb.shape)
    return emb


def build_enriched_xy(
    df: pd.DataFrame, feats: list[str], prefix: str, split: str
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve ``(X_enriquecida, y)`` para un split.

    Args:
        df: Parquet ``*_enriched`` (48 features + columnas textuales).
        feats: Lista de las 48 features numéricas.
        prefix: Prefijo de fecha del dataset (para el nombre de caché).
        split: ``train`` / ``val`` / ``test`` / ``oot``.
    """
    x_num = df[feats].to_numpy(np.float32)
    blocks = [x_num]
    for key in TEXT_FIELD_KEYS:
        texts = df.get(key, pd.Series([""] * len(df))).fillna("").astype(str).tolist()
        cache = INTERIM_DIR / f"{prefix}_{split}_{key}_emb.npz"
        blocks.append(_embed_field(texts, cache))
    x = np.concatenate(blocks, axis=1).astype(np.float32)
    expected = len(feats) + EMBEDDING_DIM * len(TEXT_FIELD_KEYS)
    assert x.shape[1] == expected, f"dim {x.shape[1]} != {expected}"
    return x, df["target"].to_numpy(int)
