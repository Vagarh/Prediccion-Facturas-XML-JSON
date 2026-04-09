# -*- coding: utf-8 -*-
"""Pipeline de enriquecimiento de features con embeddings semánticos.

Extiende el vector de 48 features numéricas del modelo XGBoost con
representaciones vectoriales densas de los campos textuales clínicos
(diagnósticos CIE-10, códigos CUPS, observaciones), respetando la
división temporal obligatoria para prevenir data leakage.

Cumplimiento:
    - Toda inferencia es local (sin llamadas a APIs externas).
    - No se usa ``print()`` — se usa ``logging``.
    - Ley 1581 de Colombia: los datos clínicos nunca salen de la infraestructura.
    - División temporal siempre por ``fecha_radicacion`` (nunca aleatoria).
    - ``RANDOM_STATE = 42`` en cualquier operación con aleatoriedad.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import spacy

from src.config import EMBEDDING_DIM, RANDOM_STATE
from src.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)

# Campos textuales clínicos que se enriquecen con embeddings
TEXT_FIELD_KEYS: list[str] = ["diagnostico_texto", "cups_texto", "observaciones"]


def build_text_fields(row: dict) -> dict[str, str]:
    """Extrae los campos textuales clínicos de un registro de factura.

    Recupera ``diagnostico_texto``, ``cups_texto`` y ``observaciones`` del
    diccionario de entrada. Los valores ausentes o ``None`` se sustituyen
    por cadena vacía ``""`` para garantizar que el pipeline no se interrumpa.

    Args:
        row: Diccionario con los datos de un registro de factura. Puede
            contener cualquier subconjunto de los campos textuales.

    Returns:
        Diccionario con exactamente tres claves:
        ``{"diagnostico_texto": str, "cups_texto": str, "observaciones": str}``.
        Los valores ausentes o ``None`` se reemplazan por ``""``.

    Example:
        >>> row = {"diagnostico_texto": "hipertensión", "valor_factura": 150000}
        >>> fields = build_text_fields(row)
        >>> fields["diagnostico_texto"]
        'hipertensión'
        >>> fields["cups_texto"]
        ''
        >>> fields["observaciones"]
        ''
    """
    result: dict[str, str] = {}
    for key in TEXT_FIELD_KEYS:
        value = row.get(key)
        result[key] = value if isinstance(value, str) and value else ""
    logger.debug(
        "build_text_fields: extraídos %d campos textuales (vacíos: %d).",
        len(result),
        sum(1 for v in result.values() if not v),
    )
    return result


def enrich_features(
    numeric_features: np.ndarray,
    text_fields: dict[str, str],
    embedding_service: EmbeddingService,
    nlp_pipeline: spacy.Language,
) -> np.ndarray:
    """Concatena embeddings semánticos al vector numérico de 48 features.

    Para cada campo textual en ``text_fields`` (en el orden definido por
    ``TEXT_FIELD_KEYS``), genera un embedding de dimensión ``EMBEDDING_DIM``
    y lo concatena al vector numérico. Los campos vacíos producen un vector
    de ceros de la misma dimensión (Requisito 2.4).

    Args:
        numeric_features: Vector NumPy de forma ``(48,)`` con las features
            numéricas del modelo XGBoost base.
        text_fields: Diccionario con los campos textuales clínicos. Se esperan
            las claves ``diagnostico_texto``, ``cups_texto`` y ``observaciones``.
            Producido típicamente por ``build_text_fields``.
        embedding_service: Instancia de ``EmbeddingService`` para codificar
            los campos textuales en vectores de dimensión ``EMBEDDING_DIM``.
        nlp_pipeline: Pipeline spaCy en español. Actualmente reservado para
            preprocesamiento adicional; el preprocesamiento principal se
            delega a ``EmbeddingService.encode`` vía ``preprocess_text``.

    Returns:
        Vector NumPy de forma ``(48 + EMBEDDING_DIM * n_text_fields,)`` donde
        ``n_text_fields`` es el número de claves en ``TEXT_FIELD_KEYS`` (3).
        El dtype es ``float32``.

    Raises:
        ValueError: Si ``numeric_features`` no tiene exactamente 48 elementos.

    Example:
        >>> import numpy as np
        >>> numeric = np.zeros(48, dtype=np.float32)
        >>> fields = {"diagnostico_texto": "", "cups_texto": "", "observaciones": ""}
        >>> # Con EmbeddingService y nlp_pipeline reales:
        >>> # enriched = enrich_features(numeric, fields, service, nlp)
        >>> # enriched.shape == (48 + 384 * 3,)
        True
    """
    if numeric_features.shape[0] != 48:
        raise ValueError(
            f"Se esperaban 48 features numéricas, se recibieron {numeric_features.shape[0]}."
        )

    embeddings: list[np.ndarray] = []
    for key in TEXT_FIELD_KEYS:
        text = text_fields.get(key, "")
        embedding = embedding_service.encode(text)
        embeddings.append(embedding)
        logger.debug(
            "enrich_features: campo '%s' → embedding shape %s.", key, embedding.shape
        )

    enriched = np.concatenate(
        [numeric_features.astype(np.float32)] + embeddings, axis=0
    )
    logger.debug(
        "enrich_features: vector enriquecido shape %s (48 + %d * %d).",
        enriched.shape,
        EMBEDDING_DIM,
        len(TEXT_FIELD_KEYS),
    )
    return enriched


def generate_split_embeddings(
    df: pd.DataFrame,
    split_col: str,
    embedding_service: EmbeddingService,
) -> np.ndarray:
    """Genera embeddings para un split respetando la división temporal.

    Ordena el DataFrame por ``fecha_radicacion`` antes de generar los
    embeddings, garantizando que el orden temporal se preserve y que no
    haya data leakage entre splits (Requisito 2.5, 8.3).

    El campo ``split_col`` indica qué columna textual se codifica
    (p. ej. ``"diagnostico_texto"``). Para generar embeddings de múltiples
    campos, llamar a esta función por separado para cada campo y concatenar.

    Args:
        df: DataFrame con al menos las columnas ``fecha_radicacion`` y
            ``split_col``. No se modifica el DataFrame original.
        split_col: Nombre de la columna textual a codificar.
        embedding_service: Instancia de ``EmbeddingService`` para codificar
            los textos en vectores de dimensión ``EMBEDDING_DIM``.

    Returns:
        Array NumPy de forma ``(len(df), EMBEDDING_DIM)`` con los embeddings
        generados en el orden temporal de ``fecha_radicacion``. El dtype es
        ``float32``.

    Raises:
        KeyError: Si ``fecha_radicacion`` o ``split_col`` no existen en ``df``.

    Example:
        >>> import pandas as pd, numpy as np
        >>> df = pd.DataFrame({
        ...     "fecha_radicacion": pd.to_datetime(["2025-06-01", "2025-06-02"]),
        ...     "diagnostico_texto": ["hipertensión", "diabetes"],
        ... })
        >>> # Con EmbeddingService real:
        >>> # embs = generate_split_embeddings(df, "diagnostico_texto", service)
        >>> # embs.shape == (2, 384)
        True
    """
    if "fecha_radicacion" not in df.columns:
        raise KeyError("El DataFrame debe contener la columna 'fecha_radicacion'.")
    if split_col not in df.columns:
        raise KeyError(f"El DataFrame no contiene la columna '{split_col}'.")

    logger.info(
        "generate_split_embeddings: ordenando %d registros por 'fecha_radicacion' (sin shuffle).",
        len(df),
    )

    # Ordenar por fecha_radicacion — división temporal obligatoria, nunca aleatoria
    df_sorted = df.sort_values("fecha_radicacion", kind="stable").reset_index(drop=True)

    texts: list[str] = df_sorted[split_col].fillna("").tolist()

    logger.info(
        "generate_split_embeddings: codificando %d textos del campo '%s'.",
        len(texts),
        split_col,
    )

    embeddings = embedding_service.encode_batch(texts)

    logger.info(
        "generate_split_embeddings: embeddings generados con shape %s.",
        embeddings.shape,
    )
    return embeddings


def save_embeddings(embeddings: np.ndarray, path: Path | str, key: str = "embeddings") -> None:
    """Serializa embeddings en formato NPZ comprimido.

    Guarda el array de embeddings usando ``np.savez_compressed`` para
    reducir el espacio en disco (Requisito 8.4).

    Args:
        embeddings: Array NumPy de embeddings a serializar.
        path: Ruta del archivo de salida (con o sin extensión ``.npz``).
        key: Nombre de la clave dentro del archivo NPZ. Por defecto ``"embeddings"``.

    Example:
        >>> import numpy as np, tempfile, pathlib
        >>> arr = np.zeros((10, 384), dtype=np.float32)
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     p = pathlib.Path(tmp) / "test_embs"
        ...     save_embeddings(arr, p)
        ...     loaded = np.load(str(p) + ".npz")
        ...     loaded["embeddings"].shape == (10, 384)
        True
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **{key: embeddings})
    logger.info(
        "save_embeddings: embeddings de shape %s guardados en '%s.npz'.",
        embeddings.shape,
        path,
    )


def load_embeddings(path: Path | str, key: str = "embeddings") -> np.ndarray:
    """Carga embeddings desde un archivo NPZ comprimido.

    Args:
        path: Ruta del archivo NPZ (con o sin extensión ``.npz``).
        key: Nombre de la clave dentro del archivo NPZ. Por defecto ``"embeddings"``.

    Returns:
        Array NumPy con los embeddings cargados.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        KeyError: Si la clave ``key`` no existe en el archivo NPZ.

    Example:
        >>> import numpy as np, tempfile, pathlib
        >>> arr = np.zeros((5, 384), dtype=np.float32)
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     p = pathlib.Path(tmp) / "embs"
        ...     save_embeddings(arr, p)
        ...     loaded = load_embeddings(p)
        ...     loaded.shape == (5, 384)
        True
    """
    path = Path(path)
    npz_path = path if path.suffix == ".npz" else Path(str(path) + ".npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"Archivo NPZ no encontrado: {npz_path}")

    data = np.load(str(npz_path))
    if key not in data:
        raise KeyError(f"Clave '{key}' no encontrada en {npz_path}. Claves disponibles: {list(data.keys())}")

    embeddings = data[key]
    logger.info(
        "load_embeddings: embeddings de shape %s cargados desde '%s'.",
        embeddings.shape,
        npz_path,
    )
    return embeddings
