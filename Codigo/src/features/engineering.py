# -*- coding: utf-8 -*-
"""Validación y preparación de features para inferencia.

Este módulo valida que los diccionarios de features recibidos por la API
contengan las 48 features esperadas por el modelo XGBoost en el orden correcto.

Nota: La API acepta features pre-computadas. El pipeline completo de
transformación (log1p, flags, interacciones) se ejecuta antes de llegar aquí.
Los parsers de JSON RIPS y XML FEV se implementarán en la Fase 1D.
"""

import functools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_LIST_PATH

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def load_feature_names(path: Path | None = None) -> list[str]:
    """Carga la lista ordenada de 48 features desde archivo.

    Args:
        path: Ruta al archivo de features. Si es None, usa la ruta por defecto.

    Returns:
        Lista de 48 nombres de features en el orden esperado por el modelo.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo no contiene exactamente 48 features.
    """
    feature_path = path or FEATURE_LIST_PATH
    logger.info("Cargando lista de features desde %s", feature_path)

    text = feature_path.read_text(encoding="utf-8").strip()
    features = [line.strip() for line in text.splitlines() if line.strip()]

    if len(features) != 48:
        raise ValueError(
            f"Se esperaban 48 features, se encontraron {len(features)} en {feature_path}"
        )

    logger.info("Features cargadas: %d", len(features))
    return features


def validate_features(
    features_dict: dict[str, float],
    feature_names: list[str],
) -> dict[str, float]:
    """Valida que el diccionario contenga las 48 features requeridas.

    Args:
        features_dict: Diccionario {nombre_feature: valor} recibido del cliente.
        feature_names: Lista ordenada de features esperadas.

    Returns:
        Diccionario validado con las 48 features.

    Raises:
        ValueError: Si faltan features o los valores no son numéricos.
    """
    expected = set(feature_names)
    received = set(features_dict.keys())

    missing = expected - received
    if missing:
        raise ValueError(
            f"Faltan {len(missing)} features requeridas: {sorted(missing)}"
        )

    # Validar que todos los valores sean numéricos
    invalid = []
    for name in feature_names:
        val = features_dict[name]
        if not isinstance(val, (int, float)):
            invalid.append(name)
        elif isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            invalid.append(name)

    if invalid:
        raise ValueError(
            f"Features con valores no numéricos o inválidos (NaN/Inf): {invalid}"
        )

    return {name: float(features_dict[name]) for name in feature_names}


def prepare_input(
    features_dict: dict[str, float],
    feature_names: list[str],
) -> pd.DataFrame:
    """Crea un DataFrame de una fila con las features en el orden del modelo.

    Args:
        features_dict: Diccionario validado de features.
        feature_names: Lista ordenada de features esperadas.

    Returns:
        DataFrame de forma (1, 48) con columnas en el orden correcto.
    """
    validated = validate_features(features_dict, feature_names)
    return pd.DataFrame([validated], columns=feature_names)


def prepare_batch_input(
    features_list: list[dict[str, float]],
    feature_names: list[str],
) -> pd.DataFrame:
    """Crea un DataFrame de N filas con las features en el orden del modelo.

    Args:
        features_list: Lista de diccionarios de features (uno por factura).
        feature_names: Lista ordenada de features esperadas.

    Returns:
        DataFrame de forma (N, 48) con columnas en el orden correcto.

    Raises:
        ValueError: Si algún diccionario de features es inválido.
    """
    validated_rows = []
    for i, feat_dict in enumerate(features_list):
        try:
            validated = validate_features(feat_dict, feature_names)
            validated_rows.append(validated)
        except ValueError as e:
            raise ValueError(f"Error en factura índice {i}: {e}") from e

    return pd.DataFrame(validated_rows, columns=feature_names)
