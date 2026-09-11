# -*- coding: utf-8 -*-
"""Carga de modelo y funciones de predicción.

Envuelve las llamadas al modelo con logging y validación apropiada.
Refactorizado desde: notebooks/05_Model_Training.ipynb.

El campeón v3 (LightGBM afinado, ver .claude/04_modelo_metricas.md) se serializa
en ``.joblib`` en vez de XGBoost-JSON nativo; ``load_model`` detecta el formato
por extensión. Ambos exponen la misma interfaz sklearn (``predict``,
``predict_proba``), así que ``predict_single``/``predict_batch`` no cambian.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import MODEL_PATH

logger = logging.getLogger(__name__)


def load_model(model_path: Path | None = None) -> Any:
    """Carga el modelo campeón desde disco.

    Soporta dos formatos, detectados por extensión:
      - ``.json``: XGBoost nativo (``xgb.XGBClassifier().load_model(...)``).
      - ``.joblib``: cualquier estimador sklearn-compatible serializado con
        ``joblib`` (LightGBM, CatBoost, ...).

    Args:
        model_path: Ruta al archivo del modelo. Si es None, usa la ruta por
            defecto definida en config.py (``MODEL_PATH``).

    Returns:
        Estimador con interfaz ``predict``/``predict_proba`` listo para usar.

    Raises:
        FileNotFoundError: Si el archivo del modelo no existe.
    """
    path = model_path or MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en {path}")

    if path.suffix == ".json":
        logger.info("Cargando modelo XGBoost (JSON nativo) desde %s", path)
        model = xgb.XGBClassifier()
        model.load_model(str(path))
    else:
        import joblib

        logger.info("Cargando modelo (%s) desde %s", path.suffix, path)
        model = joblib.load(path)

    logger.info(
        "Modelo cargado: %s (%s, %.1f MB)",
        path.name, type(model).__name__, path.stat().st_size / 1_048_576,
    )
    return model


def predict_single(
    model: Any,
    features_df: pd.DataFrame,
) -> tuple[int, np.ndarray]:
    """Realiza predicción para una sola factura.

    Args:
        model: Modelo XGBoost entrenado.
        features_df: DataFrame de forma (1, 48) con features en orden.

    Returns:
        Tupla (clase_predicha, array_probabilidades).
        - clase_predicha: int en {0, 1, 2}
        - array_probabilidades: ndarray de forma (3,) sumando ~1.0
    """
    y_pred = model.predict(features_df)[0]
    y_proba = model.predict_proba(features_df)[0]
    return int(y_pred), y_proba


def predict_batch(
    model: Any,
    features_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Realiza predicciones para un batch de facturas.

    Args:
        model: Modelo XGBoost entrenado.
        features_df: DataFrame de forma (N, 48) con features en orden.

    Returns:
        Tupla (clases_predichas, matriz_probabilidades).
        - clases_predichas: ndarray de forma (N,) con ints en {0, 1, 2}
        - matriz_probabilidades: ndarray de forma (N, 3) sumando ~1.0 por fila
    """
    logger.info("Predicción batch para %d facturas", len(features_df))
    y_pred = model.predict(features_df)
    y_proba = model.predict_proba(features_df)
    return y_pred.astype(int), y_proba
