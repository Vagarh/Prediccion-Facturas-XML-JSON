# -*- coding: utf-8 -*-
"""Explicabilidad SHAP para predicciones del modelo de auditoría.

Genera explicaciones en lenguaje natural para cada predicción, mapeando
los factores de riesgo más influyentes a causales de glosa de la
Resolución 2284 de 2023 (Manual Único de Glosas).

Refactorizado desde: notebooks/06_SHAP_Explainability.ipynb (celda 8).
"""

import logging

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.config import (
    CLASS_ACTIONS,
    CLASS_NAMES,
    FEATURE_LABELS,
    FEATURE_TO_CAUSAL,
    RISK_DESCRIPTIONS,
)

logger = logging.getLogger(__name__)


def create_shap_explainer(model: xgb.XGBClassifier) -> shap.TreeExplainer:
    """Crea un TreeExplainer para el modelo XGBoost.

    Se llama UNA sola vez al iniciar el servidor. El TreeExplainer para
    XGBoost no requiere datos de fondo — usa la estructura de los árboles
    directamente. Inicialización <1 segundo.

    Args:
        model: Modelo XGBoost entrenado.

    Returns:
        Instancia de shap.TreeExplainer lista para uso.
    """
    logger.info("Creando SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    logger.info("SHAP TreeExplainer creado exitosamente")
    return explainer


def explain_single_prediction(
    features_df: pd.DataFrame,
    model: xgb.XGBClassifier,
    explainer: shap.TreeExplainer,
    feature_names: list[str],
    top_n: int = 5,
) -> dict:
    """Genera explicación en lenguaje natural para una predicción individual.

    Calcula SHAP values para la factura y traduce los factores más
    influyentes a descripciones comprensibles para auditores médicos.

    Args:
        features_df: DataFrame de forma (1, 48) con las features en orden.
        model: Modelo XGBoost entrenado.
        explainer: TreeExplainer pre-inicializado.
        feature_names: Lista ordenada de 48 features.
        top_n: Número de factores de riesgo a retornar (por defecto 5).

    Returns:
        Diccionario con predicción, probabilidades, factores de riesgo y
        acción sugerida. Compatible con el schema de respuesta de la API.
    """
    # Predicción
    y_pred = model.predict(features_df)[0]
    y_proba = model.predict_proba(features_df)[0]
    predicted_class = int(y_pred)

    # SHAP values para la clase predicha
    sv = explainer(features_df)
    shap_for_pred_class = sv.values[0, :, predicted_class]

    # Ordenar features por |SHAP| descendente
    sorted_indices = np.argsort(np.abs(shap_for_pred_class))[::-1]

    # Generar factores de riesgo
    risk_factors = []
    causal = None

    for idx in sorted_indices[:top_n]:
        feat = feature_names[idx]
        shap_val = float(shap_for_pred_class[idx])
        feat_val = float(features_df[feat].iloc[0])

        description = _format_risk_factor(feat, shap_val, feat_val)
        risk_factors.append(
            {
                "feature": feat,
                "shap_value": shap_val,
                "feature_value": feat_val,
                "descripcion": description,
            }
        )

        if causal is None and feat in FEATURE_TO_CAUSAL:
            causal = FEATURE_TO_CAUSAL[feat]

    return {
        "estado_predicho": CLASS_NAMES[predicted_class].lower(),
        "clase": predicted_class,
        "confianza": float(y_proba[predicted_class]),
        "probabilidades": {
            "auditada": float(y_proba[0]),
            "glosada": float(y_proba[1]),
            "devuelta": float(y_proba[2]),
        },
        "factores_riesgo": [f["descripcion"] for f in risk_factors],
        "factores_detalle": risk_factors,
        "causal_probable": causal,
        "accion_sugerida": CLASS_ACTIONS[predicted_class],
    }


def explain_batch_predictions(
    features_df: pd.DataFrame,
    model: xgb.XGBClassifier,
    explainer: shap.TreeExplainer,
    feature_names: list[str],
    top_n: int = 5,
) -> list[dict]:
    """Genera explicaciones para un batch de predicciones.

    Optimizado: calcula SHAP values para todo el batch de una sola vez
    con explainer(X_batch), luego itera para armar explicaciones individuales.

    Args:
        features_df: DataFrame de forma (N, 48) con las features en orden.
        model: Modelo XGBoost entrenado.
        explainer: TreeExplainer pre-inicializado.
        feature_names: Lista ordenada de 48 features.
        top_n: Número de factores de riesgo por predicción.

    Returns:
        Lista de N diccionarios con explicaciones individuales.
    """
    n_samples = len(features_df)
    logger.info("Calculando SHAP values para batch de %d facturas", n_samples)

    # Predicciones batch
    y_pred = model.predict(features_df)
    y_proba = model.predict_proba(features_df)

    # SHAP values batch — shape: (N, 48, 3)
    sv = explainer(features_df)

    results = []
    for i in range(n_samples):
        predicted_class = int(y_pred[i])
        shap_for_pred_class = sv.values[i, :, predicted_class]

        sorted_indices = np.argsort(np.abs(shap_for_pred_class))[::-1]

        risk_factors = []
        causal = None

        for idx in sorted_indices[:top_n]:
            feat = feature_names[idx]
            shap_val = float(shap_for_pred_class[idx])
            feat_val = float(features_df.iloc[i][feat])

            description = _format_risk_factor(feat, shap_val, feat_val)
            risk_factors.append(
                {
                    "feature": feat,
                    "shap_value": shap_val,
                    "feature_value": feat_val,
                    "descripcion": description,
                }
            )

            if causal is None and feat in FEATURE_TO_CAUSAL:
                causal = FEATURE_TO_CAUSAL[feat]

        results.append(
            {
                "estado_predicho": CLASS_NAMES[predicted_class].lower(),
                "clase": predicted_class,
                "confianza": float(y_proba[i][predicted_class]),
                "probabilidades": {
                    "auditada": float(y_proba[i][0]),
                    "glosada": float(y_proba[i][1]),
                    "devuelta": float(y_proba[i][2]),
                },
                "factores_riesgo": [f["descripcion"] for f in risk_factors],
                "factores_detalle": risk_factors,
                "causal_probable": causal,
                "accion_sugerida": CLASS_ACTIONS[predicted_class],
            }
        )

    logger.info("Explicaciones generadas para %d facturas", n_samples)
    return results


def _format_risk_factor(
    feature_name: str,
    shap_value: float,
    feature_value: float,
) -> str:
    """Formatea un factor de riesgo en lenguaje natural para auditores.

    Usa las descripciones de RISK_DESCRIPTIONS si la feature está mapeada,
    o genera una descripción genérica basada en la etiqueta de la feature.

    Args:
        feature_name: Nombre de la feature.
        shap_value: Valor SHAP (positivo = aumenta probabilidad de la clase).
        feature_value: Valor numérico de la feature.

    Returns:
        Descripción en español del factor de riesgo.
    """
    if feature_name in RISK_DESCRIPTIONS:
        is_high_risky, text_high, text_low = RISK_DESCRIPTIONS[feature_name]
        if shap_value > 0:
            text = text_high if is_high_risky else text_low
        else:
            text = text_low if is_high_risky else text_high

        try:
            return text.format(val=feature_value)
        except (ValueError, KeyError):
            return text

    # Fallback: descripción genérica
    label = FEATURE_LABELS.get(feature_name, feature_name)
    direction = "alto" if shap_value > 0 else "bajo"
    return f"{label} = {feature_value:.2f} (impacto {direction})"
