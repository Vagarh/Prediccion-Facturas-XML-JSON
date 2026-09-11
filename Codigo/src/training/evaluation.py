# -*- coding: utf-8 -*-
"""Evaluación multiclase unificada para la suite de modelos (Fase 3).

Consolida el patrón de ``notebooks/05_Model_Training.ipynb`` (celda 7) y
``src/evaluation/retrain_xgboost.py`` en una función pura y testeable.

Métricas obligatorias (CLAUDE.md §2): F1-macro, AUC-ROC OvR, Recall por clase
(primaria: Recall Glosada = clase 1), matriz de confusión.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Auditada", "Glosada", "Devuelta"]
CLASS_GLOSADA = 1


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    *,
    model_name: str = "modelo",
    dataset: str = "test",
) -> dict[str, Any]:
    """Calcula todas las métricas obligatorias para una predicción multiclase.

    Args:
        y_true: Etiquetas reales, valores en ``{0, 1, 2}``.
        y_pred: Etiquetas predichas.
        y_proba: Probabilidades ``(n, 3)`` en orden de clase 0/1/2.
        model_name: Identificador del modelo (para logging/reporte).
        dataset: Nombre del conjunto evaluado (``test``/``oot``/...).

    Returns:
        Diccionario JSON-serializable con las métricas.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    recalls = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)

    try:
        auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=[0, 1, 2]))
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report = classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )

    metrics = {
        "model_name": model_name,
        "dataset": dataset,
        "n_samples": int(len(y_true)),
        "f1_macro": f1_macro,
        "auc_roc_ovr": auc,
        "recall_glosada": float(recalls[CLASS_GLOSADA]),
        "recall_per_class": {CLASS_NAMES[i]: float(recalls[i]) for i in range(3)},
        "accuracy": float((y_true == y_pred).mean()),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    logger.info(
        "[%s/%s] F1=%.4f AUC=%.4f RecallGlosada=%.4f",
        model_name, dataset, f1_macro, auc, metrics["recall_glosada"],
    )
    return metrics


def select_champion(
    results: dict[str, dict[str, dict[str, Any]]],
    *,
    min_recall_glosada_test: float = 0.80,
    min_recall_glosada_oot: float = 0.75,
    f1_tie_band: float = 0.01,
) -> dict[str, Any]:
    """Aplica la regla de selección de campeón (plan Fase 3, refinada).

    1. Filtro duro: Recall Glosada test ≥ ``min_recall_glosada_test`` y
       (si hay OOT) Recall Glosada OOT ≥ ``min_recall_glosada_oot``.
    2. Entre los que pasan, se arma un grupo "empatado en F1": todos dentro de
       ``f1_tie_band`` del máximo F1-macro (test). Diferencias de F1 menores a
       ese margen se consideran ruido de muestreo, no una ventaja real.
    3. Dentro del grupo empatado, el campeón es el de **mayor Recall Glosada
       OOT** (la métrica primaria del proyecto — CLAUDE.md §2 — evaluada en el
       escenario más honesto de generalización); si no hay OOT, se usa el de
       test. Fuera del grupo empatado, gana simplemente el de mayor F1.

    Args:
        results: ``{model_name: {"test": metrics, "oot": metrics?}}``.

    Returns:
        Dict con ``champion``, ``eligibles``, ``ranking`` (por F1, informativo)
        y ``tie_break`` (True si el campeón se decidió por Recall Glosada).
    """
    rows = []
    for name, per_ds in results.items():
        test = per_ds.get("test")
        if test is None:
            continue
        oot = per_ds.get("oot")
        rg_test = test["recall_glosada"]
        rg_oot = oot["recall_glosada"] if oot else None
        eligible = rg_test >= min_recall_glosada_test and (
            rg_oot is None or rg_oot >= min_recall_glosada_oot
        )
        degradation = (test["f1_macro"] - oot["f1_macro"]) if oot else 0.0
        rows.append({
            "model": name,
            "f1_macro_test": test["f1_macro"],
            "recall_glosada_test": rg_test,
            "recall_glosada_oot": rg_oot,
            "f1_degradation": degradation,
            "eligible": eligible,
        })

    ranking = sorted(rows, key=lambda r: (-r["f1_macro_test"], r["f1_degradation"]))
    eligibles = [r for r in ranking if r["eligible"]]

    champion = None
    tie_break = False
    if eligibles:
        best_f1 = eligibles[0]["f1_macro_test"]
        tied = [r for r in eligibles if best_f1 - r["f1_macro_test"] <= f1_tie_band]
        if len(tied) > 1:
            tied_by_rg = sorted(
                tied, key=lambda r: (r["recall_glosada_oot"] if r["recall_glosada_oot"] is not None
                                     else r["recall_glosada_test"]), reverse=True,
            )
            champion = tied_by_rg[0]["model"]
            tie_break = True
        else:
            champion = eligibles[0]["model"]

    return {
        "champion": champion, "tie_break": tie_break,
        "eligibles": [r["model"] for r in eligibles], "ranking": ranking,
    }
