# -*- coding: utf-8 -*-
"""Evaluador experimental de candidatos LLM para auditoría prestacional.

Compara múltiples modelos LLM candidatos (Llama 3.2, Mistral 7B, Qwen2.5, Phi-3 Mini)
en métricas de negocio (Recall Glosada, F1-macro, AUC-ROC), separabilidad semántica
de embeddings y latencia de inferencia. Produce un reporte JSON con recomendación
justificada o documenta el resultado negativo si ningún candidato supera el baseline.

Cumplimiento:
    - Toda inferencia es local (sin llamadas a APIs externas).
    - No se usa ``print()`` — se usa ``logging``.
    - Ley 1581 de Colombia: los datos clínicos nunca salen de la infraestructura.
    - ``RANDOM_STATE = 42`` en todos los componentes con aleatoriedad.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import silhouette_score

from src.config import LLM_CANDIDATES, RANDOM_STATE, REPORTS_DIR
from src.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)

# Umbral de Recall Glosada del modelo base XGBoost v1
BASELINE_RECALL_GLOSADA: float = 0.7261

# Índice de la clase Glosada en la tarea multiclase (0=Auditada, 1=Glosada, 2=Devuelta)
CLASS_GLOSADA: int = 1
CLASS_NAMES: list[str] = ["Auditada", "Glosada", "Devuelta"]


class LLMEvaluator:
    """Evaluador experimental de candidatos LLM para el pipeline de auditoría.

    Evalúa cada modelo candidato en métricas de negocio (Recall Glosada, F1-macro,
    AUC-ROC), separabilidad semántica de embeddings y latencia de inferencia.
    Produce un reporte JSON comparativo con recomendación justificada.

    Attributes:
        embedding_service: Servicio de embeddings para generar representaciones
            vectoriales de los campos textuales clínicos.
        random_state: Semilla de aleatoriedad para reproducibilidad.

    Example:
        >>> from src.embeddings.service import EmbeddingService
        >>> service = EmbeddingService()
        >>> evaluator = LLMEvaluator(embedding_service=service)
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        random_state: int = RANDOM_STATE,
    ) -> None:
        """Inicializa el evaluador con el servicio de embeddings.

        Args:
            embedding_service: Instancia de ``EmbeddingService`` para generar
                embeddings semánticos de los campos textuales clínicos.
            random_state: Semilla de aleatoriedad para reproducibilidad.
                Por defecto usa ``RANDOM_STATE`` de ``config.py`` (42).
        """
        self.embedding_service = embedding_service
        self.random_state = random_state
        logger.info(
            "LLMEvaluator inicializado (random_state=%d).", random_state
        )

    def evaluate_candidate(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model: Any | None = None,
    ) -> dict[str, Any]:
        """Evalúa un modelo candidato en el conjunto de test.

        Calcula métricas de negocio: Recall Glosada, F1-macro, AUC-ROC,
        Recall por clase y Matriz de Confusión. Si se proporciona un ``model``
        con interfaz sklearn (``predict`` / ``predict_proba``), se usa para
        obtener predicciones. De lo contrario, se espera que ``X_test``
        contenga las predicciones directamente (columna de clases predichas).

        Args:
            model_name: Nombre del modelo candidato (p. ej. ``"llama3.2"``).
            X_test: Array NumPy de forma ``(n_samples, n_features)`` con las
                features del conjunto de test, o de forma ``(n_samples,)`` con
                las clases predichas si no se proporciona ``model``.
            y_test: Array NumPy de forma ``(n_samples,)`` con las etiquetas
                verdaderas (0=Auditada, 1=Glosada, 2=Devuelta).
            model: Modelo con interfaz sklearn (``predict`` y ``predict_proba``).
                Si es ``None``, se interpreta ``X_test`` como predicciones directas
                de forma ``(n_samples,)``.

        Returns:
            Diccionario con las métricas calculadas::

                {
                    "model_name": str,
                    "recall_glosada": float,
                    "f1_macro": float,
                    "auc_roc": float,
                    "recall_per_class": {"Auditada": float, "Glosada": float, "Devuelta": float},
                    "confusion_matrix": list[list[int]],
                    "exceeds_baseline": bool,
                }

        Raises:
            ValueError: Si ``y_test`` no contiene exactamente 3 clases distintas
                o si las dimensiones de ``X_test`` y ``y_test`` son incompatibles.

        Example:
            >>> import numpy as np
            >>> evaluator = LLMEvaluator(embedding_service=service)
            >>> y_pred = np.array([0, 1, 2, 1, 0])
            >>> y_true = np.array([0, 1, 2, 0, 0])
            >>> metrics = evaluator.evaluate_candidate("llama3.2", y_pred, y_true)
            >>> "recall_glosada" in metrics
            True
        """
        logger.info("evaluate_candidate: evaluando modelo '%s'.", model_name)

        # Obtener predicciones
        if model is not None:
            logger.debug(
                "evaluate_candidate: usando model.predict sobre X_test shape %s.",
                X_test.shape,
            )
            y_pred = model.predict(X_test)
            try:
                y_proba = model.predict_proba(X_test)
            except AttributeError:
                logger.warning(
                    "evaluate_candidate: modelo '%s' no tiene predict_proba; "
                    "AUC-ROC se calculará con one-hot de predicciones.",
                    model_name,
                )
                y_proba = _one_hot_predictions(y_pred, n_classes=3)
        else:
            # X_test contiene directamente las clases predichas
            if X_test.ndim != 1:
                raise ValueError(
                    "Cuando no se proporciona 'model', X_test debe ser 1-D "
                    f"(clases predichas). Se recibió shape {X_test.shape}."
                )
            y_pred = X_test.astype(int)
            y_proba = _one_hot_predictions(y_pred, n_classes=3)

        if y_pred.shape[0] != y_test.shape[0]:
            raise ValueError(
                f"Dimensiones incompatibles: y_pred tiene {y_pred.shape[0]} muestras "
                f"pero y_test tiene {y_test.shape[0]}."
            )

        # Métricas sklearn
        recall_per_class_arr = recall_score(
            y_test, y_pred, average=None, labels=[0, 1, 2], zero_division=0
        )
        recall_glosada = float(recall_per_class_arr[CLASS_GLOSADA])
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        try:
            auc_roc = float(
                roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
            )
        except ValueError as exc:
            logger.warning(
                "evaluate_candidate: no se pudo calcular AUC-ROC para '%s': %s. "
                "Se asigna 0.0.",
                model_name,
                exc,
            )
            auc_roc = 0.0

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

        recall_per_class = {
            CLASS_NAMES[i]: float(recall_per_class_arr[i]) for i in range(3)
        }

        exceeds_baseline = recall_glosada > BASELINE_RECALL_GLOSADA

        result: dict[str, Any] = {
            "model_name": model_name,
            "recall_glosada": recall_glosada,
            "f1_macro": f1_macro,
            "auc_roc": auc_roc,
            "recall_per_class": recall_per_class,
            "confusion_matrix": cm.tolist(),
            "exceeds_baseline": exceeds_baseline,
        }

        logger.info(
            "evaluate_candidate: '%s' → recall_glosada=%.4f, f1_macro=%.4f, "
            "auc_roc=%.4f, exceeds_baseline=%s.",
            model_name,
            recall_glosada,
            f1_macro,
            auc_roc,
            exceeds_baseline,
        )
        return result

    def measure_embedding_separability(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Calcula la separabilidad semántica entre las 3 clases usando silhouette score.

        Mide qué tan bien separados están los embeddings de las clases
        Auditada (0), Glosada (1) y Devuelta (2) en el espacio vectorial.
        Un valor cercano a 1.0 indica alta separabilidad; cercano a -1.0
        indica que las clases se solapan.

        Args:
            embeddings: Array NumPy de forma ``(n_samples, embedding_dim)``
                con los vectores de embeddings generados por el modelo candidato.
            labels: Array NumPy de forma ``(n_samples,)`` con las etiquetas
                de clase (0=Auditada, 1=Glosada, 2=Devuelta).

        Returns:
            Silhouette score en el rango ``[-1.0, 1.0]``. Retorna ``0.0`` si
            no hay suficientes muestras o clases para calcular la métrica.

        Raises:
            ValueError: Si ``embeddings`` y ``labels`` tienen longitudes distintas.

        Example:
            >>> import numpy as np
            >>> embs = np.random.randn(300, 384)
            >>> lbls = np.array([0] * 100 + [1] * 100 + [2] * 100)
            >>> score = evaluator.measure_embedding_separability(embs, lbls)
            >>> -1.0 <= score <= 1.0
            True
        """
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                f"embeddings tiene {embeddings.shape[0]} muestras pero "
                f"labels tiene {labels.shape[0]}."
            )

        n_samples = embeddings.shape[0]
        n_unique_labels = len(np.unique(labels))

        if n_samples < 2 or n_unique_labels < 2:
            logger.warning(
                "measure_embedding_separability: insuficientes muestras (%d) o "
                "clases (%d) para calcular silhouette score. Retornando 0.0.",
                n_samples,
                n_unique_labels,
            )
            return 0.0

        try:
            score = float(
                silhouette_score(
                    embeddings,
                    labels,
                    random_state=self.random_state,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "measure_embedding_separability: error calculando silhouette score: %s. "
                "Retornando 0.0.",
                exc,
            )
            return 0.0

        logger.info(
            "measure_embedding_separability: silhouette score = %.4f "
            "(%d muestras, %d clases).",
            score,
            n_samples,
            n_unique_labels,
        )
        return score

    def measure_latency(
        self,
        model_name: str,
        sample_size: int = 1000,
    ) -> dict[str, float]:
        """Mide la latencia de inferencia del modelo de embeddings.

        Cronometra ``sample_size`` llamadas a ``EmbeddingService.encode()``
        con textos sintéticos representativos y calcula los percentiles
        p50 y p95 en milisegundos.

        Args:
            model_name: Nombre del modelo candidato (usado para logging).
            sample_size: Número de llamadas a cronometrar. Por defecto 1000.

        Returns:
            Diccionario con los percentiles de latencia::

                {
                    "p50_ms": float,  # Mediana de latencia en ms
                    "p95_ms": float,  # Percentil 95 de latencia en ms
                }

        Example:
            >>> latency = evaluator.measure_latency("llama3.2", sample_size=10)
            >>> "p50_ms" in latency and "p95_ms" in latency
            True
            >>> latency["p50_ms"] >= 0.0
            True
        """
        logger.info(
            "measure_latency: midiendo latencia de '%s' con %d muestras.",
            model_name,
            sample_size,
        )

        # Textos sintéticos representativos de diagnósticos CIE-10 y CUPS
        sample_texts = _generate_sample_texts(sample_size, random_state=self.random_state)

        latencies_ms: list[float] = []
        for text in sample_texts:
            t0 = time.perf_counter()
            self.embedding_service.encode(text)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

        arr = np.array(latencies_ms, dtype=np.float64)
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))

        logger.info(
            "measure_latency: '%s' → p50=%.2f ms, p95=%.2f ms.",
            model_name,
            p50,
            p95,
        )
        return {"p50_ms": p50, "p95_ms": p95}

    def run_full_evaluation(
        self,
        candidates: list[str],
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        embeddings_per_candidate: dict[str, np.ndarray] | None = None,
        models: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evalúa todos los candidatos LLM y produce un reporte JSON comparativo.

        Para cada candidato en ``candidates``, ejecuta:
        1. ``evaluate_candidate`` — métricas de negocio (si se proporcionan datos).
        2. ``measure_embedding_separability`` — separabilidad semántica (si se
           proporcionan embeddings).
        3. ``measure_latency`` — latencia de inferencia p50/p95.

        Aplica la lógica de resultado negativo: si ningún candidato supera el
        Recall Glosada baseline (0.7261), documenta el resultado y recomienda
        continuar con el pipeline XGBoost base sin integración LLM.

        Serializa el reporte en ``Codigo/reports/llm_evaluation_{DATE_PREFIX}.json``.

        Args:
            candidates: Lista de nombres de modelos candidatos a evaluar.
                Por defecto se usa ``LLM_CANDIDATES`` de ``config.py``.
            X_test: Array NumPy con features o predicciones del conjunto de test.
                Si es ``None``, se omite ``evaluate_candidate``.
            y_test: Array NumPy con etiquetas verdaderas del conjunto de test.
                Si es ``None``, se omite ``evaluate_candidate``.
            embeddings_per_candidate: Diccionario ``{model_name: embeddings_array}``
                con los embeddings generados por cada candidato. Si es ``None``,
                se omite ``measure_embedding_separability``.
            models: Diccionario ``{model_name: model_object}`` con los modelos
                entrenados por candidato. Si es ``None``, se usa ``X_test`` como
                predicciones directas.

        Returns:
            Diccionario con el reporte completo::

                {
                    "timestamp": str,           # ISO-8601
                    "baseline_recall_glosada": float,
                    "candidates": {
                        "<model_name>": {
                            "metrics": dict | None,
                            "separability": float | None,
                            "latency": dict,
                        },
                        ...
                    },
                    "recommendation": {
                        "selected_model": str | None,
                        "justification": str,
                        "negative_result": bool,
                    },
                }

        Example:
            >>> report = evaluator.run_full_evaluation(["llama3.2"])
            >>> "recommendation" in report
            True
            >>> report["recommendation"]["negative_result"] in (True, False)
            True
        """
        logger.info(
            "run_full_evaluation: iniciando evaluación de %d candidatos: %s.",
            len(candidates),
            candidates,
        )

        report: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "baseline_recall_glosada": BASELINE_RECALL_GLOSADA,
            "candidates": {},
        }

        best_model: str | None = None
        best_recall: float = -1.0

        for candidate in candidates:
            logger.info("run_full_evaluation: procesando candidato '%s'.", candidate)
            candidate_result: dict[str, Any] = {
                "metrics": None,
                "separability": None,
                "latency": None,
            }

            # 1. Métricas de negocio
            if X_test is not None and y_test is not None:
                model_obj = models.get(candidate) if models else None
                try:
                    metrics = self.evaluate_candidate(
                        model_name=candidate,
                        X_test=X_test,
                        y_test=y_test,
                        model=model_obj,
                    )
                    candidate_result["metrics"] = metrics
                    if metrics["recall_glosada"] > best_recall:
                        best_recall = metrics["recall_glosada"]
                        best_model = candidate
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "run_full_evaluation: error evaluando métricas de '%s': %s.",
                        candidate,
                        exc,
                    )

            # 2. Separabilidad semántica
            if embeddings_per_candidate is not None:
                embs = embeddings_per_candidate.get(candidate)
                if embs is not None and y_test is not None:
                    try:
                        sep = self.measure_embedding_separability(embs, y_test)
                        candidate_result["separability"] = sep
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "run_full_evaluation: error midiendo separabilidad de '%s': %s.",
                            candidate,
                            exc,
                        )

            # 3. Latencia
            try:
                latency = self.measure_latency(candidate)
                candidate_result["latency"] = latency
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "run_full_evaluation: error midiendo latencia de '%s': %s.",
                    candidate,
                    exc,
                )

            report["candidates"][candidate] = candidate_result

        # Lógica de resultado negativo (Requisito 3.6)
        negative_result = best_recall <= BASELINE_RECALL_GLOSADA or best_model is None

        if negative_result:
            justification = (
                f"Ningún candidato LLM evaluado superó el Recall Glosada baseline "
                f"del modelo XGBoost base ({BASELINE_RECALL_GLOSADA:.4f}). "
                f"Mejor resultado obtenido: {best_recall:.4f} "
                f"({'sin datos de evaluación' if best_model is None else best_model}). "
                f"Se recomienda continuar con el pipeline XGBoost base sin integración LLM "
                f"hasta disponer de mejores candidatos o datos adicionales de entrenamiento."
            )
            selected_model = None
            logger.warning(
                "run_full_evaluation: RESULTADO NEGATIVO — ningún candidato supera "
                "el baseline. Recomendación: continuar con pipeline base."
            )
        else:
            best_metrics = report["candidates"][best_model]["metrics"] or {}
            justification = (
                f"El modelo '{best_model}' supera el Recall Glosada baseline "
                f"({BASELINE_RECALL_GLOSADA:.4f}) con un valor de {best_recall:.4f} "
                f"(+{best_recall - BASELINE_RECALL_GLOSADA:.4f}). "
                f"F1-macro: {best_metrics.get('f1_macro', 'N/A'):.4f}, "
                f"AUC-ROC: {best_metrics.get('auc_roc', 'N/A'):.4f}. "
                f"Se recomienda integrar '{best_model}' en el pipeline de producción."
            )
            selected_model = best_model
            logger.info(
                "run_full_evaluation: modelo recomendado '%s' con recall_glosada=%.4f.",
                best_model,
                best_recall,
            )

        report["recommendation"] = {
            "selected_model": selected_model,
            "justification": justification,
            "negative_result": negative_result,
        }

        # Serializar reporte a JSON
        _save_report(report)

        return report


# ---------------------------------------------------------------------------
# Funciones auxiliares privadas
# ---------------------------------------------------------------------------


def _one_hot_predictions(y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Convierte predicciones de clase en matriz one-hot para AUC-ROC.

    Args:
        y_pred: Array 1-D con clases predichas (enteros 0..n_classes-1).
        n_classes: Número de clases. Por defecto 3.

    Returns:
        Array NumPy de forma ``(len(y_pred), n_classes)`` con valores 0/1.
    """
    n = len(y_pred)
    one_hot = np.zeros((n, n_classes), dtype=np.float32)
    for i, cls in enumerate(y_pred):
        if 0 <= int(cls) < n_classes:
            one_hot[i, int(cls)] = 1.0
    return one_hot


def _generate_sample_texts(n: int, random_state: int = RANDOM_STATE) -> list[str]:
    """Genera textos sintéticos representativos de diagnósticos CIE-10 y CUPS.

    Usado para medir la latencia de inferencia del servicio de embeddings
    con entradas realistas del dominio clínico.

    Args:
        n: Número de textos a generar.
        random_state: Semilla de aleatoriedad para reproducibilidad.

    Returns:
        Lista de ``n`` cadenas de texto sintéticas.
    """
    rng = np.random.default_rng(random_state)
    templates = [
        "hipertensión arterial esencial",
        "diabetes mellitus tipo 2 sin complicaciones",
        "insuficiencia cardíaca congestiva",
        "neumonía bacteriana adquirida en la comunidad",
        "fractura de fémur proximal",
        "consulta médica general por dolor abdominal",
        "procedimiento quirúrgico ambulatorio",
        "control prenatal de bajo riesgo",
        "terapia física y rehabilitación",
        "examen de laboratorio clínico de rutina",
    ]
    indices = rng.integers(0, len(templates), size=n)
    return [templates[i] for i in indices]


def _save_report(report: dict[str, Any]) -> None:
    """Serializa el reporte de evaluación en formato JSON.

    Guarda el reporte en ``Codigo/reports/llm_evaluation_{DATE_PREFIX}.json``
    donde ``DATE_PREFIX`` tiene el formato ``YYYYMMDD`` de la fecha actual.

    Args:
        report: Diccionario con el reporte completo de evaluación.
    """
    date_prefix = datetime.utcnow().strftime("%Y%m%d")
    reports_dir = Path(REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_path = reports_dir / f"llm_evaluation_{date_prefix}.json"

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    logger.info(
        "_save_report: reporte serializado en '%s'.",
        output_path,
    )
