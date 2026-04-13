# -*- coding: utf-8 -*-
"""Monitor de rendimiento del modelo en producción.

Almacena métricas diarias en SQLite, detecta degradación y genera
reportes mensuales de tendencias.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, recall_score

from src.config import MODELS_DIR, MODEL_METRICS

logger = logging.getLogger(__name__)

DB_PATH = MODELS_DIR / "monitor.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS performance_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT NOT NULL,
    n_samples   INTEGER NOT NULL,
    f1_macro    REAL NOT NULL,
    recall_glosada REAL NOT NULL,
    recall_auditada REAL NOT NULL,
    recall_devuelta REAL NOT NULL,
    accuracy    REAL NOT NULL,
    extra       TEXT
);
"""


def _get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Retorna conexión SQLite inicializada.

    Args:
        db_path: Ruta al archivo de base de datos.

    Returns:
        Conexión SQLite con row_factory configurado.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


class ModelMonitor:
    """Monitor de rendimiento del modelo XGBoost en producción.

    Registra métricas diarias, detecta degradación y genera reportes
    mensuales para el sistema de reentrenamiento automático.

    Attributes:
        db_path: Ruta a la base de datos SQLite de métricas.
        baseline_metrics: Métricas de referencia del modelo en producción.
    """

    def __init__(
        self,
        db_path: Path = DB_PATH,
        baseline_metrics: dict[str, float] | None = None,
    ) -> None:
        """Inicializa el monitor con métricas de referencia.

        Args:
            db_path: Ruta a la base de datos SQLite.
            baseline_metrics: Métricas base del modelo actual. Si None,
                usa MODEL_METRICS de config.
        """
        self.db_path = db_path
        self.baseline_metrics = baseline_metrics or MODEL_METRICS
        logger.info("ModelMonitor inicializado con baseline: %s", self.baseline_metrics)

    def track_performance(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Registra métricas de rendimiento para un lote de predicciones.

        Args:
            predictions: Array de predicciones del modelo (0, 1, 2).
            actuals: Array de etiquetas reales (0, 1, 2).
            extra: Metadatos adicionales opcionales (e.g., fecha, prestador).

        Returns:
            Diccionario con métricas calculadas y almacenadas.
        """
        predictions = np.asarray(predictions)
        actuals = np.asarray(actuals)

        labels = [0, 1, 2]
        recall_per_class = recall_score(actuals, predictions, labels=labels, average=None, zero_division=0)
        metrics = {
            "f1_macro": float(f1_score(actuals, predictions, average="macro", zero_division=0)),
            "recall_glosada": float(recall_per_class[1]),
            "recall_auditada": float(recall_per_class[0]),
            "recall_devuelta": float(recall_per_class[2]),
            "accuracy": float(np.mean(predictions == actuals)),
        }

        with _get_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO performance_log
                    (recorded_at, n_samples, f1_macro, recall_glosada,
                     recall_auditada, recall_devuelta, accuracy, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    len(actuals),
                    metrics["f1_macro"],
                    metrics["recall_glosada"],
                    metrics["recall_auditada"],
                    metrics["recall_devuelta"],
                    metrics["accuracy"],
                    json.dumps(extra) if extra else None,
                ),
            )
            conn.commit()

        logger.info(
            "Métricas registradas: F1=%.4f, Recall_Glosada=%.4f (n=%d)",
            metrics["f1_macro"],
            metrics["recall_glosada"],
            len(actuals),
        )
        return metrics

    def detect_performance_drop(self, threshold: float = 0.05) -> dict[str, Any]:
        """Detecta caída de rendimiento respecto al baseline.

        Compara el promedio de las últimas 7 entradas con el baseline.

        Args:
            threshold: Caída mínima (absoluta) para activar alerta.

        Returns:
            Diccionario con ``drop_detected`` (bool), ``current_metrics``,
            ``baseline_metrics`` y ``drops`` (dict de caídas por métrica).
        """
        history = self.get_model_metrics_history(last_n=7)
        if not history:
            return {"drop_detected": False, "reason": "Sin historial suficiente"}

        avg_f1 = float(np.mean([r["f1_macro"] for r in history]))
        avg_recall_glosada = float(np.mean([r["recall_glosada"] for r in history]))

        drops = {
            "f1_macro": self.baseline_metrics.get("f1_macro", 0) - avg_f1,
            "recall_glosada": self.baseline_metrics.get("recall_glosada", 0) - avg_recall_glosada,
        }

        drop_detected = any(v > threshold for v in drops.values())

        if drop_detected:
            logger.warning(
                "Degradación detectada: F1 drop=%.4f, Recall_Glosada drop=%.4f",
                drops["f1_macro"],
                drops["recall_glosada"],
            )

        return {
            "drop_detected": drop_detected,
            "current_metrics": {"f1_macro": avg_f1, "recall_glosada": avg_recall_glosada},
            "baseline_metrics": self.baseline_metrics,
            "drops": drops,
            "threshold": threshold,
        }

    def get_model_metrics_history(
        self, last_n: int | None = None, days_back: int | None = None
    ) -> list[dict[str, Any]]:
        """Retorna el histórico de métricas registradas.

        Args:
            last_n: Número de registros más recientes a retornar.
            days_back: Filtrar por los últimos N días.

        Returns:
            Lista de diccionarios con métricas ordenadas por fecha descendente.
        """
        query = "SELECT * FROM performance_log"
        params: list[Any] = []

        if days_back is not None:
            cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
            query += " WHERE recorded_at >= ?"
            params.append(cutoff)

        query += " ORDER BY recorded_at DESC"

        if last_n is not None:
            query += f" LIMIT {int(last_n)}"

        with _get_conn(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            {
                "id": r["id"],
                "recorded_at": r["recorded_at"],
                "n_samples": r["n_samples"],
                "f1_macro": r["f1_macro"],
                "recall_glosada": r["recall_glosada"],
                "recall_auditada": r["recall_auditada"],
                "recall_devuelta": r["recall_devuelta"],
                "accuracy": r["accuracy"],
            }
            for r in rows
        ]

    def generate_monthly_report(self) -> dict[str, Any]:
        """Genera reporte de rendimiento del último mes.

        Returns:
            Diccionario con estadísticas del mes: promedio, mínimo, máximo
            por métrica, tendencia y alertas de degradación.
        """
        history = self.get_model_metrics_history(days_back=30)
        if not history:
            return {"status": "sin_datos", "period_days": 30}

        f1_values = [r["f1_macro"] for r in history]
        recall_values = [r["recall_glosada"] for r in history]

        report = {
            "period_days": 30,
            "n_records": len(history),
            "f1_macro": {
                "mean": float(np.mean(f1_values)),
                "min": float(np.min(f1_values)),
                "max": float(np.max(f1_values)),
                "std": float(np.std(f1_values)),
            },
            "recall_glosada": {
                "mean": float(np.mean(recall_values)),
                "min": float(np.min(recall_values)),
                "max": float(np.max(recall_values)),
                "std": float(np.std(recall_values)),
            },
            "degradation_check": self.detect_performance_drop(),
            "generated_at": datetime.utcnow().isoformat(),
        }

        logger.info(
            "Reporte mensual generado: F1_mean=%.4f, Recall_Glosada_mean=%.4f",
            report["f1_macro"]["mean"],
            report["recall_glosada"]["mean"],
        )
        return report
