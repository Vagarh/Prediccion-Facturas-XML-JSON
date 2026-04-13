# -*- coding: utf-8 -*-
"""Logger de experimentos de evaluación LLM en formato JSON Lines.

Registra cada ejecución de evaluación con timestamp ISO-8601, versión del
modelo, hiperparámetros y métricas completas para reproducibilidad.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import DATE_PREFIX, REPORTS_DIR

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Registra experimentos de evaluación en formato JSON Lines.

    Cada llamada a ``log_run`` agrega una línea JSON al archivo de log,
    permitiendo reconstruir el historial completo de experimentos.

    Attributes:
        log_path: Ruta al archivo JSON Lines de experimentos.
    """

    def __init__(self, log_path: Path | None = None) -> None:
        """Inicializa el logger con ruta al archivo de salida.

        Args:
            log_path: Ruta al archivo de log. Si None, usa
                ``REPORTS_DIR/experiment_log_{DATE_PREFIX}.json``.
        """
        if log_path is None:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            log_path = REPORTS_DIR / f"experiment_log_{DATE_PREFIX}.json"
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("ExperimentLogger inicializado: %s", self.log_path)

    def log_run(
        self,
        model_name: str,
        hyperparams: dict[str, Any],
        metrics: dict[str, Any],
        notes: str = "",
    ) -> dict[str, Any]:
        """Registra una ejecución de experimento en el archivo JSON Lines.

        Args:
            model_name: Nombre del modelo LLM o XGBoost evaluado.
            hyperparams: Diccionario de hiperparámetros usados.
            metrics: Métricas completas (F1-macro, Recall Glosada, AUC-ROC,
                Matriz de Confusión, etc.).
            notes: Notas opcionales sobre la ejecución.

        Returns:
            El registro completo serializado como diccionario.
        """
        record: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_name": model_name,
            "hyperparams": hyperparams,
            "metrics": metrics,
            "notes": notes,
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            "Experimento registrado: model=%s, F1=%.4f, Recall_Glosada=%.4f",
            model_name,
            metrics.get("f1_macro", 0.0),
            metrics.get("recall_glosada", 0.0),
        )
        return record

    def load_all(self) -> list[dict[str, Any]]:
        """Carga todos los registros del archivo JSON Lines.

        Returns:
            Lista de registros ordenados por timestamp ascendente.
            Lista vacía si el archivo no existe.
        """
        if not self.log_path.exists():
            return []

        records = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning("Línea inválida en log: %s", e)

        return sorted(records, key=lambda r: r.get("timestamp", ""))

    def get_best_run(self, metric: str = "recall_glosada") -> dict[str, Any] | None:
        """Retorna el experimento con la mejor métrica especificada.

        Args:
            metric: Nombre de la métrica a maximizar.

        Returns:
            Registro del mejor experimento, o None si no hay registros.
        """
        records = self.load_all()
        if not records:
            return None

        return max(
            records,
            key=lambda r: r.get("metrics", {}).get(metric, float("-inf")),
        )
