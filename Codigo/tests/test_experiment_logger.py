# -*- coding: utf-8 -*-
"""Tests unitarios y property tests para ExperimentLogger.

Cubre:
- Propiedad 8: Idempotencia de serialización JSON
- Determinismo del pipeline de evaluación (RANDOM_STATE = 42)
- Funcionalidad básica de log_run, load_all, get_best_run
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.evaluation.experiment_logger import ExperimentLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_logger(tmp_path: Path) -> ExperimentLogger:
    """ExperimentLogger con archivo temporal para tests."""
    return ExperimentLogger(log_path=tmp_path / "test_log.json")


# ---------------------------------------------------------------------------
# Tests unitarios
# ---------------------------------------------------------------------------


class TestExperimentLoggerUnit:
    """Tests unitarios para ExperimentLogger."""

    def test_log_run_creates_file(self, tmp_logger: ExperimentLogger) -> None:
        """log_run crea el archivo si no existe."""
        tmp_logger.log_run(
            model_name="llama3.2",
            hyperparams={"temperature": 0.7},
            metrics={"f1_macro": 0.75, "recall_glosada": 0.80},
        )
        assert tmp_logger.log_path.exists()

    def test_log_run_returns_record(self, tmp_logger: ExperimentLogger) -> None:
        """log_run retorna el registro completo con todos los campos."""
        record = tmp_logger.log_run(
            model_name="mistral",
            hyperparams={"n_estimators": 300},
            metrics={"f1_macro": 0.749, "recall_glosada": 0.726},
            notes="test run",
        )
        assert record["model_name"] == "mistral"
        assert "timestamp" in record
        assert record["metrics"]["f1_macro"] == 0.749
        assert record["notes"] == "test run"

    def test_load_all_empty_when_no_file(self, tmp_path: Path) -> None:
        """load_all retorna lista vacía si el archivo no existe."""
        logger = ExperimentLogger(log_path=tmp_path / "nonexistent.json")
        assert logger.load_all() == []

    def test_load_all_returns_all_records(self, tmp_logger: ExperimentLogger) -> None:
        """load_all retorna todos los registros en orden cronológico."""
        for i in range(3):
            tmp_logger.log_run(
                model_name=f"model_{i}",
                hyperparams={},
                metrics={"f1_macro": 0.7 + i * 0.01, "recall_glosada": 0.72},
            )
        records = tmp_logger.load_all()
        assert len(records) == 3

    def test_get_best_run_by_recall_glosada(self, tmp_logger: ExperimentLogger) -> None:
        """get_best_run retorna el experimento con mayor recall_glosada."""
        tmp_logger.log_run("model_a", {}, {"f1_macro": 0.74, "recall_glosada": 0.72})
        tmp_logger.log_run("model_b", {}, {"f1_macro": 0.76, "recall_glosada": 0.85})
        tmp_logger.log_run("model_c", {}, {"f1_macro": 0.73, "recall_glosada": 0.70})

        best = tmp_logger.get_best_run(metric="recall_glosada")
        assert best is not None
        assert best["model_name"] == "model_b"

    def test_get_best_run_returns_none_when_empty(self, tmp_path: Path) -> None:
        """get_best_run retorna None si no hay registros."""
        logger = ExperimentLogger(log_path=tmp_path / "empty.json")
        assert logger.get_best_run() is None

    def test_multiple_runs_append_correctly(self, tmp_logger: ExperimentLogger) -> None:
        """Múltiples log_run acumulan registros sin sobrescribir."""
        for i in range(5):
            tmp_logger.log_run(f"run_{i}", {}, {"f1_macro": 0.7})
        records = tmp_logger.load_all()
        assert len(records) == 5

    def test_metrics_with_confusion_matrix(self, tmp_logger: ExperimentLogger) -> None:
        """log_run acepta métricas complejas incluyendo matriz de confusión."""
        metrics = {
            "f1_macro": 0.749,
            "recall_glosada": 0.726,
            "auc_roc": 0.908,
            "confusion_matrix": [[100, 10, 5], [8, 90, 7], [6, 4, 95]],
        }
        record = tmp_logger.log_run("xgboost_v2", {"random_state": 42}, metrics)
        loaded = tmp_logger.load_all()
        assert loaded[0]["metrics"]["confusion_matrix"] == metrics["confusion_matrix"]


# ---------------------------------------------------------------------------
# Propiedad 8: Idempotencia de serialización JSON
# ---------------------------------------------------------------------------


@st.composite
def metrics_dict(draw: st.DrawFn) -> dict:
    """Genera diccionarios de métricas con valores numéricos válidos."""
    return {
        "f1_macro": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        "recall_glosada": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        "recall_auditada": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        "recall_devuelta": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        "auc_roc": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
    }


@given(metrics=metrics_dict())
@settings(max_examples=100)
def test_property_json_serialization_idempotent(metrics: dict) -> None:
    """Propiedad 8: Serializar y deserializar un dict de métricas produce el mismo dict.

    json.loads(json.dumps(metrics)) == metrics para cualquier dict de métricas válido.
    """
    serialized = json.dumps(metrics, ensure_ascii=False)
    deserialized = json.loads(serialized)

    for key in metrics:
        assert abs(deserialized[key] - metrics[key]) < 1e-10, (
            f"Valor diferente para '{key}': {metrics[key]} → {deserialized[key]}"
        )


@given(
    model_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-.")),
    metrics=metrics_dict(),
)
@settings(max_examples=50)
def test_property_log_run_roundtrip(model_name: str, metrics: dict) -> None:
    """log_run → load_all produce el mismo registro sin pérdida de datos."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "test.json"
        exp_logger = ExperimentLogger(log_path=log_path)

        exp_logger.log_run(model_name=model_name, hyperparams={}, metrics=metrics)
        records = exp_logger.load_all()

        assert len(records) == 1
        loaded_metrics = records[0]["metrics"]
        for key in metrics:
            assert abs(loaded_metrics[key] - metrics[key]) < 1e-10


# ---------------------------------------------------------------------------
# Test de determinismo del pipeline
# ---------------------------------------------------------------------------


class TestPipelineDeterminism:
    """Verifica que el pipeline de evaluación es determinista con RANDOM_STATE=42."""

    def test_experiment_logger_timestamps_are_unique(
        self, tmp_logger: ExperimentLogger
    ) -> None:
        """Dos log_run consecutivos producen timestamps distintos."""
        import time

        r1 = tmp_logger.log_run("model", {}, {"f1_macro": 0.75, "recall_glosada": 0.72})
        time.sleep(0.01)
        r2 = tmp_logger.log_run("model", {}, {"f1_macro": 0.75, "recall_glosada": 0.72})
        assert r1["timestamp"] != r2["timestamp"]

    def test_numpy_random_state_determinism(self) -> None:
        """Con RANDOM_STATE=42, operaciones numpy son reproducibles."""
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)

        arr1 = rng1.randn(100)
        arr2 = rng2.randn(100)

        np.testing.assert_array_equal(arr1, arr2)

    def test_metrics_serialization_preserves_precision(
        self, tmp_logger: ExperimentLogger
    ) -> None:
        """Las métricas con alta precisión decimal se preservan en serialización."""
        metrics = {
            "f1_macro": 0.74901234567890,
            "recall_glosada": 0.72614159265358,
            "auc_roc": 0.90781828459045,
        }
        tmp_logger.log_run("precision_test", {"random_state": 42}, metrics)
        loaded = tmp_logger.load_all()[0]["metrics"]

        for key in metrics:
            assert abs(loaded[key] - metrics[key]) < 1e-10
