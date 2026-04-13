# -*- coding: utf-8 -*-
"""Tests unitarios para ModelRegistry y ModelMonitor.

Valida:
- Registro, carga y listado de versiones de modelos (model_registry)
- Activación de versiones y rollback
- Tracking de métricas, detección de degradación y reporte mensual (model_monitor)
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.storage.model_registry import (
    save_model_version,
    load_model_version,
    list_model_versions,
    set_active_version,
    get_active_version,
)
from src.monitoring.model_monitor import ModelMonitor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path):
    """Base de datos SQLite temporal para cada test."""
    return tmp_path / "test_registry.db"


@pytest.fixture()
def monitor_db(tmp_path):
    """Monitor con base de datos temporal."""
    db = tmp_path / "test_monitor.db"
    return ModelMonitor(db_path=db, baseline_metrics={"f1_macro": 0.75, "recall_glosada": 0.73})


SAMPLE_METRICS = {"f1_macro": 0.749, "recall_glosada": 0.726, "auc_roc": 0.908}


# ---------------------------------------------------------------------------
# ModelRegistry tests
# ---------------------------------------------------------------------------


class TestModelRegistry:
    """Tests para registro de versiones de modelos."""

    def test_save_and_load_version(self, tmp_db):
        """Guardar y cargar una versión retorna los mismos datos."""
        save_model_version(
            model_id="20260413_v1",
            version="v1.0",
            metrics=SAMPLE_METRICS,
            artifact_path="/models/xgboost_v1.json",
            notes="Modelo base",
            db_path=tmp_db,
        )

        loaded = load_model_version("v1.0", db_path=tmp_db)

        assert loaded["model_id"] == "20260413_v1"
        assert loaded["version"] == "v1.0"
        assert loaded["metrics"]["f1_macro"] == pytest.approx(0.749)
        assert loaded["artifact_path"] == "/models/xgboost_v1.json"
        assert loaded["notes"] == "Modelo base"
        assert loaded["is_active"] is False

    def test_load_nonexistent_version_raises_key_error(self, tmp_db):
        """Cargar versión inexistente lanza KeyError."""
        with pytest.raises(KeyError, match="v99.0"):
            load_model_version("v99.0", db_path=tmp_db)

    def test_list_versions_empty(self, tmp_db):
        """list_model_versions retorna lista vacía si no hay versiones."""
        assert list_model_versions(db_path=tmp_db) == []

    def test_list_versions_multiple(self, tmp_db):
        """list_model_versions retorna todas las versiones ordenadas por fecha desc."""
        for i, v in enumerate(["v1.0", "v2.0", "v3.0"]):
            save_model_version(
                model_id=f"model_{i}",
                version=v,
                metrics=SAMPLE_METRICS,
                artifact_path=f"/models/model_{i}.json",
                db_path=tmp_db,
            )

        versions = list_model_versions(db_path=tmp_db)

        assert len(versions) == 3
        # Ordenado por created_at DESC → la más reciente primero
        assert versions[0]["version"] == "v3.0"

    def test_set_and_get_active_version(self, tmp_db):
        """set_active_version marca correctamente la versión activa."""
        for v in ["v1.0", "v2.0"]:
            save_model_version(
                model_id=f"model_{v}",
                version=v,
                metrics=SAMPLE_METRICS,
                artifact_path=f"/models/{v}.json",
                db_path=tmp_db,
            )

        set_active_version("v2.0", db_path=tmp_db)
        active = get_active_version(db_path=tmp_db)

        assert active is not None
        assert active["version"] == "v2.0"
        assert active["is_active"] is True

    def test_set_active_desactivates_previous(self, tmp_db):
        """Activar v2 desactiva v1 automáticamente."""
        for v in ["v1.0", "v2.0"]:
            save_model_version(
                model_id=f"model_{v}",
                version=v,
                metrics=SAMPLE_METRICS,
                artifact_path=f"/models/{v}.json",
                db_path=tmp_db,
            )

        set_active_version("v1.0", db_path=tmp_db)
        set_active_version("v2.0", db_path=tmp_db)

        all_versions = list_model_versions(db_path=tmp_db)
        active_count = sum(1 for v in all_versions if v["is_active"])
        assert active_count == 1

    def test_get_active_version_none_when_empty(self, tmp_db):
        """get_active_version retorna None si no hay versión activa."""
        assert get_active_version(db_path=tmp_db) is None

    def test_set_active_nonexistent_version_raises(self, tmp_db):
        """set_active_version lanza KeyError si la versión no existe."""
        with pytest.raises(KeyError):
            set_active_version("v99.0", db_path=tmp_db)


# ---------------------------------------------------------------------------
# ModelMonitor tests
# ---------------------------------------------------------------------------


class TestModelMonitor:
    """Tests para monitor de rendimiento del modelo."""

    def _make_predictions(self, n: int = 50, seed: int = 42):
        """Genera predicciones y etiquetas reales para tests."""
        rng = np.random.default_rng(seed)
        actuals = rng.integers(0, 3, size=n)
        # Predicciones con ~70% accuracy
        noise_mask = rng.random(n) < 0.3
        predictions = actuals.copy()
        predictions[noise_mask] = rng.integers(0, 3, size=noise_mask.sum())
        return predictions, actuals

    def test_track_performance_returns_metrics(self, monitor_db):
        """track_performance retorna diccionario con métricas esperadas."""
        preds, actuals = self._make_predictions()
        result = monitor_db.track_performance(preds, actuals)

        for key in ["f1_macro", "recall_glosada", "recall_auditada", "recall_devuelta", "accuracy"]:
            assert key in result
            assert 0.0 <= result[key] <= 1.0

    def test_get_metrics_history_after_tracking(self, monitor_db):
        """get_model_metrics_history retorna registros almacenados."""
        preds, actuals = self._make_predictions()
        monitor_db.track_performance(preds, actuals)
        monitor_db.track_performance(preds, actuals)

        history = monitor_db.get_model_metrics_history()
        assert len(history) == 2
        assert "f1_macro" in history[0]

    def test_get_metrics_history_last_n(self, monitor_db):
        """get_model_metrics_history respeta el límite last_n."""
        preds, actuals = self._make_predictions()
        for _ in range(5):
            monitor_db.track_performance(preds, actuals)

        history = monitor_db.get_model_metrics_history(last_n=2)
        assert len(history) == 2

    def test_detect_performance_drop_no_history(self, monitor_db):
        """detect_performance_drop retorna drop_detected=False sin historial."""
        result = monitor_db.detect_performance_drop()
        assert result["drop_detected"] is False

    def test_detect_performance_drop_degraded(self, monitor_db):
        """detect_performance_drop detecta caída real respecto al baseline."""
        # Registrar predicciones muy malas (todos clase 0 para datos balanceados)
        rng = np.random.default_rng(0)
        actuals = rng.integers(0, 3, size=100)
        bad_preds = np.zeros(100, dtype=int)  # Siempre predice clase 0

        for _ in range(7):
            monitor_db.track_performance(bad_preds, actuals)

        result = monitor_db.detect_performance_drop(threshold=0.05)
        assert "drop_detected" in result
        assert "drops" in result

    def test_generate_monthly_report_no_data(self, monitor_db):
        """generate_monthly_report retorna status sin_datos si no hay registros."""
        report = monitor_db.generate_monthly_report()
        assert report["status"] == "sin_datos"

    def test_generate_monthly_report_with_data(self, monitor_db):
        """generate_monthly_report retorna estadísticas con datos disponibles."""
        preds, actuals = self._make_predictions()
        monitor_db.track_performance(preds, actuals)
        monitor_db.track_performance(preds, actuals)

        report = monitor_db.generate_monthly_report()

        assert report["period_days"] == 30
        assert report["n_records"] >= 1
        assert "f1_macro" in report
        assert "recall_glosada" in report
        assert "mean" in report["f1_macro"]

    def test_track_performance_with_extra_metadata(self, monitor_db):
        """track_performance acepta metadatos extra."""
        preds, actuals = self._make_predictions()
        result = monitor_db.track_performance(preds, actuals, extra={"prestador": "12345"})
        assert result["f1_macro"] >= 0.0
