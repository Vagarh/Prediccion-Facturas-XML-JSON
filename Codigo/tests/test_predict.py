# -*- coding: utf-8 -*-
"""Tests para el módulo de carga de modelo y predicción."""

from pathlib import Path

import numpy as np
import pytest

from src.config import MODEL_PATH
from src.features.engineering import prepare_batch_input, prepare_input
from src.models.predict import load_model, predict_batch, predict_single

# Saltar todos los tests si el modelo no existe en disco
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"Modelo no encontrado en {MODEL_PATH}",
)


@pytest.fixture(scope="module")
def model():
    """Carga el modelo XGBoost una sola vez para el módulo."""
    return load_model()


class TestLoadModel:
    """Tests para la carga del modelo."""

    def test_model_loads_successfully(self, model) -> None:
        """El modelo se carga sin errores."""
        assert model is not None

    def test_model_has_predict_proba(self, model) -> None:
        """El modelo tiene el método predict_proba."""
        assert hasattr(model, "predict_proba")

    def test_model_file_not_found_raises(self) -> None:
        """Ruta inexistente lanza FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_model(Path("/nonexistent/model.json"))


class TestPredictSingle:
    """Tests para predicción individual."""

    def test_returns_valid_class(
        self, model, sample_features: dict, feature_names: list[str]
    ) -> None:
        """La clase predicha está en {0, 1, 2}."""
        df = prepare_input(sample_features, feature_names)
        pred_class, _ = predict_single(model, df)
        assert pred_class in {0, 1, 2}

    def test_probabilities_sum_to_one(
        self, model, sample_features: dict, feature_names: list[str]
    ) -> None:
        """Las probabilidades suman aproximadamente 1.0."""
        df = prepare_input(sample_features, feature_names)
        _, proba = predict_single(model, df)
        assert proba.shape == (3,)
        assert abs(proba.sum() - 1.0) < 1e-6

    def test_realistic_features_predict(
        self, model, realistic_features: dict, feature_names: list[str]
    ) -> None:
        """Features realistas producen una predicción válida."""
        df = prepare_input(realistic_features, feature_names)
        pred_class, proba = predict_single(model, df)
        assert pred_class in {0, 1, 2}
        assert all(0 <= p <= 1 for p in proba)


class TestPredictBatch:
    """Tests para predicción batch."""

    def test_batch_shapes(
        self, model, sample_features: dict, feature_names: list[str]
    ) -> None:
        """Las salidas batch tienen formas correctas (N,) y (N, 3)."""
        batch = [sample_features] * 5
        df = prepare_batch_input(batch, feature_names)
        classes, proba_matrix = predict_batch(model, df)
        assert classes.shape == (5,)
        assert proba_matrix.shape == (5, 3)

    def test_batch_all_valid_classes(
        self, model, sample_features: dict, feature_names: list[str]
    ) -> None:
        """Todas las clases en batch están en {0, 1, 2}."""
        batch = [sample_features] * 3
        df = prepare_batch_input(batch, feature_names)
        classes, _ = predict_batch(model, df)
        assert all(c in {0, 1, 2} for c in classes)

    def test_batch_probabilities_sum(
        self, model, sample_features: dict, feature_names: list[str]
    ) -> None:
        """Cada fila de probabilidades suma ~1.0."""
        batch = [sample_features] * 3
        df = prepare_batch_input(batch, feature_names)
        _, proba_matrix = predict_batch(model, df)
        row_sums = proba_matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)
