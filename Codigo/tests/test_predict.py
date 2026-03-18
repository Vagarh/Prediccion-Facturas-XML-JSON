# -*- coding: utf-8 -*-
"""Tests para el módulo de carga de modelo y predicción."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.config import MODEL_PATH
from src.features.engineering import load_feature_names, prepare_batch_input, prepare_input
from src.models.predict import load_model, predict_batch, predict_single
from tests.fixtures.mock_models import make_mock_model, make_mock_model_batch
from tests.fixtures.sample_data import make_realistic_features


@pytest.fixture(scope="module")
def model():
    """Carga el modelo XGBoost una sola vez para el módulo."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Modelo no encontrado en {MODEL_PATH}")
    return load_model()


@pytest.mark.skipif(not MODEL_PATH.exists(), reason=f"Modelo no encontrado en {MODEL_PATH}")
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


@pytest.mark.skipif(not MODEL_PATH.exists(), reason=f"Modelo no encontrado en {MODEL_PATH}")
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


@pytest.mark.skipif(not MODEL_PATH.exists(), reason=f"Modelo no encontrado en {MODEL_PATH}")
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


# ---------------------------------------------------------------------------
# Unit tests con mocks (no requieren modelo real)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPredictSingleMocked:
    """Tests unitarios de predict_single con mock model."""

    def test_mock_returns_valid_class(self) -> None:
        """Mock model retorna clase válida via predict_single."""
        feature_names = load_feature_names()
        features = make_realistic_features()
        df = pd.DataFrame([features], columns=feature_names)
        mock_model = make_mock_model()

        pred_class, proba = predict_single(mock_model, df)
        assert pred_class in {0, 1, 2}

    def test_mock_proba_shape(self) -> None:
        """Mock model retorna probabilidades con shape (3,)."""
        feature_names = load_feature_names()
        features = make_realistic_features()
        df = pd.DataFrame([features], columns=feature_names)
        mock_model = make_mock_model()

        _, proba = predict_single(mock_model, df)
        assert proba.shape == (3,)


@pytest.mark.unit
class TestPredictBatchMocked:
    """Tests unitarios de predict_batch con mock model."""

    def test_mock_batch_shapes(self) -> None:
        """Mock batch retorna shapes correctos."""
        feature_names = load_feature_names()
        features = make_realistic_features()
        n = 5
        df = pd.DataFrame([features] * n, columns=feature_names)
        mock_model = make_mock_model_batch(n)

        classes, proba_matrix = predict_batch(mock_model, df)
        assert classes.shape == (n,)
        assert proba_matrix.shape == (n, 3)

    def test_mock_batch_classes_valid(self) -> None:
        """Todas las clases del mock batch son válidas."""
        feature_names = load_feature_names()
        features = make_realistic_features()
        n = 3
        df = pd.DataFrame([features] * n, columns=feature_names)
        mock_model = make_mock_model_batch(n)

        classes, _ = predict_batch(mock_model, df)
        assert all(c in {0, 1, 2} for c in classes)


# ---------------------------------------------------------------------------
# Property tests (requieren modelo real)
# ---------------------------------------------------------------------------


@pytest.mark.property
@pytest.mark.skipif(not MODEL_PATH.exists(), reason=f"Modelo no encontrado en {MODEL_PATH}")
class TestPredictProperties:
    """Property-based tests para predicción."""

    @given(n=st.integers(min_value=1, max_value=10))
    def test_batch_output_length_matches_input(self, n: int) -> None:
        """predict_batch con N inputs produce N outputs."""
        model = load_model()
        feature_names = load_feature_names()
        features = make_realistic_features()
        df = pd.DataFrame([features] * n, columns=feature_names)

        classes, proba = predict_batch(model, df)
        assert len(classes) == n
        assert proba.shape[0] == n

    def test_deterministic_predictions(self) -> None:
        """Mismo input produce mismo output."""
        model = load_model()
        feature_names = load_feature_names()
        features = make_realistic_features()
        df = pd.DataFrame([features], columns=feature_names)

        c1, p1 = predict_single(model, df)
        c2, p2 = predict_single(model, df)
        assert c1 == c2
        assert np.array_equal(p1, p2)
