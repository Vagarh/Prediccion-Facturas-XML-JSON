# -*- coding: utf-8 -*-
"""Tests de regresión del modelo contra dataset de prueba."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from src.config import MODEL_PATH
from src.features.engineering import load_feature_names
from src.models.predict import load_model, predict_batch

DATASET_PATH = Path(__file__).parent / "fixtures" / "test_dataset_100.parquet"

pytestmark = [
    pytest.mark.regression,
    pytest.mark.skipif(not MODEL_PATH.exists(), reason=f"Modelo no encontrado en {MODEL_PATH}"),
    pytest.mark.skipif(not DATASET_PATH.exists(), reason=f"Dataset de test no encontrado en {DATASET_PATH}"),
]


@pytest.fixture(scope="module")
def test_dataset() -> pd.DataFrame:
    """Carga el dataset de 100 muestras estratificadas."""
    return pd.read_parquet(DATASET_PATH)


@pytest.fixture(scope="module")
def predictions(test_dataset: pd.DataFrame):
    """Ejecuta predicciones sobre el dataset de test."""
    feature_names = load_feature_names()
    model = load_model()

    features_df = test_dataset[feature_names]
    classes, proba = predict_batch(model, features_df)
    y_true = test_dataset["target"].values

    return y_true, classes, proba


# ---------------------------------------------------------------------------
# Tests de integridad del dataset
# ---------------------------------------------------------------------------


class TestDatasetIntegrity:
    """Verifica la integridad del dataset de prueba."""

    def test_dataset_has_100_samples(self, test_dataset: pd.DataFrame) -> None:
        """El dataset tiene exactamente 100 muestras."""
        assert len(test_dataset) == 100

    def test_dataset_class_distribution(self, test_dataset: pd.DataFrame) -> None:
        """Distribución de clases: 33/33/34."""
        counts = test_dataset["target"].value_counts().sort_index()
        assert counts[0] == 33
        assert counts[1] == 33
        assert counts[2] == 34


# ---------------------------------------------------------------------------
# Tests de métricas del modelo
# ---------------------------------------------------------------------------


class TestModelMetrics:
    """Verifica que las métricas del modelo están por encima de los umbrales."""

    def test_f1_macro_above_threshold(self, predictions) -> None:
        """F1-macro >= 0.65 en el dataset de 100 muestras."""
        y_true, y_pred, _ = predictions
        f1 = f1_score(y_true, y_pred, average="macro")
        assert f1 >= 0.65, f"F1-macro: {f1:.4f} (mín: 0.65)"

    def test_recall_glosada_above_threshold(self, predictions) -> None:
        """Recall de clase Glosada (1) >= 0.60 en dataset de 100 muestras."""
        y_true, y_pred, _ = predictions
        recall = recall_score(y_true, y_pred, labels=[1], average=None)[0]
        assert recall >= 0.60, f"Recall Glosada: {recall:.4f} (mín: 0.60)"

    def test_auc_roc_above_threshold(self, predictions) -> None:
        """AUC-ROC >= 0.85 en dataset de 100 muestras."""
        y_true, _, y_proba = predictions
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        assert auc >= 0.85, f"AUC-ROC: {auc:.4f} (mín: 0.85)"

    def test_all_classes_predicted(self, predictions) -> None:
        """El modelo predice las 3 clases en el dataset de 100 muestras."""
        _, y_pred, _ = predictions
        unique_classes = set(y_pred)
        assert unique_classes == {0, 1, 2}, f"Clases predichas: {unique_classes}"

    def test_accuracy_above_threshold(self, predictions) -> None:
        """Accuracy >= 0.60 en dataset de 100 muestras."""
        y_true, y_pred, _ = predictions
        acc = accuracy_score(y_true, y_pred)
        assert acc >= 0.60, f"Accuracy: {acc:.4f} (mín: 0.60)"


# ---------------------------------------------------------------------------
# Tests de estabilidad del modelo
# ---------------------------------------------------------------------------


class TestModelStability:
    """Verifica la estabilidad y consistencia del modelo."""

    def test_deterministic_on_same_input(self, test_dataset: pd.DataFrame) -> None:
        """Dos ejecuciones con el mismo input producen resultados idénticos."""
        feature_names = load_feature_names()
        model = load_model()
        features_df = test_dataset[feature_names]

        c1, p1 = predict_batch(model, features_df)
        c2, p2 = predict_batch(model, features_df)

        assert np.array_equal(c1, c2)
        assert np.array_equal(p1, p2)

    def test_probability_distribution_reasonable(self, predictions) -> None:
        """Las probabilidades no son todas idénticas (hay variabilidad)."""
        _, _, y_proba = predictions
        for class_idx in range(3):
            std = y_proba[:, class_idx].std()
            assert std > 0.01, (
                f"Clase {class_idx}: desviación estándar muy baja ({std:.6f}), "
                "modelo posiblemente degenerado"
            )

    def test_no_nan_in_predictions(self, predictions) -> None:
        """No hay NaN en predicciones ni probabilidades."""
        y_true, y_pred, y_proba = predictions
        assert not np.any(np.isnan(y_pred))
        assert not np.any(np.isnan(y_proba))
