# -*- coding: utf-8 -*-
"""Tests para el módulo de explicabilidad SHAP."""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    CLASS_ACTIONS,
    CLASS_NAMES,
    FEATURE_LABELS,
    FEATURE_TO_CAUSAL,
    MODEL_PATH,
    RISK_DESCRIPTIONS,
)
from src.explainability.shap_explanations import (
    _format_risk_factor,
    explain_batch_predictions,
    explain_single_prediction,
)
from tests.fixtures.mock_models import (
    make_mock_explainer,
    make_mock_explainer_batch,
    make_mock_model,
    make_mock_model_batch,
)
from tests.fixtures.sample_data import make_realistic_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_features_df(feature_names: list[str]) -> pd.DataFrame:
    """Construye un DataFrame (1, 48) con features realistas."""
    features = make_realistic_features()
    return pd.DataFrame([features], columns=feature_names)


def _build_batch_df(feature_names: list[str], n: int = 3) -> pd.DataFrame:
    """Construye un DataFrame (N, 48) con features realistas."""
    features = make_realistic_features()
    return pd.DataFrame([features] * n, columns=feature_names)


# ---------------------------------------------------------------------------
# Unit tests — explain_single_prediction (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExplainSinglePrediction:
    """Tests unitarios para explain_single_prediction con mocks."""

    def test_returns_expected_keys(self, feature_names: list[str]) -> None:
        """El resultado contiene todas las claves esperadas."""
        df = _build_features_df(feature_names)
        mock_model = make_mock_model()
        mock_explainer = make_mock_explainer()

        result = explain_single_prediction(
            df, mock_model, mock_explainer, feature_names
        )

        expected_keys = {
            "estado_predicho",
            "clase",
            "confianza",
            "probabilidades",
            "factores_riesgo",
            "factores_detalle",
            "causal_probable",
            "accion_sugerida",
        }
        assert set(result.keys()) == expected_keys

    def test_estado_predicho_matches_class(self, feature_names: list[str]) -> None:
        """estado_predicho corresponde al nombre de la clase predicha."""
        df = _build_features_df(feature_names)
        mock_model = make_mock_model()  # predice clase 1 (Glosada)
        mock_explainer = make_mock_explainer()

        result = explain_single_prediction(
            df, mock_model, mock_explainer, feature_names
        )

        assert result["estado_predicho"] == CLASS_NAMES[result["clase"]].lower()

    def test_confianza_matches_predicted_class_probability(
        self, feature_names: list[str]
    ) -> None:
        """confianza coincide con la probabilidad de la clase predicha."""
        df = _build_features_df(feature_names)
        mock_model = make_mock_model()  # proba [0.2, 0.6, 0.2], clase 1
        mock_explainer = make_mock_explainer()

        result = explain_single_prediction(
            df, mock_model, mock_explainer, feature_names
        )

        estado = result["estado_predicho"]
        assert abs(result["confianza"] - result["probabilidades"][estado]) < 1e-6

    def test_probabilities_sum_to_one(self, feature_names: list[str]) -> None:
        """Las probabilidades suman ~1.0."""
        df = _build_features_df(feature_names)
        result = explain_single_prediction(
            df, make_mock_model(), make_mock_explainer(), feature_names
        )

        probs = result["probabilidades"]
        total = probs["auditada"] + probs["glosada"] + probs["devuelta"]
        assert abs(total - 1.0) < 1e-4

    def test_top_n_risk_factors(self, feature_names: list[str]) -> None:
        """Se retornan exactamente top_n factores de riesgo."""
        df = _build_features_df(feature_names)
        top_n = 3
        result = explain_single_prediction(
            df, make_mock_model(), make_mock_explainer(), feature_names, top_n=top_n
        )

        assert len(result["factores_riesgo"]) == top_n
        assert len(result["factores_detalle"]) == top_n

    def test_risk_factors_have_required_fields(
        self, feature_names: list[str]
    ) -> None:
        """Cada factor de riesgo tiene los campos requeridos."""
        df = _build_features_df(feature_names)
        result = explain_single_prediction(
            df, make_mock_model(), make_mock_explainer(), feature_names
        )

        for factor in result["factores_detalle"]:
            assert "feature" in factor
            assert "shap_value" in factor
            assert "feature_value" in factor
            assert "descripcion" in factor
            assert isinstance(factor["shap_value"], float)
            assert isinstance(factor["feature_value"], float)
            assert isinstance(factor["descripcion"], str)

    def test_accion_sugerida_matches_class(self, feature_names: list[str]) -> None:
        """La acción sugerida corresponde a la clase predicha."""
        df = _build_features_df(feature_names)
        result = explain_single_prediction(
            df, make_mock_model(), make_mock_explainer(), feature_names
        )

        assert result["accion_sugerida"] == CLASS_ACTIONS[result["clase"]]


# ---------------------------------------------------------------------------
# Unit tests — explain_batch_predictions (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExplainBatchPredictions:
    """Tests unitarios para explain_batch_predictions con mocks."""

    def test_returns_correct_length(self, feature_names: list[str]) -> None:
        """El batch retorna el número correcto de explicaciones."""
        n = 5
        df = _build_batch_df(feature_names, n=n)
        mock_model = make_mock_model_batch(n)
        mock_explainer = make_mock_explainer_batch(n)

        results = explain_batch_predictions(
            df, mock_model, mock_explainer, feature_names
        )

        assert len(results) == n

    def test_each_result_has_expected_keys(self, feature_names: list[str]) -> None:
        """Cada resultado del batch tiene las claves esperadas."""
        n = 3
        df = _build_batch_df(feature_names, n=n)
        results = explain_batch_predictions(
            df, make_mock_model_batch(n), make_mock_explainer_batch(n), feature_names
        )

        expected_keys = {
            "estado_predicho",
            "clase",
            "confianza",
            "probabilidades",
            "factores_riesgo",
            "factores_detalle",
            "causal_probable",
            "accion_sugerida",
        }
        for result in results:
            assert set(result.keys()) == expected_keys

    def test_all_classes_valid(self, feature_names: list[str]) -> None:
        """Todas las clases del batch son válidas."""
        n = 5
        df = _build_batch_df(feature_names, n=n)
        results = explain_batch_predictions(
            df, make_mock_model_batch(n), make_mock_explainer_batch(n), feature_names
        )

        for result in results:
            assert result["clase"] in {0, 1, 2}


# ---------------------------------------------------------------------------
# Unit tests — _format_risk_factor
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormatRiskFactor:
    """Tests unitarios para _format_risk_factor."""

    def test_known_feature_positive_shap(self) -> None:
        """Feature conocida con SHAP positivo y is_high_risky=True usa text_high."""
        # tasa_glosa_nit: is_high_risky=True
        result = _format_risk_factor("tasa_glosa_nit", shap_value=0.5, feature_value=0.25)
        assert "25.0%" in result or "alta" in result.lower()

    def test_known_feature_negative_shap(self) -> None:
        """Feature conocida con SHAP negativo y is_high_risky=True usa text_low."""
        result = _format_risk_factor("tasa_glosa_nit", shap_value=-0.3, feature_value=0.05)
        assert "baja" in result.lower()

    def test_known_feature_not_high_risky_positive_shap(self) -> None:
        """Feature con is_high_risky=False y SHAP positivo usa text_low."""
        # ratio_autorizacion: is_high_risky=False
        result = _format_risk_factor("ratio_autorizacion", shap_value=0.3, feature_value=0.1)
        # Para is_high_risky=False, shap>0 -> text_low (which says "Alto ratio")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unknown_feature_fallback(self) -> None:
        """Feature no mapeada usa el formato genérico con label."""
        result = _format_risk_factor("log_num_items", shap_value=0.2, feature_value=1.5)
        assert "1.50" in result
        assert "alto" in result.lower()

    def test_unknown_feature_negative_shap(self) -> None:
        """Feature no mapeada con SHAP negativo indica impacto bajo."""
        result = _format_risk_factor("log_num_items", shap_value=-0.2, feature_value=1.5)
        assert "bajo" in result.lower()

    def test_format_with_val_placeholder(self) -> None:
        """El placeholder {val} se reemplaza correctamente."""
        result = _format_risk_factor("valor_factura", shap_value=0.5, feature_value=500000.0)
        assert "500,000" in result or "500000" in result

    def test_format_without_val_placeholder(self) -> None:
        """Texto sin {val} se retorna sin error."""
        # is_extemporanea text_high no tiene {val} en la parte sin formato
        result = _format_risk_factor("is_extemporanea", shap_value=0.5, feature_value=1.0)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Integration tests — con modelo real
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not MODEL_PATH.exists(), reason=f"Modelo no encontrado en {MODEL_PATH}")
class TestExplainSingleIntegration:
    """Tests de integración con modelo y SHAP explainer reales."""

    def test_real_model_returns_valid_explanation(
        self, model, explainer, feature_names: list[str]
    ) -> None:
        """Explicación con modelo real tiene estructura válida."""
        df = _build_features_df(feature_names)
        result = explain_single_prediction(df, model, explainer, feature_names)

        assert result["clase"] in {0, 1, 2}
        assert 0 <= result["confianza"] <= 1
        assert len(result["factores_riesgo"]) == 5

    def test_real_model_probabilities_sum(
        self, model, explainer, feature_names: list[str]
    ) -> None:
        """Probabilidades del modelo real suman ~1.0."""
        df = _build_features_df(feature_names)
        result = explain_single_prediction(df, model, explainer, feature_names)

        probs = result["probabilidades"]
        total = probs["auditada"] + probs["glosada"] + probs["devuelta"]
        assert abs(total - 1.0) < 1e-4

    def test_real_model_deterministic(
        self, model, explainer, feature_names: list[str]
    ) -> None:
        """Dos ejecuciones con el mismo input producen el mismo resultado."""
        df = _build_features_df(feature_names)
        r1 = explain_single_prediction(df, model, explainer, feature_names)
        r2 = explain_single_prediction(df, model, explainer, feature_names)

        assert r1["clase"] == r2["clase"]
        assert r1["confianza"] == r2["confianza"]
