# -*- coding: utf-8 -*-
"""Tests para el módulo de validación de features."""

import pytest

from src.features.engineering import (
    load_feature_names,
    prepare_batch_input,
    prepare_input,
    validate_features,
)


class TestLoadFeatureNames:
    """Tests para la carga de lista de features."""

    def test_load_returns_48_features(self, feature_names: list[str]) -> None:
        """Verifica que se cargan exactamente 48 features."""
        assert len(feature_names) == 48

    def test_features_are_strings(self, feature_names: list[str]) -> None:
        """Verifica que todos los nombres son strings no vacíos."""
        for name in feature_names:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_features_are_sorted_alphabetically(self, feature_names: list[str]) -> None:
        """Verifica que las features están en orden alfabético."""
        assert feature_names == sorted(feature_names)

    def test_known_features_present(self, feature_names: list[str]) -> None:
        """Verifica que features conocidas están presentes."""
        expected = ["valor_factura", "tasa_glosa_nit", "num_items", "is_extemporanea"]
        for feat in expected:
            assert feat in feature_names

    def test_load_is_cached(self) -> None:
        """Verifica que llamadas sucesivas retornan el mismo objeto (lru_cache)."""
        result1 = load_feature_names()
        result2 = load_feature_names()
        assert result1 is result2


class TestValidateFeatures:
    """Tests para la validación de diccionarios de features."""

    def test_valid_features_pass(
        self, sample_features: dict, feature_names: list[str]
    ) -> None:
        """Un diccionario con las 48 features válidas no lanza error."""
        result = validate_features(sample_features, feature_names)
        assert len(result) == 48

    def test_missing_features_raises(self, feature_names: list[str]) -> None:
        """Diccionario incompleto lanza ValueError."""
        incomplete = {"valor_factura": 1.0}
        with pytest.raises(ValueError, match="Faltan"):
            validate_features(incomplete, feature_names)

    def test_nan_value_raises(
        self, sample_features: dict, feature_names: list[str]
    ) -> None:
        """Valor NaN lanza ValueError."""
        bad = {**sample_features, "valor_factura": float("nan")}
        with pytest.raises(ValueError, match="inválidos"):
            validate_features(bad, feature_names)

    def test_inf_value_raises(
        self, sample_features: dict, feature_names: list[str]
    ) -> None:
        """Valor Inf lanza ValueError."""
        bad = {**sample_features, "valor_factura": float("inf")}
        with pytest.raises(ValueError, match="inválidos"):
            validate_features(bad, feature_names)

    def test_string_value_raises(
        self, sample_features: dict, feature_names: list[str]
    ) -> None:
        """Valor string lanza ValueError."""
        bad = {**sample_features, "valor_factura": "not_a_number"}
        with pytest.raises(ValueError, match="inválidos"):
            validate_features(bad, feature_names)

    def test_int_values_accepted(
        self, sample_features: dict, feature_names: list[str]
    ) -> None:
        """Valores enteros son aceptados y convertidos a float."""
        int_features = {k: 1 for k in feature_names}
        result = validate_features(int_features, feature_names)
        assert all(isinstance(v, float) for v in result.values())


class TestPrepareInput:
    """Tests para la preparación de DataFrames de entrada."""

    def test_single_input_shape(
        self, sample_features: dict, feature_names: list[str]
    ) -> None:
        """DataFrame single tiene forma (1, 48)."""
        df = prepare_input(sample_features, feature_names)
        assert df.shape == (1, 48)

    def test_single_input_column_order(
        self, sample_features: dict, feature_names: list[str]
    ) -> None:
        """Las columnas están en el orden exacto del modelo."""
        df = prepare_input(sample_features, feature_names)
        assert list(df.columns) == feature_names

    def test_batch_input_shape(
        self, sample_features: dict, feature_names: list[str]
    ) -> None:
        """DataFrame batch tiene forma (N, 48)."""
        batch = [sample_features, sample_features, sample_features]
        df = prepare_batch_input(batch, feature_names)
        assert df.shape == (3, 48)

    def test_batch_invalid_raises_with_index(
        self, sample_features: dict, feature_names: list[str]
    ) -> None:
        """Error en batch indica el índice de la factura problemática."""
        bad = {"valor_factura": 1.0}  # Incompleto
        batch = [sample_features, bad]
        with pytest.raises(ValueError, match="índice 1"):
            prepare_batch_input(batch, feature_names)
