# -*- coding: utf-8 -*-
"""Tests para el módulo de validación de features."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

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

    def test_column_dtypes_are_float(
        self, sample_features: dict, feature_names: list[str]
    ) -> None:
        """Todas las columnas del DataFrame resultante son float."""
        df = prepare_input(sample_features, feature_names)
        for col in df.columns:
            assert np.issubdtype(df[col].dtype, np.floating)

    def test_values_preserved(
        self, realistic_features: dict, feature_names: list[str]
    ) -> None:
        """Los valores del input se preservan en el DataFrame."""
        df = prepare_input(realistic_features, feature_names)
        for name in feature_names:
            assert df[name].iloc[0] == pytest.approx(realistic_features[name])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestValidateFeaturesEdgeCases:
    """Tests de borde para validate_features."""

    def test_empty_dict_raises(self, feature_names: list[str]) -> None:
        """Diccionario vacío lanza ValueError."""
        with pytest.raises(ValueError, match="Faltan"):
            validate_features({}, feature_names)

    def test_bool_values_accepted(
        self, sample_features: dict, feature_names: list[str]
    ) -> None:
        """Valores booleanos son aceptados (bool es subclase de int)."""
        bool_features = {k: True for k in feature_names}
        result = validate_features(bool_features, feature_names)
        assert all(isinstance(v, float) for v in result.values())


# ---------------------------------------------------------------------------
# Property tests (Hypothesis)
# ---------------------------------------------------------------------------


@pytest.mark.property
class TestFeaturesProperties:
    """Property-based tests para el módulo de features."""

    @given(val=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e15, max_value=1e15))
    def test_any_finite_float_valid(self, val: float) -> None:
        """Cualquier float finito es aceptado como valor de feature."""
        feature_names = load_feature_names()
        features = {name: val for name in feature_names}
        result = validate_features(features, feature_names)
        assert len(result) == 48

    @given(n=st.integers(min_value=1, max_value=10))
    def test_batch_size_matches_output(self, n: int) -> None:
        """prepare_batch_input con N dicts produce DataFrame de N filas."""
        feature_names = load_feature_names()
        sample = {name: 0.0 for name in feature_names}
        batch = [sample] * n
        df = prepare_batch_input(batch, feature_names)
        assert df.shape == (n, 48)

    @given(val=st.sampled_from([float("nan"), float("inf"), float("-inf")]))
    def test_nan_inf_always_rejected(self, val: float) -> None:
        """NaN e Inf siempre son rechazados."""
        feature_names = load_feature_names()
        features = {name: 0.0 for name in feature_names}
        features["valor_factura"] = val
        with pytest.raises(ValueError, match="inválidos"):
            validate_features(features, feature_names)
