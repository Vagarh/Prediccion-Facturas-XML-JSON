# -*- coding: utf-8 -*-
"""Tests para los schemas Pydantic v2 de la API."""

from datetime import datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeatureInput,
    HealthResponse,
    ModelInfoResponse,
    Probabilities,
    PredictionRequest,
    PredictionResponse,
    RiskFactor,
)
from src.features.engineering import load_feature_names
from tests.fixtures.sample_data import make_realistic_features


# ---------------------------------------------------------------------------
# TestFeatureInput
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFeatureInput:
    """Tests para el schema FeatureInput (48 features)."""

    def test_valid_features_accepted(self) -> None:
        """Features realistas son aceptadas sin error."""
        features = make_realistic_features()
        fi = FeatureInput(**features)
        assert fi.valor_factura == 150000.0

    def test_exactly_48_fields(self) -> None:
        """FeatureInput tiene exactamente 48 campos."""
        assert len(FeatureInput.model_fields) == 48

    def test_extra_field_rejected(self) -> None:
        """Campo extra es rechazado (extra='forbid')."""
        features = make_realistic_features()
        features["campo_extra"] = 999.0
        with pytest.raises(ValidationError, match="extra"):
            FeatureInput(**features)

    def test_missing_field_rejected(self) -> None:
        """Campo faltante es rechazado."""
        features = make_realistic_features()
        del features["valor_factura"]
        with pytest.raises(ValidationError):
            FeatureInput(**features)

    def test_string_numeric_coerced(self) -> None:
        """String numérico se convierte a float."""
        features = make_realistic_features()
        features["valor_factura"] = "100000.0"
        fi = FeatureInput(**features)
        assert fi.valor_factura == 100000.0

    def test_non_numeric_string_rejected(self) -> None:
        """String no numérico es rechazado."""
        features = make_realistic_features()
        features["valor_factura"] = "abc"
        with pytest.raises(ValidationError):
            FeatureInput(**features)

    def test_none_value_rejected(self) -> None:
        """None es rechazado para un campo float."""
        features = make_realistic_features()
        features["valor_factura"] = None
        with pytest.raises(ValidationError):
            FeatureInput(**features)

    def test_int_value_accepted(self) -> None:
        """Valores enteros son aceptados."""
        features = make_realistic_features()
        features["valor_factura"] = 150000
        fi = FeatureInput(**features)
        assert isinstance(fi.valor_factura, float)

    def test_field_names_match_feature_list(self) -> None:
        """Los nombres de los campos coinciden con la lista de features."""
        schema_fields = set(FeatureInput.model_fields.keys())
        feature_list = set(load_feature_names())
        assert schema_fields == feature_list

    def test_model_dump_preserves_values(self) -> None:
        """model_dump() preserva los valores correctamente."""
        features = make_realistic_features()
        fi = FeatureInput(**features)
        dumped = fi.model_dump()
        assert dumped["valor_factura"] == 150000.0
        assert len(dumped) == 48


# ---------------------------------------------------------------------------
# TestPredictionRequest
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPredictionRequest:
    """Tests para el schema PredictionRequest."""

    def test_valid_request(self) -> None:
        """Request válido con numero_factura y features."""
        features = make_realistic_features()
        req = PredictionRequest(
            numero_factura="FEV-2024-001",
            features=FeatureInput(**features),
        )
        assert req.numero_factura == "FEV-2024-001"

    def test_missing_numero_factura(self) -> None:
        """Falta numero_factura lanza ValidationError."""
        features = make_realistic_features()
        with pytest.raises(ValidationError):
            PredictionRequest(features=FeatureInput(**features))

    def test_missing_features(self) -> None:
        """Falta features lanza ValidationError."""
        with pytest.raises(ValidationError):
            PredictionRequest(numero_factura="FEV-001")


# ---------------------------------------------------------------------------
# TestBatchPredictionRequest
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBatchPredictionRequest:
    """Tests para el schema BatchPredictionRequest."""

    def test_valid_batch(self) -> None:
        """Batch con lista de PredictionRequests es válido."""
        features = make_realistic_features()
        req = BatchPredictionRequest(
            facturas=[
                PredictionRequest(
                    numero_factura=f"BATCH-{i}",
                    features=FeatureInput(**features),
                )
                for i in range(3)
            ]
        )
        assert len(req.facturas) == 3

    def test_max_length_1000(self) -> None:
        """Lista con >1000 facturas es rechazada."""
        features = make_realistic_features()
        with pytest.raises(ValidationError, match="1000"):
            BatchPredictionRequest(
                facturas=[
                    PredictionRequest(
                        numero_factura=f"FAC-{i}",
                        features=FeatureInput(**features),
                    )
                    for i in range(1001)
                ]
            )

    def test_empty_list_accepted(self) -> None:
        """Lista vacía es aceptada."""
        req = BatchPredictionRequest(facturas=[])
        assert len(req.facturas) == 0


# ---------------------------------------------------------------------------
# TestPredictionResponse
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPredictionResponse:
    """Tests para el schema PredictionResponse."""

    def test_all_response_fields(self) -> None:
        """PredictionResponse se construye con todos los campos."""
        resp = PredictionResponse(
            numero_factura="TEST-001",
            estado_predicho="glosada",
            clase=1,
            confianza=0.6,
            probabilidades=Probabilities(auditada=0.2, glosada=0.6, devuelta=0.2),
            factores_riesgo=["Factor 1", "Factor 2"],
            factores_detalle=[
                RiskFactor(
                    feature="tasa_glosa_nit",
                    shap_value=0.15,
                    feature_value=0.25,
                    descripcion="Tasa de glosa alta",
                )
            ],
            accion_sugerida="Revisar",
            timestamp_prediccion=datetime.now(),
        )
        assert resp.clase == 1
        assert resp.causal_probable is None

    def test_optional_causal_probable(self) -> None:
        """causal_probable puede ser None o un string."""
        resp = PredictionResponse(
            numero_factura="TEST-002",
            estado_predicho="glosada",
            clase=1,
            confianza=0.6,
            probabilidades=Probabilities(auditada=0.2, glosada=0.6, devuelta=0.2),
            factores_riesgo=[],
            accion_sugerida="Revisar",
            timestamp_prediccion=datetime.now(),
            causal_probable="DE16",
        )
        assert resp.causal_probable == "DE16"


# ---------------------------------------------------------------------------
# TestHealthResponse & TestModelInfoResponse
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHealthResponse:
    """Tests para el schema HealthResponse."""

    def test_health_response_fields(self) -> None:
        """HealthResponse se construye correctamente."""
        resp = HealthResponse(
            status="ok",
            modelo_cargado=True,
            version="v1.0.0",
            timestamp=datetime.now(),
        )
        assert resp.status == "ok"
        assert resp.modelo_cargado is True


@pytest.mark.unit
class TestModelInfoResponse:
    """Tests para el schema ModelInfoResponse."""

    def test_model_info_response_fields(self) -> None:
        """ModelInfoResponse se construye correctamente."""
        resp = ModelInfoResponse(
            modelo="XGBoost",
            version="v1",
            fecha_entrenamiento="2026-03-09",
            metricas={"f1_macro": 0.749},
            n_features=48,
            features=["feat_1"] * 48,
            clases={"0": "Auditada", "1": "Glosada", "2": "Devuelta"},
        )
        assert resp.n_features == 48


# ---------------------------------------------------------------------------
# Property tests (Hypothesis)
# ---------------------------------------------------------------------------


@pytest.mark.property
class TestFeatureInputProperties:
    """Property-based tests para FeatureInput."""

    @given(val=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e15, max_value=1e15))
    def test_any_finite_float_accepted_for_valor_factura(self, val: float) -> None:
        """Cualquier float finito es aceptado para valor_factura."""
        features = make_realistic_features()
        features["valor_factura"] = val
        fi = FeatureInput(**features)
        assert fi.valor_factura == val

    @given(text=st.text(min_size=1, max_size=100))
    def test_any_string_accepted_for_numero_factura(self, text: str) -> None:
        """Cualquier string no vacío es aceptado como numero_factura."""
        features = make_realistic_features()
        req = PredictionRequest(
            numero_factura=text,
            features=FeatureInput(**features),
        )
        assert req.numero_factura == text

    @given(extra_name=st.text(min_size=1, max_size=30).filter(
        lambda x: x not in FeatureInput.model_fields
    ))
    def test_extra_field_always_rejected(self, extra_name: str) -> None:
        """Cualquier campo extra siempre es rechazado."""
        features = make_realistic_features()
        features[extra_name] = 0.0
        with pytest.raises(ValidationError):
            FeatureInput(**features)
