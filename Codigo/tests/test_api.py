# -*- coding: utf-8 -*-
"""Tests para los endpoints de la API REST."""

import pytest

from src.config import MODEL_PATH

# Saltar todos los tests si el modelo no existe en disco
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"Modelo no encontrado en {MODEL_PATH}",
)


class TestHealthEndpoint:
    """Tests para GET /v1/health."""

    def test_health_returns_200(self, test_client) -> None:
        """El health check retorna 200."""
        response = test_client.get("/v1/health")
        assert response.status_code == 200

    def test_health_model_loaded(self, test_client) -> None:
        """El modelo está cargado tras el startup."""
        response = test_client.get("/v1/health")
        data = response.json()
        assert data["status"] == "ok"
        assert data["modelo_cargado"] is True

    def test_health_has_timestamp(self, test_client) -> None:
        """El health check incluye timestamp."""
        response = test_client.get("/v1/health")
        data = response.json()
        assert "timestamp" in data


class TestModelInfoEndpoint:
    """Tests para GET /v1/model/info."""

    def test_model_info_returns_200(self, test_client) -> None:
        """Model info retorna 200."""
        response = test_client.get("/v1/model/info")
        assert response.status_code == 200

    def test_model_info_structure(self, test_client) -> None:
        """Model info tiene la estructura esperada."""
        response = test_client.get("/v1/model/info")
        data = response.json()
        assert data["modelo"] == "XGBoost"
        assert data["n_features"] == 48
        assert len(data["features"]) == 48
        assert "f1_macro" in data["metricas"]

    def test_model_info_classes(self, test_client) -> None:
        """Model info incluye las 3 clases."""
        response = test_client.get("/v1/model/info")
        data = response.json()
        assert len(data["clases"]) == 3
        assert data["clases"]["0"] == "Auditada"
        assert data["clases"]["1"] == "Glosada"
        assert data["clases"]["2"] == "Devuelta"


class TestPredictEndpoint:
    """Tests para POST /v1/predict."""

    def test_predict_returns_200(
        self, test_client, realistic_features: dict
    ) -> None:
        """Predicción con features válidas retorna 200."""
        payload = {
            "numero_factura": "TEST-001",
            "features": realistic_features,
        }
        response = test_client.post("/v1/predict", json=payload)
        assert response.status_code == 200

    def test_predict_response_schema(
        self, test_client, realistic_features: dict
    ) -> None:
        """La respuesta tiene todos los campos del schema."""
        payload = {
            "numero_factura": "TEST-002",
            "features": realistic_features,
        }
        response = test_client.post("/v1/predict", json=payload)
        data = response.json()

        assert data["numero_factura"] == "TEST-002"
        assert data["estado_predicho"] in {"auditada", "glosada", "devuelta"}
        assert data["clase"] in {0, 1, 2}
        assert 0 <= data["confianza"] <= 1
        assert "auditada" in data["probabilidades"]
        assert "glosada" in data["probabilidades"]
        assert "devuelta" in data["probabilidades"]
        assert len(data["factores_riesgo"]) > 0
        assert data["accion_sugerida"] != ""
        assert "timestamp_prediccion" in data

    def test_predict_probabilities_sum(
        self, test_client, realistic_features: dict
    ) -> None:
        """Las probabilidades suman ~1.0."""
        payload = {
            "numero_factura": "TEST-003",
            "features": realistic_features,
        }
        response = test_client.post("/v1/predict", json=payload)
        probs = response.json()["probabilidades"]
        total = probs["auditada"] + probs["glosada"] + probs["devuelta"]
        assert abs(total - 1.0) < 1e-4

    def test_predict_missing_features_returns_422(self, test_client) -> None:
        """Features incompletas retornan 422."""
        payload = {
            "numero_factura": "TEST-BAD",
            "features": {"valor_factura": 1.0},
        }
        response = test_client.post("/v1/predict", json=payload)
        assert response.status_code == 422

    def test_predict_extra_field_returns_422(
        self, test_client, realistic_features: dict
    ) -> None:
        """Features con campo extra retornan 422 (extra='forbid')."""
        features_with_extra = {**realistic_features, "campo_invalido": 999.0}
        payload = {
            "numero_factura": "TEST-EXTRA",
            "features": features_with_extra,
        }
        response = test_client.post("/v1/predict", json=payload)
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests para POST /v1/predict/batch."""

    def test_batch_returns_200(
        self, test_client, realistic_features: dict
    ) -> None:
        """Batch con facturas válidas retorna 200."""
        payload = {
            "facturas": [
                {"numero_factura": "BATCH-001", "features": realistic_features},
                {"numero_factura": "BATCH-002", "features": realistic_features},
                {"numero_factura": "BATCH-003", "features": realistic_features},
            ]
        }
        response = test_client.post("/v1/predict/batch", json=payload)
        assert response.status_code == 200

    def test_batch_response_count(
        self, test_client, realistic_features: dict
    ) -> None:
        """El batch retorna el número correcto de predicciones."""
        payload = {
            "facturas": [
                {"numero_factura": f"BATCH-{i:03d}", "features": realistic_features}
                for i in range(5)
            ]
        }
        response = test_client.post("/v1/predict/batch", json=payload)
        data = response.json()
        assert data["total"] == 5
        assert len(data["predicciones"]) == 5
        assert data["tiempo_procesamiento_ms"] > 0

    def test_batch_preserves_factura_numbers(
        self, test_client, realistic_features: dict
    ) -> None:
        """Los números de factura se preservan en el orden correcto."""
        numbers = ["FEV-001", "FEV-002"]
        payload = {
            "facturas": [
                {"numero_factura": n, "features": realistic_features}
                for n in numbers
            ]
        }
        response = test_client.post("/v1/predict/batch", json=payload)
        data = response.json()
        returned_numbers = [p["numero_factura"] for p in data["predicciones"]]
        assert returned_numbers == numbers

    def test_batch_single_item(
        self, test_client, realistic_features: dict
    ) -> None:
        """Batch con un solo elemento funciona correctamente."""
        payload = {
            "facturas": [
                {"numero_factura": "SINGLE-001", "features": realistic_features}
            ]
        }
        response = test_client.post("/v1/predict/batch", json=payload)
        assert response.status_code == 200
        assert response.json()["total"] == 1

    def test_batch_processing_time_positive(
        self, test_client, realistic_features: dict
    ) -> None:
        """El tiempo de procesamiento es positivo."""
        payload = {
            "facturas": [
                {"numero_factura": "TIME-001", "features": realistic_features}
            ]
        }
        response = test_client.post("/v1/predict/batch", json=payload)
        assert response.json()["tiempo_procesamiento_ms"] > 0

    def test_batch_empty_returns_error(self, test_client) -> None:
        """Lista vacía retorna error (422 o 500 dependiendo del handler)."""
        payload = {"facturas": []}
        response = test_client.post("/v1/predict/batch", json=payload)
        assert response.status_code in {422, 500}


class TestPredictEndpointExtended:
    """Tests adicionales para POST /v1/predict."""

    def test_predict_wrong_type_returns_422(
        self, test_client, realistic_features: dict
    ) -> None:
        """Valor de tipo incorrecto retorna 422."""
        bad_features = {**realistic_features, "valor_factura": "not_a_number"}
        payload = {
            "numero_factura": "TEST-TYPE",
            "features": bad_features,
        }
        response = test_client.post("/v1/predict", json=payload)
        assert response.status_code == 422

    def test_predict_empty_body_returns_422(self, test_client) -> None:
        """Body vacío retorna 422."""
        response = test_client.post("/v1/predict", json={})
        assert response.status_code == 422

    def test_predict_deterministic(
        self, test_client, realistic_features: dict
    ) -> None:
        """Dos predicciones con el mismo input producen la misma clase."""
        payload = {
            "numero_factura": "DETERM-001",
            "features": realistic_features,
        }
        r1 = test_client.post("/v1/predict", json=payload).json()
        r2 = test_client.post("/v1/predict", json=payload).json()
        assert r1["clase"] == r2["clase"]
        assert r1["confianza"] == r2["confianza"]


class TestNonExistentEndpoints:
    """Tests para endpoints que no existen."""

    def test_404_for_unknown_route(self, test_client) -> None:
        """Ruta desconocida retorna 404."""
        response = test_client.get("/v1/nonexistent")
        assert response.status_code == 404

    def test_wrong_method(self, test_client) -> None:
        """Método HTTP incorrecto retorna 405."""
        response = test_client.put("/v1/predict", json={})
        assert response.status_code == 405
