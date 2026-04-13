# -*- coding: utf-8 -*-
"""Tests de integración para endpoints LLM de la API FastAPI.

Verifica:
- POST /v1/predict sin campos textuales retorna response idéntico (compatibilidad)
- GET /v1/llm/status retorna 200 con schema correcto
- Degradación graciosa cuando Ollama no está disponible
"""

import contextlib
import concurrent.futures
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import LLMStatusResponse

MOCK_PREDICT_RESULT = {
    'estado_predicho': 'auditada', 'clase': 0, 'confianza': 0.75,
    'probabilidades': {'auditada': 0.75, 'glosada': 0.15, 'devuelta': 0.10},
    'factores_riesgo': ['Factor 1'], 'factores_detalle': [], 'causal_probable': None,
    'accion_sugerida': 'Fast-track: pago programado',
}

_MISSING = object()


@contextlib.contextmanager
def state_attrs(**kwargs):
    """Context manager que setea atributos en app.state y los restaura al salir."""
    old_values = {}
    for key, value in kwargs.items():
        old_values[key] = getattr(app.state, key, _MISSING)
        setattr(app.state, key, value)
    try:
        yield
    finally:
        for key, old_val in old_values.items():
            if old_val is _MISSING:
                try:
                    delattr(app.state, key)
                except AttributeError:
                    pass
            else:
                setattr(app.state, key, old_val)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_FEATURES = {
    "anio_radicacion": 2024.0, "avg_valor_item": 50000.0,
    "desviacion_dias_nit": 2.5, "dias_hasta_radicacion": 15.0,
    "dias_radicacion_promedio_nit": 12.0, "dias_x_log_valor": 180.5,
    "discrepancia_valor": 1000.0, "is_extemporanea": 0.0,
    "log_avg_valor_item": 10.82, "log_discrepancia_valor": 6.91,
    "log_max_valor_item": 11.51, "log_num_cups_distintos": 1.39,
    "log_num_dx_distintos": 0.69, "log_num_items": 2.30,
    "log_num_items_con_autorizacion": 1.61, "log_sum_valor_facturado": 11.92,
    "log_total_facturas_nit": 8.52, "log_valor_copago": 0.0,
    "log_valor_cuota_moderadora": 0.0, "log_valor_factura": 11.92,
    "log_valor_promedio_nit": 10.85, "log_valor_x_tasa_glosa": 0.85,
    "max_valor_item": 100000.0, "mes_radicacion": 3.0,
    "num_cups_distintos": 3.0, "num_dx_distintos": 1.0,
    "num_items": 10.0, "num_items_con_autorizacion": 4.0,
    "num_tipos_servicio": 2.0, "porcentaje_mayor_item": 0.67,
    "ratio_autorizacion": 0.4, "ratio_cups_items": 0.3,
    "ratio_valor_vs_nit": 1.05, "regimen_encoded": 1.0,
    "sin_detalles": 0.0, "sum_valor_facturado": 150000.0,
    "tasa_devolucion_nit": 0.05, "tasa_glosa_nit": 0.12,
    "tiene_copago": 0.0, "tiene_cuota_mod": 0.0, "tiene_fev": 1.0,
    "tipo_contrato_encoded": 1.0, "total_facturas_nit": 5000.0,
    "trimestre_radicacion": 1.0, "valor_copago": 0.0,
    "valor_cuota_moderadora": 0.0, "valor_factura": 150000.0,
    "valor_promedio_nit": 142000.0,
}


class TestAPILLMIntegration:
    """Tests de integración para funcionalidades LLM en la API."""

    def setup_method(self):
        self.client = TestClient(app, raise_server_exceptions=False)
        self.sample_features = SAMPLE_FEATURES.copy()

        # Pre-inicializar app.state con mocks para que los endpoints funcionen
        # sin necesidad de correr el lifespan completo.
        # Se usa setattr siempre (no solo si no existe) para evitar state leak entre tests.
        from unittest.mock import MagicMock
        import numpy as np

        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.7, 0.2, 0.1]]
        setattr(app.state, "model", mock_model)

        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = [np.zeros((1, 48))] * 3
        setattr(app.state, "explainer", mock_explainer)

        setattr(app.state, "feature_names", list(SAMPLE_FEATURES.keys()))

        # Estado LLM se resetea a False/None en cada test para evitar contaminación
        setattr(app.state, "llm_available", False)
        setattr(app.state, "explanation_generator", None)
        setattr(app.state, "embedding_service", None)
        setattr(app.state, "nlp_pipeline", None)
        setattr(app.state, "drift_detector", None)
        setattr(app.state, "model_monitor", None)
        setattr(app.state, "retraining_scheduler", None)

    # ------------------------------------------------------------------
    # Compatibilidad hacia atrás
    # ------------------------------------------------------------------

    def test_predict_without_text_fields_backward_compatibility(self):
        """POST /v1/predict sin campos textuales mantiene compatibilidad."""
        request_data = {"numero_factura": "FEV-2024-001234", "features": self.sample_features}

        mock_result = {
            "estado_predicho": "auditada", "clase": 0, "confianza": 0.75,
            "probabilidades": {"auditada": 0.75, "glosada": 0.15, "devuelta": 0.10},
            "factores_riesgo": ["Factor 1", "Factor 2"],
            "factores_detalle": [], "causal_probable": None,
            "accion_sugerida": "Fast-track: pago programado",
        }

        with state_attrs(llm_available=False), \
             patch("src.api.main.explain_single_prediction", return_value=mock_result):
            response = self.client.post("/v1/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()

        for field in ["numero_factura", "estado_predicho", "clase", "confianza",
                      "probabilidades", "factores_riesgo", "factores_detalle",
                      "causal_probable", "accion_sugerida", "timestamp_prediccion"]:
            assert field in data

        assert "explicacion_enriquecida" in data
        assert "drift_warning" in data
        assert data["explicacion_enriquecida"] is None
        assert data["drift_warning"] is None
        assert data["numero_factura"] == "FEV-2024-001234"
        assert data["clase"] in [0, 1, 2]
        assert 0.0 <= data["confianza"] <= 1.0

    # ------------------------------------------------------------------
    # Predicción con LLM
    # ------------------------------------------------------------------

    def test_predict_with_text_fields_llm_available(self):
        """Predicción con campos textuales cuando LLM está disponible."""
        request_data = {
            "numero_factura": "FEV-2024-005678",
            "features": self.sample_features,
            "diagnostico_texto": "Insuficiencia cardíaca congestiva",
            "cups_texto": "Consulta de cardiología",
            "observaciones": "Paciente con antecedentes de hipertensión",
        }

        mock_gen = Mock()
        mock_gen.generate.return_value = "Explicación enriquecida de prueba"

        with state_attrs(
            llm_available=True,
            explanation_generator=mock_gen,
            embedding_service=Mock(),
            nlp_pipeline=Mock(),
        ), patch("src.api.main.enrich_features", return_value=[1.0] * 100) as mock_enrich:
            response = self.client.post("/v1/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["explicacion_enriquecida"] == "Explicación enriquecida de prueba"
        mock_enrich.assert_called_once()

    def test_predict_with_text_fields_llm_unavailable(self):
        """Predicción con campos textuales cuando LLM no está disponible."""
        request_data = {
            "numero_factura": "FEV-2024-009999",
            "features": self.sample_features,
            "diagnostico_texto": "Diabetes mellitus tipo 2",
        }

        with state_attrs(llm_available=False):
            response = self.client.post("/v1/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["explicacion_enriquecida"] is None
        assert data["numero_factura"] == "FEV-2024-009999"
        assert data["clase"] in [0, 1, 2]

    # ------------------------------------------------------------------
    # LLM Status endpoint
    # ------------------------------------------------------------------

    def test_llm_status_endpoint_available(self):
        """GET /v1/llm/status cuando LLM está disponible."""
        mock_gen = Mock()
        mock_gen.get_status.return_value = {
            "ollama_available": True,
            "llm_model": "llama3.2",
            "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        }
        mock_gen.test_generation.return_value = {"success": True, "elapsed_time_s": 2.5}

        with state_attrs(llm_available=True, explanation_generator=mock_gen, drift_detector=Mock()):
            response = self.client.get("/v1/llm/status")

        assert response.status_code == 200
        data = response.json()
        for field in ["ollama_available", "active_model", "embedding_model",
                      "latency_p50_ms", "latency_p95_ms", "drift_monitoring_active", "last_drift_check"]:
            assert field in data
        assert data["ollama_available"] is True
        assert data["active_model"] == "llama3.2"
        assert data["drift_monitoring_active"] is True
        assert data["latency_p50_ms"] == 2500.0

    def test_llm_status_endpoint_unavailable(self):
        """GET /v1/llm/status cuando LLM no está disponible."""
        with state_attrs(llm_available=False):
            response = self.client.get("/v1/llm/status")

        assert response.status_code == 200
        data = response.json()
        assert data["ollama_available"] is False
        assert data["active_model"] is None
        assert data["drift_monitoring_active"] is False
        assert data["latency_p50_ms"] is None

    def test_llm_status_endpoint_error_handling(self):
        """Manejo de errores en endpoint LLM status."""
        mock_gen = Mock()
        mock_gen.get_status.side_effect = Exception("Test error")

        with state_attrs(llm_available=True, explanation_generator=mock_gen):
            response = self.client.get("/v1/llm/status")

        assert response.status_code == 200
        data = response.json()
        assert data["ollama_available"] is False

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def test_batch_predict_with_llm_integration(self):
        """Predicción batch con integración LLM."""
        request_data = {
            "facturas": [
                {"numero_factura": "FEV-2024-BATCH-001", "features": self.sample_features},
                {"numero_factura": "FEV-2024-BATCH-002", "features": self.sample_features,
                 "diagnostico_texto": "Hipertensión arterial"},
            ]
        }

        mock_gen = Mock()
        mock_gen.generate.return_value = "Explicación batch de prueba"

        with state_attrs(llm_available=True, explanation_generator=mock_gen), \
             patch("src.api.main.explain_batch_predictions", return_value=[MOCK_PREDICT_RESULT] * 2):
            response = self.client.post("/v1/predict/batch", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        for pred in data["predicciones"]:
            assert pred["explicacion_enriquecida"] == "Explicación batch de prueba"

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_predict_explanation_generation_error_handling(self):
        """Manejo de errores en generación de explicaciones."""
        request_data = {
            "numero_factura": "FEV-2024-ERROR-TEST",
            "features": self.sample_features,
            "diagnostico_texto": "Test error handling",
        }

        mock_gen = Mock()
        mock_gen.generate.side_effect = Exception("Explanation generation failed")

        with state_attrs(llm_available=True, explanation_generator=mock_gen), \
             patch("src.api.main.enrich_features", return_value=[1.0] * 100):
            response = self.client.post("/v1/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["explicacion_enriquecida"] is None
        assert data["clase"] in [0, 1, 2]

    def test_predict_embedding_enrichment_error_handling(self):
        """Manejo de errores en enriquecimiento de embeddings."""
        request_data = {
            "numero_factura": "FEV-2024-EMBED-ERROR",
            "features": self.sample_features,
            "diagnostico_texto": "Test embedding error",
            "cups_texto": "Test CUPS",
        }

        with state_attrs(llm_available=True), \
             patch("src.api.main.enrich_features", side_effect=Exception("Embedding failed")):
            response = self.client.post("/v1/predict", json=request_data)

        assert response.status_code == 200
        assert response.json()["numero_factura"] == "FEV-2024-EMBED-ERROR"

    # ------------------------------------------------------------------
    # Compatibilidad endpoints existentes
    # ------------------------------------------------------------------

    def test_health_check_compatibility(self):
        """Health check mantiene compatibilidad."""
        response = self.client.get("/v1/health")
        assert response.status_code == 200
        for field in ["status", "modelo_cargado", "version", "timestamp"]:
            assert field in response.json()

    def test_model_info_compatibility(self):
        """Model info mantiene compatibilidad."""
        response = self.client.get("/v1/model/info")
        assert response.status_code == 200
        for field in ["modelo", "version", "fecha_entrenamiento", "metricas",
                      "n_features", "features", "clases"]:
            assert field in response.json()

    def test_request_validation_with_new_fields(self):
        """Validación de request con nuevos campos opcionales."""
        valid_request = {
            "numero_factura": "FEV-2024-VALIDATION",
            "features": self.sample_features,
            "diagnostico_texto": "Diagnóstico válido",
            "cups_texto": "CUPS válido",
            "observaciones": "Observaciones válidas",
        }
        with state_attrs(llm_available=False):
            assert self.client.post("/v1/predict", json=valid_request).status_code == 200

    def test_request_validation_with_null_text_fields(self):
        """Validación con campos textuales nulos."""
        request_with_nulls = {
            "numero_factura": "FEV-2024-NULLS",
            "features": self.sample_features,
            "diagnostico_texto": None,
            "cups_texto": None,
            "observaciones": None,
        }
        with state_attrs(llm_available=False):
            assert self.client.post("/v1/predict", json=request_with_nulls).status_code == 200

    def test_llm_status_response_schema_validation(self):
        """Respuesta de LLM status cumple con schema."""
        with state_attrs(llm_available=False):
            response = self.client.get("/v1/llm/status")
        assert response.status_code == 200
        data = response.json()
        llm_status = LLMStatusResponse(**data)
        assert isinstance(llm_status.ollama_available, bool)
        assert isinstance(llm_status.embedding_model, str)
        assert isinstance(llm_status.drift_monitoring_active, bool)

    def test_concurrent_requests_with_llm(self):
        """Manejo de requests concurrentes con LLM."""
        request_data = {
            "numero_factura": "FEV-2024-CONCURRENT",
            "features": self.sample_features,
            "diagnostico_texto": "Test concurrencia",
        }

        mock_gen = Mock()
        mock_gen.generate.return_value = "Explicación concurrente"

        # Los mocks se aplican en el hilo principal para evitar race conditions en app.state
        with state_attrs(llm_available=True, explanation_generator=mock_gen), \
             patch("src.api.main.enrich_features", return_value=[1.0] * 100):
            def make_request():
                return self.client.post("/v1/predict", json=request_data)

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                responses = [f.result() for f in [executor.submit(make_request) for _ in range(3)]]

        for response in responses:
            assert response.status_code == 200
            assert response.json()["explicacion_enriquecida"] == "Explicación concurrente"
