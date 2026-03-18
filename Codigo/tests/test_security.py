# -*- coding: utf-8 -*-
"""Tests de seguridad para la API REST."""

import pytest

from src.config import MODEL_PATH
from tests.fixtures.sample_data import make_malicious_payload, make_realistic_features

pytestmark = [
    pytest.mark.security,
    pytest.mark.skipif(not MODEL_PATH.exists(), reason=f"Modelo no encontrado en {MODEL_PATH}"),
]


def _predict_payload(numero_factura: str, features: dict | None = None) -> dict:
    """Construye un payload de predicción."""
    return {
        "numero_factura": numero_factura,
        "features": features or make_realistic_features(),
    }


class TestSQLInjection:
    """Tests contra inyección SQL."""

    def test_sql_in_numero_factura(self, test_client) -> None:
        """SQL injection en numero_factura no causa error 500."""
        payload = _predict_payload(make_malicious_payload("sql_injection"))
        response = test_client.post("/v1/predict", json=payload)
        assert response.status_code in {200, 422}

    def test_sql_in_feature_value(self, test_client) -> None:
        """SQL injection como valor de feature retorna 422."""
        features = make_realistic_features()
        features["valor_factura"] = "'; DROP TABLE facturas; --"
        payload = _predict_payload("SQL-TEST", features)
        response = test_client.post("/v1/predict", json=payload)
        assert response.status_code == 422

    def test_sql_payload_no_error_leak(self, test_client) -> None:
        """La respuesta no contiene errores SQL internos (stacktraces, etc)."""
        payload = _predict_payload(make_malicious_payload("sql_injection"))
        response = test_client.post("/v1/predict", json=payload)
        body = response.text.lower()
        # El numero_factura se echoea, eso es esperado.
        # Lo que NO debe aparecer es info de motor de BD.
        assert "traceback" not in body
        assert "sqlalchemy" not in body
        assert "syntax error" not in body


class TestXSS:
    """Tests contra Cross-Site Scripting."""

    def test_xss_in_numero_factura(self, test_client) -> None:
        """XSS en numero_factura no causa error 500."""
        payload = _predict_payload(make_malicious_payload("xss"))
        response = test_client.post("/v1/predict", json=payload)
        assert response.status_code in {200, 422}

    def test_xss_in_feature_value(self, test_client) -> None:
        """XSS como valor de feature retorna 422."""
        features = make_realistic_features()
        features["valor_factura"] = "<script>alert('xss')</script>"
        payload = _predict_payload("XSS-TEST", features)
        response = test_client.post("/v1/predict", json=payload)
        assert response.status_code == 422

    def test_response_content_type_is_json(self, test_client) -> None:
        """La respuesta siempre es application/json."""
        payload = _predict_payload(make_malicious_payload("xss"))
        response = test_client.post("/v1/predict", json=payload)
        assert "application/json" in response.headers.get("content-type", "")


class TestPathTraversal:
    """Tests contra path traversal."""

    def test_path_traversal_in_numero_factura(self, test_client) -> None:
        """Path traversal en numero_factura no filtra info del filesystem."""
        payload = _predict_payload(make_malicious_payload("path_traversal"))
        response = test_client.post("/v1/predict", json=payload)
        assert response.status_code in {200, 422}

    def test_no_filesystem_data_in_response(self, test_client) -> None:
        """La respuesta no contiene contenido real del filesystem."""
        payload = _predict_payload(make_malicious_payload("path_traversal"))
        response = test_client.post("/v1/predict", json=payload)
        body = response.text
        # El numero_factura se echoea (contiene "../../etc/passwd"), eso es esperado.
        # Lo que NO debe aparecer es contenido real del filesystem.
        assert "root:" not in body  # Contenido real de /etc/passwd
        assert "Program Files" not in body
        assert response.status_code == 200


class TestInputEdgeCases:
    """Tests de edge cases en inputs."""

    def test_extremely_large_number(self, test_client) -> None:
        """Valor extremadamente grande es manejado sin crash."""
        features = make_realistic_features()
        features["valor_factura"] = 1e308
        payload = _predict_payload("LARGE-001", features)
        response = test_client.post("/v1/predict", json=payload)
        assert response.status_code in {200, 422}

    def test_extremely_small_number(self, test_client) -> None:
        """Valor extremadamente pequeño es manejado sin crash."""
        features = make_realistic_features()
        features["valor_factura"] = -1e308
        payload = _predict_payload("SMALL-001", features)
        response = test_client.post("/v1/predict", json=payload)
        assert response.status_code in {200, 422}

    def test_very_long_numero_factura(self, test_client) -> None:
        """String muy largo en numero_factura no causa error 500."""
        payload = _predict_payload("A" * 10000)
        response = test_client.post("/v1/predict", json=payload)
        assert response.status_code in {200, 422}

    def test_unicode_in_numero_factura(self, test_client) -> None:
        """Caracteres Unicode en numero_factura son manejados."""
        payload = _predict_payload("FEV-2024-ñ-日本語-🔥")
        response = test_client.post("/v1/predict", json=payload)
        assert response.status_code in {200, 422}
