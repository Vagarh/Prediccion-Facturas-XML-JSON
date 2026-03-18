# -*- coding: utf-8 -*-
"""Tests de rendimiento para la API REST."""

import time

import psutil
import pytest

from src.config import MODEL_PATH
from tests.fixtures.sample_data import make_realistic_features

pytestmark = [
    pytest.mark.performance,
    pytest.mark.skipif(not MODEL_PATH.exists(), reason=f"Modelo no encontrado en {MODEL_PATH}"),
]


def _build_predict_payload() -> dict:
    """Construye un payload de predicción individual."""
    return {
        "numero_factura": "PERF-001",
        "features": make_realistic_features(),
    }


def _build_batch_payload(n: int) -> dict:
    """Construye un payload de predicción batch con N facturas."""
    features = make_realistic_features()
    return {
        "facturas": [
            {"numero_factura": f"PERF-{i:04d}", "features": features}
            for i in range(n)
        ]
    }


class TestSinglePredictionLatency:
    """Tests de latencia para predicción individual."""

    def test_single_prediction_under_500ms(self, test_client) -> None:
        """Predicción individual completa en <500ms."""
        payload = _build_predict_payload()

        # Warmup
        test_client.post("/v1/predict", json=payload)

        t0 = time.perf_counter()
        response = test_client.post("/v1/predict", json=payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert response.status_code == 200
        assert elapsed_ms < 500, f"Latencia: {elapsed_ms:.1f}ms (máx: 500ms)"

    def test_single_prediction_p95_under_500ms(self, test_client) -> None:
        """P95 de predicción individual en <500ms (20 runs)."""
        payload = _build_predict_payload()

        # Warmup
        test_client.post("/v1/predict", json=payload)

        latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            test_client.post("/v1/predict", json=payload)
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        assert p95 < 500, f"P95: {p95:.1f}ms (máx: 500ms)"

    def test_model_info_under_100ms(self, test_client) -> None:
        """GET /v1/model/info completa en <100ms."""
        # Warmup
        test_client.get("/v1/model/info")

        t0 = time.perf_counter()
        response = test_client.get("/v1/model/info")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert response.status_code == 200
        assert elapsed_ms < 100, f"Latencia: {elapsed_ms:.1f}ms (máx: 100ms)"


class TestBatchPredictionLatency:
    """Tests de latencia para predicción batch."""

    def test_batch_10_under_5s(self, test_client) -> None:
        """Batch de 10 facturas completa en <5s."""
        payload = _build_batch_payload(10)

        t0 = time.perf_counter()
        response = test_client.post("/v1/predict/batch", json=payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert response.status_code == 200
        assert elapsed_ms < 5000, f"Latencia: {elapsed_ms:.1f}ms (máx: 5000ms)"

    def test_batch_50_under_30s(self, test_client) -> None:
        """Batch de 50 facturas completa en <30s."""
        payload = _build_batch_payload(50)

        t0 = time.perf_counter()
        response = test_client.post("/v1/predict/batch", json=payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert response.status_code == 200
        assert elapsed_ms < 30000, f"Latencia: {elapsed_ms:.1f}ms (máx: 30000ms)"


class TestMemoryUsage:
    """Tests de uso de memoria."""

    def test_single_prediction_memory_under_100mb(self, test_client) -> None:
        """Delta de memoria para una predicción es <100MB."""
        payload = _build_predict_payload()
        process = psutil.Process()

        # Warmup
        test_client.post("/v1/predict", json=payload)

        mem_before = process.memory_info().rss / (1024 * 1024)
        test_client.post("/v1/predict", json=payload)
        mem_after = process.memory_info().rss / (1024 * 1024)

        delta_mb = mem_after - mem_before
        assert delta_mb < 100, f"Delta memoria: {delta_mb:.1f}MB (máx: 100MB)"

    def test_batch_10_memory_under_200mb(self, test_client) -> None:
        """Delta de memoria para batch de 10 es <200MB."""
        payload = _build_batch_payload(10)
        process = psutil.Process()

        mem_before = process.memory_info().rss / (1024 * 1024)
        test_client.post("/v1/predict/batch", json=payload)
        mem_after = process.memory_info().rss / (1024 * 1024)

        delta_mb = mem_after - mem_before
        assert delta_mb < 200, f"Delta memoria: {delta_mb:.1f}MB (máx: 200MB)"
