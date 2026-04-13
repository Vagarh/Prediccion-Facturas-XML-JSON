# -*- coding: utf-8 -*-
"""Fixtures compartidos para los tests del proyecto."""

import os
import sys
from pathlib import Path

import pytest

# Asegurar que el directorio raíz del proyecto está en el path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.engineering import load_feature_names


# ---------------------------------------------------------------------------
# Session-scoped fixtures (loaded once per test session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def feature_names() -> list[str]:
    """Lista de 48 features en el orden esperado por el modelo.
    
    Scope: session - Cargado una vez al inicio de la sesión de tests.
    """
    return load_feature_names()


@pytest.fixture(scope="session")
def model():
    """Modelo XGBoost cargado desde artifacts para integration tests.
    
    Scope: session - Cargado una vez al inicio de la sesión de tests.
    Solo se usa en integration tests, no en unit tests (que usan mocks).
    
    Returns:
        xgb.XGBClassifier: Modelo entrenado listo para predicción.
    """
    from src.models.predict import load_model
    
    return load_model()


@pytest.fixture(scope="session")
def explainer(model):
    """SHAP TreeExplainer para integration tests.
    
    Scope: session - Creado una vez al inicio de la sesión de tests.
    Depende del fixture model. Solo se usa en integration tests.
    
    Args:
        model: Fixture del modelo XGBoost.
    
    Returns:
        shap.TreeExplainer: Explainer listo para calcular SHAP values.
    """
    from src.explainability.shap_explanations import create_shap_explainer
    
    return create_shap_explainer(model)


# ---------------------------------------------------------------------------
# Module-scoped fixtures (loaded once per test module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_client():
    """TestClient de FastAPI con modelo cargado via lifespan context.
    
    Scope: module - Creado una vez por módulo de tests.
    Usa el lifespan context de FastAPI para cargar modelo al inicio.
    
    Returns:
        TestClient: Cliente de pruebas de FastAPI.
    """
    from fastapi.testclient import TestClient
    from src.api.main import app

    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# Function-scoped fixtures (fresh instance per test)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Function-scoped fixtures (fresh instance per test)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_features(feature_names: list[str]) -> dict[str, float]:
    """Diccionario de features de ejemplo con valores válidos (todos en 0.0).

    Útil para tests que no dependen de valores realistas.
    
    Scope: function - Nueva instancia por cada test.
    
    Args:
        feature_names: Fixture con lista de 48 features.
    
    Returns:
        dict: Diccionario con 48 features, todas en 0.0.
    """
    return {name: 0.0 for name in feature_names}


@pytest.fixture
def realistic_features(feature_names: list[str]) -> dict[str, float]:
    """Diccionario de features con valores realistas para una factura típica.

    Basado en medianas aproximadas del dataset de entrenamiento.
    
    Scope: function - Nueva instancia por cada test.
    
    Args:
        feature_names: Fixture con lista de 48 features.
    
    Returns:
        dict: Diccionario con 48 features con valores realistas.
    """
    return {
        "anio_radicacion": 2025.0,
        "avg_valor_item": 50000.0,
        "desviacion_dias_nit": 1.2,
        "dias_hasta_radicacion": 15.0,
        "dias_radicacion_promedio_nit": 12.0,
        "dias_x_log_valor": 180.0,
        "discrepancia_valor": 0.0,
        "is_extemporanea": 0.0,
        "log_avg_valor_item": 10.82,
        "log_discrepancia_valor": 0.0,
        "log_max_valor_item": 11.5,
        "log_num_cups_distintos": 1.1,
        "log_num_dx_distintos": 0.69,
        "log_num_items": 1.1,
        "log_num_items_con_autorizacion": 0.69,
        "log_sum_valor_facturado": 12.0,
        "log_total_facturas_nit": 10.8,
        "log_valor_copago": 0.0,
        "log_valor_cuota_moderadora": 0.0,
        "log_valor_factura": 12.0,
        "log_valor_promedio_nit": 11.5,
        "log_valor_x_tasa_glosa": 1.5,
        "max_valor_item": 100000.0,
        "mes_radicacion": 6.0,
        "num_cups_distintos": 2.0,
        "num_dx_distintos": 1.0,
        "num_items": 3.0,
        "num_items_con_autorizacion": 1.0,
        "num_tipos_servicio": 2.0,
        "porcentaje_mayor_item": 0.6,
        "ratio_autorizacion": 0.33,
        "ratio_cups_items": 0.67,
        "ratio_valor_vs_nit": 1.1,
        "regimen_encoded": 0.0,
        "sin_detalles": 0.0,
        "sum_valor_facturado": 150000.0,
        "tasa_devolucion_nit": 0.03,
        "tasa_glosa_nit": 0.13,
        "tiene_copago": 0.0,
        "tiene_cuota_mod": 0.0,
        "tiene_fev": 1.0,
        "tipo_contrato_encoded": 1.0,
        "total_facturas_nit": 50000.0,
        "trimestre_radicacion": 2.0,
        "valor_copago": 0.0,
        "valor_cuota_moderadora": 0.0,
        "valor_factura": 150000.0,
        "valor_promedio_nit": 130000.0,
    }


# ---------------------------------------------------------------------------
# Hypothesis configuration
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Configure Hypothesis profiles for CI and dev environments."""
    from hypothesis import settings, HealthCheck
    
    # CI profile: more iterations, stricter timeouts
    settings.register_profile(
        "ci",
        max_examples=100,
        deadline=5000,  # 5 seconds per test
        suppress_health_check=[HealthCheck.too_slow],
    )
    
    # Dev profile: fewer iterations, no deadline
    settings.register_profile(
        "dev",
        max_examples=20,
        deadline=None,
    )
    
    # Load profile based on environment
    profile = "ci" if os.getenv("CI") else "dev"
    settings.load_profile(profile)


# ---------------------------------------------------------------------------
# spaCy mock — evita cargar es_core_news_md en tests unitarios
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_spacy_model_if_not_installed():
    """Mockea spacy.load si es_core_news_md no está instalado.

    Permite que los tests unitarios corran sin el modelo spaCy instalado.
    Los tests de integración que necesiten el modelo real deben marcarse
    con @pytest.mark.integration.
    """
    import importlib
    try:
        import spacy
        spacy.load("es_core_news_md")
        # Modelo disponible — no mockear
        yield
    except OSError:
        # Modelo no disponible — mockear el singleton
        import src.nlp.pipeline as nlp_module
        from unittest.mock import Mock, patch

        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([]))
        mock_nlp.return_value = mock_doc

        original_instance = nlp_module._nlp_instance

        with patch('spacy.load', return_value=mock_nlp):
            nlp_module._nlp_instance = None
            yield
            nlp_module._nlp_instance = original_instance
