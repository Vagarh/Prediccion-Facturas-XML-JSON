# -*- coding: utf-8 -*-
"""Mocks de modelos y explainers para unit tests.

Proporciona mocks ligeros de XGBoost y SHAP TreeExplainer para tests unitarios
que no requieren cargar artefactos reales desde disco.
"""

from unittest.mock import Mock

import numpy as np


def make_mock_model() -> Mock:
    """Crea un mock de XGBoost model con métodos predict/predict_proba.
    
    El mock retorna predicciones determinísticas para facilitar testing:
    - predict() retorna clase 1 (Glosada)
    - predict_proba() retorna probabilidades [0.2, 0.6, 0.2]
    
    Returns:
        Mock: Mock object con spec de XGBClassifier.
    
    Examples:
        >>> mock_model = make_mock_model()
        >>> mock_model.predict(np.zeros((1, 48)))
        array([1])
        >>> mock_model.predict_proba(np.zeros((1, 48)))
        array([[0.2, 0.6, 0.2]])
    """
    mock = Mock(spec=['predict', 'predict_proba', 'load_model', 'get_booster'])
    
    # Configurar predict() para retornar clase 1 (Glosada)
    mock.predict.return_value = np.array([1])
    
    # Configurar predict_proba() para retornar probabilidades válidas
    mock.predict_proba.return_value = np.array([[0.2, 0.6, 0.2]])
    
    return mock


def make_mock_model_batch(n_samples: int = 10) -> Mock:
    """Crea un mock de XGBoost model para predicciones batch.
    
    Args:
        n_samples: Número de muestras en el batch.
    
    Returns:
        Mock: Mock object configurado para batch predictions.
    
    Examples:
        >>> mock_model = make_mock_model_batch(n_samples=5)
        >>> mock_model.predict(np.zeros((5, 48))).shape
        (5,)
        >>> mock_model.predict_proba(np.zeros((5, 48))).shape
        (5, 3)
    """
    mock = Mock(spec=['predict', 'predict_proba', 'load_model', 'get_booster'])
    
    # Configurar predict() para retornar array de clases
    mock.predict.return_value = np.array([1] * n_samples)
    
    # Configurar predict_proba() para retornar matriz de probabilidades
    mock.predict_proba.return_value = np.array(
        [[0.2, 0.6, 0.2]] * n_samples
    )
    
    return mock


def make_mock_explainer(n_features: int = 48, n_classes: int = 3) -> Mock:
    """Crea un mock de SHAP TreeExplainer con __call__ method.
    
    El mock retorna SHAP values simulados con distribución normal.
    
    Args:
        n_features: Número de features (default 48).
        n_classes: Número de clases (default 3).
    
    Returns:
        Mock: Mock object con método __call__ que retorna SHAP values.
    
    Examples:
        >>> mock_explainer = make_mock_explainer()
        >>> result = mock_explainer(np.zeros((1, 48)))
        >>> result.values.shape
        (1, 48, 3)
    """
    mock = Mock()
    
    # Crear mock de SHAP values con estructura correcta
    mock_shap_values = Mock()
    # Shape: (n_samples, n_features, n_classes)
    mock_shap_values.values = np.random.randn(1, n_features, n_classes) * 0.1
    mock_shap_values.base_values = np.array([[0.3, 0.5, 0.2]])  # Base probabilities
    
    # Configurar __call__ para retornar mock_shap_values
    mock.return_value = mock_shap_values
    
    return mock


def make_mock_explainer_batch(
    n_samples: int = 10,
    n_features: int = 48,
    n_classes: int = 3,
) -> Mock:
    """Crea un mock de SHAP TreeExplainer para batch predictions.
    
    Args:
        n_samples: Número de muestras en el batch.
        n_features: Número de features (default 48).
        n_classes: Número de clases (default 3).
    
    Returns:
        Mock: Mock object configurado para batch SHAP values.
    
    Examples:
        >>> mock_explainer = make_mock_explainer_batch(n_samples=5)
        >>> result = mock_explainer(np.zeros((5, 48)))
        >>> result.values.shape
        (5, 48, 3)
    """
    mock = Mock()
    
    # Crear mock de SHAP values con estructura correcta para batch
    mock_shap_values = Mock()
    # Shape: (n_samples, n_features, n_classes)
    mock_shap_values.values = np.random.randn(n_samples, n_features, n_classes) * 0.1
    mock_shap_values.base_values = np.array([[0.3, 0.5, 0.2]] * n_samples)
    
    # Configurar __call__ para retornar mock_shap_values
    mock.return_value = mock_shap_values
    
    return mock


def make_mock_model_with_feature_importance(n_features: int = 48) -> Mock:
    """Crea un mock de XGBoost model con feature importance.
    
    Args:
        n_features: Número de features (default 48).
    
    Returns:
        Mock: Mock object con método get_booster() que retorna feature importance.
    
    Examples:
        >>> mock_model = make_mock_model_with_feature_importance()
        >>> booster = mock_model.get_booster()
        >>> importance = booster.get_score(importance_type='weight')
        >>> len(importance)
        48
    """
    mock = Mock(spec=['predict', 'predict_proba', 'load_model', 'get_booster'])
    
    # Configurar predict methods
    mock.predict.return_value = np.array([1])
    mock.predict_proba.return_value = np.array([[0.2, 0.6, 0.2]])
    
    # Configurar booster con feature importance
    mock_booster = Mock()
    feature_importance = {f"f{i}": float(n_features - i) for i in range(n_features)}
    mock_booster.get_score.return_value = feature_importance
    mock.get_booster.return_value = mock_booster
    
    return mock
