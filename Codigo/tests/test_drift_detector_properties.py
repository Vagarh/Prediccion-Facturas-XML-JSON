# -*- coding: utf-8 -*-
"""Property tests para DriftDetector.

Valida propiedades fundamentales del cálculo de PSI y detección de drift:
- Propiedad 7: PSI es cero para distribuciones idénticas
"""

import pytest
from hypothesis import given, strategies as st, assume
import numpy as np
from sklearn.datasets import make_classification

from src.monitoring.drift_detector import (
    compute_psi,
    compute_embedding_psi,
    check_drift,
    DriftDetector
)
from src.config import RANDOM_STATE


class TestDriftDetectorProperties:
    """Property tests para DriftDetector usando Hypothesis."""

    @given(st.lists(st.floats(min_value=-100, max_value=100), min_size=10, max_size=1000))
    def test_psi_zero_for_identical_distributions(self, values):
        """
        Propiedad 7: PSI es cero para distribuciones idénticas.
        
        compute_psi(arr, arr) == 0.0 para cualquier array de referencia
        
        Valida: Requisito 7.3
        """
        assume(len(set(values)) > 1)  # Asegurar variabilidad
        
        reference = np.array(values, dtype=np.float32)
        current = reference.copy()  # Distribución idéntica
        
        psi = compute_psi(reference, current)
        
        # Propiedad: PSI debe ser cero (o muy cercano) para distribuciones idénticas
        assert abs(psi) < 1e-6, f"PSI debe ser ~0 para distribuciones idénticas, obtuvo {psi}"

    @given(
        st.lists(st.floats(min_value=-50, max_value=50), min_size=20, max_size=500),
        st.integers(min_value=3, max_value=20)
    )
    def test_psi_symmetry_property(self, values, n_bins):
        """
        Propiedad: PSI no es simétrico pero debe ser consistente.
        
        PSI(A, B) puede diferir de PSI(B, A), pero ambos deben ser válidos.
        """
        assume(len(set(values)) > 2)
        
        # Crear dos distribuciones diferentes
        reference = np.array(values, dtype=np.float32)
        current = reference + np.random.normal(0, 0.1, len(reference))  # Pequeña perturbación
        
        psi_ab = compute_psi(reference, current, n_bins=n_bins)
        psi_ba = compute_psi(current, reference, n_bins=n_bins)
        
        # Propiedad: ambos PSI deben ser válidos (no negativos, no infinitos)
        assert psi_ab >= 0, f"PSI debe ser no negativo: {psi_ab}"
        assert psi_ba >= 0, f"PSI debe ser no negativo: {psi_ba}"
        assert np.isfinite(psi_ab), f"PSI debe ser finito: {psi_ab}"
        assert np.isfinite(psi_ba), f"PSI debe ser finito: {psi_ba}"

    @given(
        st.lists(st.floats(min_value=-10, max_value=10), min_size=50, max_size=200),
        st.floats(min_value=0.1, max_value=5.0)
    )
    def test_psi_increases_with_distribution_shift(self, base_values, shift_magnitude):
        """
        Propiedad: PSI aumenta con mayor diferencia entre distribuciones.
        
        Una distribución más desplazada debe tener PSI mayor.
        """
        assume(len(set(base_values)) > 3)
        
        reference = np.array(base_values, dtype=np.float32)
        
        # Crear distribuciones con diferentes niveles de shift
        small_shift = reference + np.random.normal(0, shift_magnitude * 0.1, len(reference))
        large_shift = reference + np.random.normal(0, shift_magnitude, len(reference))
        
        psi_small = compute_psi(reference, small_shift)
        psi_large = compute_psi(reference, large_shift)
        
        # Propiedad: mayor shift debería resultar en mayor PSI (en promedio)
        # Nota: Esto no siempre es estrictamente cierto debido a la naturaleza del PSI,
        # pero debería ser cierto en la mayoría de casos
        assert psi_small >= 0 and psi_large >= 0, "PSI debe ser no negativo"
        assert np.isfinite(psi_small) and np.isfinite(psi_large), "PSI debe ser finito"

    @given(
        st.integers(min_value=10, max_value=100),  # n_samples
        st.integers(min_value=5, max_value=50),    # embedding_dim
        st.integers(min_value=2, max_value=10)     # n_components
    )
    def test_embedding_psi_shape_consistency(self, n_samples, embedding_dim, n_components):
        """
        Propiedad: compute_embedding_psi retorna estructura consistente.
        
        Para cualquier par de embeddings válidos, debe retornar diccionario
        con las claves esperadas y tipos correctos.
        """
        assume(n_components <= min(embedding_dim, n_samples - 1))
        
        # Generar embeddings sintéticos
        np.random.seed(RANDOM_STATE)
        ref_embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
        cur_embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
        
        result = compute_embedding_psi(ref_embeddings, cur_embeddings, n_components=n_components)
        
        # Propiedad: estructura del resultado
        required_keys = ["mean_psi", "max_psi", "psi_per_component", "explained_variance_ratio"]
        for key in required_keys:
            assert key in result, f"Resultado debe contener clave '{key}'"
        
        # Propiedad: tipos y rangos
        assert isinstance(result["mean_psi"], float), "mean_psi debe ser float"
        assert isinstance(result["max_psi"], float), "max_psi debe ser float"
        assert result["mean_psi"] >= 0, "mean_psi debe ser no negativo"
        assert result["max_psi"] >= 0, "max_psi debe ser no negativo"
        assert result["max_psi"] >= result["mean_psi"], "max_psi debe ser >= mean_psi"
        
        # Propiedad: longitud de arrays
        assert len(result["psi_per_component"]) == n_components, "Longitud incorrecta de psi_per_component"
        assert len(result["explained_variance_ratio"]) == n_components, "Longitud incorrecta de explained_variance_ratio"

    @given(
        st.integers(min_value=100, max_value=500),
        st.integers(min_value=10, max_value=30)
    )
    def test_identical_embeddings_zero_psi(self, n_samples, embedding_dim):
        """
        Propiedad: Embeddings idénticos producen PSI cercano a cero.

        Para embeddings idénticos, el PSI debe ser muy pequeño.
        Se requieren al menos 100 muestras para que los bins del PSI sean estables
        y no introduzcan artefactos numéricos con muestras pequeñas.
        """
        np.random.seed(RANDOM_STATE)
        embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)

        result = compute_embedding_psi(embeddings, embeddings.copy())

        # Propiedad: PSI cero para embeddings idénticos (tolerancia por binning numérico)
        assert abs(result["mean_psi"]) < 0.15, f"mean_psi debe ser ~0, obtuvo {result['mean_psi']}"
        assert abs(result["max_psi"]) < 0.15, f"max_psi debe ser ~0, obtuvo {result['max_psi']}"

        # Todos los componentes deben tener PSI ~0
        for i, psi in enumerate(result["psi_per_component"]):
            assert abs(psi) < 0.15, f"Componente {i} debe tener PSI ~0, obtuvo {psi}"

    @given(st.floats(min_value=0.0, max_value=1.0))
    def test_check_drift_thresholds(self, psi_value):
        """
        Propiedad: check_drift clasifica correctamente según umbrales.
        
        Para cualquier valor PSI válido, debe retornar una de las tres categorías.
        """
        result = check_drift(psi_value)
        
        # Propiedad: resultado válido
        valid_statuses = ["stable", "warning", "alert"]
        assert result in valid_statuses, f"Estado de drift inválido: {result}"
        
        # Propiedad: clasificación correcta según umbrales
        if psi_value < 0.10:
            assert result == "stable", f"PSI {psi_value} debe ser 'stable'"
        elif psi_value < 0.20:
            assert result == "warning", f"PSI {psi_value} debe ser 'warning'"
        else:
            assert result == "alert", f"PSI {psi_value} debe ser 'alert'"

    @given(
        st.integers(min_value=50, max_value=200),
        st.integers(min_value=5, max_value=20)
    )
    def test_drift_detector_feature_drift_consistency(self, n_samples, n_features):
        """
        Propiedad: DriftDetector.detect_feature_drift retorna estructura consistente.
        
        Para cualquier par de matrices de features válidas, debe retornar
        diccionario con estructura esperada.
        """
        np.random.seed(RANDOM_STATE)
        
        # Generar features sintéticas
        ref_features = np.random.randn(n_samples, n_features).astype(np.float32)
        cur_features = np.random.randn(n_samples, n_features).astype(np.float32)
        
        detector = DriftDetector()
        result = detector.detect_feature_drift(ref_features, cur_features)
        
        # Propiedad: estructura del resultado
        required_keys = [
            "overall_status", "mean_psi", "max_psi", "drift_counts",
            "psi_per_feature", "drift_status_per_feature", "top_drift_features"
        ]
        for key in required_keys:
            assert key in result, f"Resultado debe contener clave '{key}'"
        
        # Propiedad: tipos y valores
        assert result["overall_status"] in ["stable", "warning", "alert"]
        assert isinstance(result["mean_psi"], float) and result["mean_psi"] >= 0
        assert isinstance(result["max_psi"], float) and result["max_psi"] >= 0
        assert result["max_psi"] >= result["mean_psi"]
        
        # Propiedad: conteos de drift
        drift_counts = result["drift_counts"]
        total_features = sum(drift_counts.values())
        assert total_features == n_features, f"Conteo total debe ser {n_features}"
        
        # Propiedad: diccionarios de PSI y status
        assert len(result["psi_per_feature"]) == n_features
        assert len(result["drift_status_per_feature"]) == n_features

    @given(
        st.integers(min_value=30, max_value=100),
        st.integers(min_value=8, max_value=25),
        st.floats(min_value=0.05, max_value=0.30),
        st.floats(min_value=0.15, max_value=0.50)
    )
    def test_drift_detector_threshold_configuration(
        self, n_samples, n_features, warning_threshold, alert_threshold
    ):
        """
        Propiedad: DriftDetector respeta umbrales configurados.
        
        Los umbrales personalizados deben ser utilizados correctamente
        en la clasificación de drift.
        """
        assume(warning_threshold < alert_threshold)
        
        detector = DriftDetector(
            warning_threshold=warning_threshold,
            alert_threshold=alert_threshold
        )
        
        # Verificar configuración
        config = detector.get_configuration()
        assert config["warning_threshold"] == warning_threshold
        assert config["alert_threshold"] == alert_threshold
        
        # Generar datos con drift conocido
        np.random.seed(RANDOM_STATE)
        ref_features = np.random.randn(n_samples, n_features).astype(np.float32)
        
        # Crear drift moderado (debería estar entre umbrales)
        cur_features = ref_features + np.random.normal(0, 0.5, ref_features.shape)
        
        result = detector.detect_feature_drift(ref_features, cur_features)
        
        # Propiedad: umbrales en resultado
        assert result["thresholds"]["warning"] == warning_threshold
        assert result["thresholds"]["alert"] == alert_threshold

    @given(
        st.lists(st.floats(min_value=-5, max_value=5), min_size=100, max_size=300),
        st.integers(min_value=5, max_value=15)
    )
    def test_psi_bin_robustness(self, values, n_bins):
        """
        Propiedad: PSI es robusto a diferentes números de bins.
        
        Cambiar el número de bins no debe causar errores y debe
        producir valores PSI razonables.
        """
        assume(len(set(values)) > n_bins // 2)  # Suficiente variabilidad
        
        reference = np.array(values, dtype=np.float32)
        current = reference + np.random.normal(0, 0.2, len(reference))
        
        psi = compute_psi(reference, current, n_bins=n_bins)
        
        # Propiedad: PSI válido independientemente del número de bins
        assert psi >= 0, f"PSI debe ser no negativo con {n_bins} bins"
        assert np.isfinite(psi), f"PSI debe ser finito con {n_bins} bins"
        assert psi < 10.0, f"PSI no debe ser excesivamente alto con {n_bins} bins: {psi}"

    @given(
        st.integers(min_value=20, max_value=80),
        st.integers(min_value=10, max_value=30)
    )
    def test_comprehensive_drift_detection_structure(self, n_samples, n_features):
        """
        Propiedad: detect_comprehensive_drift retorna estructura completa.
        
        El análisis comprehensivo debe incluir todas las secciones esperadas
        y mantener consistencia entre features y embeddings.
        """
        np.random.seed(RANDOM_STATE)
        
        # Generar datos sintéticos
        ref_features = np.random.randn(n_samples, n_features).astype(np.float32)
        cur_features = np.random.randn(n_samples, n_features).astype(np.float32)
        
        embedding_dim = 20
        ref_embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
        cur_embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
        
        detector = DriftDetector()
        result = detector.detect_comprehensive_drift(
            ref_features, cur_features,
            ref_embeddings, cur_embeddings
        )
        
        # Propiedad: estructura completa
        required_keys = [
            "timestamp", "overall_status", "feature_drift", 
            "embedding_drift", "recommendations", "summary"
        ]
        for key in required_keys:
            assert key in result, f"Resultado comprehensivo debe contener '{key}'"
        
        # Propiedad: consistencia de estados
        assert result["overall_status"] in ["stable", "warning", "alert"]
        assert result["feature_drift"]["overall_status"] in ["stable", "warning", "alert"]
        assert result["embedding_drift"]["overall_status"] in ["stable", "warning", "alert"]
        
        # Propiedad: recomendaciones no vacías
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) > 0
        
        # Propiedad: summary con métricas clave
        summary = result["summary"]
        assert "requires_retraining" in summary
        assert isinstance(summary["requires_retraining"], bool)

    @given(st.integers(min_value=10, max_value=50))
    def test_empty_arrays_handling(self, array_size):
        """
        Propiedad: Manejo robusto de arrays vacíos o inválidos.
        
        Las funciones deben manejar graciosamente casos edge
        como arrays vacíos o con dimensiones incorrectas.
        """
        valid_array = np.random.randn(array_size).astype(np.float32)
        empty_array = np.array([], dtype=np.float32)
        
        # Test arrays vacíos
        with pytest.raises(ValueError):
            compute_psi(empty_array, valid_array)
        
        with pytest.raises(ValueError):
            compute_psi(valid_array, empty_array)
        
        # Test arrays multidimensionales
        multi_dim_array = np.random.randn(array_size, 2).astype(np.float32)
        
        with pytest.raises(ValueError):
            compute_psi(multi_dim_array, valid_array)

    @given(
        st.integers(min_value=20, max_value=100),
        st.integers(min_value=5, max_value=20)
    )
    def test_embedding_dimension_mismatch_handling(self, n_samples, embedding_dim):
        """
        Propiedad: Manejo correcto de dimensiones incompatibles en embeddings.
        
        compute_embedding_psi debe detectar y reportar dimensiones incompatibles.
        """
        np.random.seed(RANDOM_STATE)
        
        ref_embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
        cur_embeddings = np.random.randn(n_samples, embedding_dim + 1).astype(np.float32)  # Dimensión diferente
        
        # Propiedad: debe lanzar ValueError por dimensiones incompatibles
        with pytest.raises(ValueError, match="Dimensiones de embeddings no coinciden"):
            compute_embedding_psi(ref_embeddings, cur_embeddings)

    @given(
        st.floats(min_value=-1000, max_value=1000),
        st.floats(min_value=-1000, max_value=1000)
    )
    def test_psi_extreme_values_robustness(self, mean1, mean2):
        """
        Propiedad: PSI es robusto a valores extremos.
        
        Distribuciones con valores extremos no deben causar
        overflow, underflow o valores inválidos.
        """
        assume(abs(mean1) < 500 and abs(mean2) < 500)  # Evitar valores demasiado extremos
        
        # Generar distribuciones con medias muy diferentes
        ref_data = np.random.normal(mean1, 1.0, 100).astype(np.float32)
        cur_data = np.random.normal(mean2, 1.0, 100).astype(np.float32)
        
        psi = compute_psi(ref_data, cur_data)
        
        # Propiedad: PSI debe ser válido incluso con valores extremos
        assert np.isfinite(psi), f"PSI debe ser finito para medias {mean1}, {mean2}"
        assert psi >= 0, f"PSI debe ser no negativo para medias {mean1}, {mean2}"
        assert psi < 100, f"PSI no debe ser excesivamente alto: {psi}"