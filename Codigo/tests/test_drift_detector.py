# -*- coding: utf-8 -*-
"""Tests unitarios para DriftDetector.

Verifica funcionalidad específica del detector de drift:
- PSI > 0.20 activa nivel "alert"
- Distribuciones idénticas producen PSI ≈ 0
- Configuración de umbrales personalizada
"""

import pytest
import numpy as np
from unittest.mock import patch

from src.monitoring.drift_detector import (
    compute_psi,
    compute_embedding_psi,
    check_drift,
    DriftDetector,
    PSI_THRESHOLDS
)
from src.config import RANDOM_STATE


class TestDriftDetector:
    """Tests unitarios para DriftDetector."""

    def test_compute_psi_identical_distributions(self):
        """Test que distribuciones idénticas producen PSI ≈ 0."""
        np.random.seed(RANDOM_STATE)
        data = np.random.normal(0, 1, 1000)
        
        psi = compute_psi(data, data.copy())
        
        assert abs(psi) < 1e-10, f"PSI para distribuciones idénticas debe ser ~0, obtuvo {psi}"

    def test_compute_psi_different_distributions(self):
        """Test que distribuciones diferentes producen PSI > 0."""
        np.random.seed(RANDOM_STATE)
        
        # Distribuciones con diferentes medias
        ref_data = np.random.normal(0, 1, 1000)
        cur_data = np.random.normal(2, 1, 1000)  # Media desplazada
        
        psi = compute_psi(ref_data, cur_data)
        
        assert psi > 0.1, f"PSI para distribuciones diferentes debe ser > 0.1, obtuvo {psi}"
        assert np.isfinite(psi), "PSI debe ser finito"

    def test_compute_psi_high_drift_scenario(self):
        """Test PSI alto para distribuciones muy diferentes."""
        np.random.seed(RANDOM_STATE)
        
        # Distribuciones muy diferentes
        ref_data = np.random.normal(0, 1, 500)
        cur_data = np.random.normal(5, 2, 500)  # Media y varianza muy diferentes
        
        psi = compute_psi(ref_data, cur_data)
        
        assert psi > 0.5, f"PSI para distribuciones muy diferentes debe ser alto, obtuvo {psi}"

    def test_compute_psi_custom_bins(self):
        """Test PSI con diferentes números de bins."""
        np.random.seed(RANDOM_STATE)
        
        ref_data = np.random.normal(0, 1, 200)
        cur_data = np.random.normal(1, 1, 200)
        
        # Probar diferentes números de bins
        for n_bins in [5, 10, 15, 20]:
            psi = compute_psi(ref_data, cur_data, n_bins=n_bins)
            assert psi > 0, f"PSI debe ser positivo con {n_bins} bins"
            assert np.isfinite(psi), f"PSI debe ser finito con {n_bins} bins"

    def test_compute_psi_edge_cases(self):
        """Test PSI con casos edge."""
        # Array con valores constantes
        constant_data = np.full(100, 5.0)
        slightly_different = constant_data + np.random.normal(0, 0.01, 100)
        
        psi = compute_psi(constant_data, slightly_different)
        assert psi >= 0, "PSI debe ser no negativo para datos casi constantes"
        
        # Arrays muy pequeños
        small_ref = np.array([1.0, 2.0, 3.0])
        small_cur = np.array([1.1, 2.1, 3.1])
        
        psi = compute_psi(small_ref, small_cur)
        assert psi >= 0, "PSI debe ser no negativo para arrays pequeños"

    def test_compute_psi_invalid_inputs(self):
        """Test que compute_psi maneja inputs inválidos correctamente."""
        valid_data = np.random.normal(0, 1, 100)
        
        # Arrays vacíos
        with pytest.raises(ValueError, match="Arrays no pueden estar vacíos"):
            compute_psi(np.array([]), valid_data)
        
        # Arrays multidimensionales
        multi_dim = np.random.normal(0, 1, (50, 2))
        with pytest.raises(ValueError, match="Arrays deben ser unidimensionales"):
            compute_psi(multi_dim, valid_data)

    def test_compute_embedding_psi_identical_embeddings(self):
        """Test que embeddings idénticos producen PSI ≈ 0."""
        np.random.seed(RANDOM_STATE)
        
        embeddings = np.random.normal(0, 1, (100, 20)).astype(np.float32)
        
        result = compute_embedding_psi(embeddings, embeddings.copy())
        
        # PCA introduce ruido numérico de punto flotante; tolerancia razonable
        assert abs(result["mean_psi"]) < 0.10, "mean_psi debe ser ~0 para embeddings idénticos"
        assert abs(result["max_psi"]) < 0.10, "max_psi debe ser ~0 para embeddings idénticos"
        
        # Todos los componentes deben tener PSI ~0
        for psi in result["psi_per_component"]:
            assert abs(psi) < 0.10, f"Componente PSI debe ser ~0, obtuvo {psi}"

    def test_compute_embedding_psi_different_embeddings(self):
        """Test PSI para embeddings diferentes."""
        np.random.seed(RANDOM_STATE)
        
        ref_embeddings = np.random.normal(0, 1, (100, 15)).astype(np.float32)
        cur_embeddings = np.random.normal(1, 1, (100, 15)).astype(np.float32)  # Desplazados
        
        result = compute_embedding_psi(ref_embeddings, cur_embeddings)
        
        assert result["mean_psi"] > 0, "mean_psi debe ser > 0 para embeddings diferentes"
        assert result["max_psi"] >= result["mean_psi"], "max_psi debe ser >= mean_psi"
        assert result["n_components_used"] <= min(15, 100 - 1), "Componentes usados debe ser válido"
        assert 0 < result["total_variance_explained"] <= 1, "Varianza explicada debe estar en (0,1]"

    def test_compute_embedding_psi_dimension_mismatch(self):
        """Test que dimensiones incompatibles lanzan error."""
        ref_embeddings = np.random.normal(0, 1, (50, 10)).astype(np.float32)
        cur_embeddings = np.random.normal(0, 1, (50, 12)).astype(np.float32)  # Dimensión diferente
        
        with pytest.raises(ValueError, match="Dimensiones de embeddings no coinciden"):
            compute_embedding_psi(ref_embeddings, cur_embeddings)

    def test_compute_embedding_psi_empty_embeddings(self):
        """Test que embeddings vacíos lanzan error."""
        empty_embeddings = np.array([]).reshape(0, 10).astype(np.float32)
        valid_embeddings = np.random.normal(0, 1, (50, 10)).astype(np.float32)
        
        with pytest.raises(ValueError, match="Embeddings no pueden estar vacíos"):
            compute_embedding_psi(empty_embeddings, valid_embeddings)

    def test_check_drift_thresholds(self):
        """Test que check_drift clasifica correctamente según umbrales."""
        # Valores estables
        assert check_drift(0.05) == "stable"
        assert check_drift(0.09) == "stable"
        
        # Valores de warning
        assert check_drift(0.10) == "warning"
        assert check_drift(0.15) == "warning"
        assert check_drift(0.19) == "warning"
        
        # Valores de alert
        assert check_drift(0.20) == "alert"
        assert check_drift(0.25) == "alert"
        assert check_drift(1.0) == "alert"

    def test_check_drift_boundary_values(self):
        """Test valores en los límites de umbrales."""
        # Exactamente en los umbrales
        assert check_drift(0.099999) == "stable"
        assert check_drift(0.100001) == "warning"
        assert check_drift(0.199999) == "warning"
        assert check_drift(0.200001) == "alert"

    def test_drift_detector_initialization(self):
        """Test inicialización de DriftDetector con parámetros personalizados."""
        detector = DriftDetector(
            warning_threshold=0.15,
            alert_threshold=0.25,
            n_pca_components=8,
            n_bins=12
        )
        
        assert detector.warning_threshold == 0.15
        assert detector.alert_threshold == 0.25
        assert detector.n_pca_components == 8
        assert detector.n_bins == 12
        
        # Verificar que los umbrales globales se actualizaron
        assert PSI_THRESHOLDS["warning"] == 0.15
        assert PSI_THRESHOLDS["alert"] == 0.25

    def test_drift_detector_default_initialization(self):
        """Test inicialización con valores por defecto."""
        detector = DriftDetector()
        
        # Verificar valores por defecto de config
        from src.config import PSI_DRIFT_WARN_THRESHOLD, PSI_DRIFT_ALERT_THRESHOLD
        
        assert detector.warning_threshold == PSI_DRIFT_WARN_THRESHOLD
        assert detector.alert_threshold == PSI_DRIFT_ALERT_THRESHOLD
        assert detector.n_pca_components == 10
        assert detector.n_bins == 10

    def test_detect_feature_drift_stable_case(self):
        """Test detección de drift estable en features."""
        np.random.seed(RANDOM_STATE)
        
        # Features muy similares (drift mínimo)
        ref_features = np.random.normal(0, 1, (200, 5)).astype(np.float32)
        cur_features = ref_features + np.random.normal(0, 0.01, ref_features.shape)  # Ruido mínimo
        
        detector = DriftDetector()
        result = detector.detect_feature_drift(ref_features, cur_features)
        
        assert result["overall_status"] == "stable"
        assert result["mean_psi"] < 0.1
        assert result["drift_counts"]["stable"] >= 3  # Mayoría estables
        assert result["drift_counts"]["alert"] == 0   # Sin alerts

    def test_detect_feature_drift_alert_case(self):
        """Test detección de drift con alert en features."""
        np.random.seed(RANDOM_STATE)
        
        # Features con drift significativo
        ref_features = np.random.normal(0, 1, (150, 4)).astype(np.float32)
        cur_features = np.random.normal(3, 2, (150, 4)).astype(np.float32)  # Muy diferentes
        
        detector = DriftDetector(alert_threshold=0.20)
        result = detector.detect_feature_drift(ref_features, cur_features)
        
        assert result["overall_status"] == "alert"
        assert result["max_psi"] > 0.20
        assert result["drift_counts"]["alert"] > 0
        assert len(result["top_drift_features"]) <= 5

    def test_detect_feature_drift_with_feature_names(self):
        """Test detección con nombres de features personalizados."""
        np.random.seed(RANDOM_STATE)
        
        ref_features = np.random.normal(0, 1, (100, 3)).astype(np.float32)
        cur_features = np.random.normal(0.5, 1, (100, 3)).astype(np.float32)
        
        feature_names = ["valor_facturado", "codigo_cups", "dias_estancia"]
        
        detector = DriftDetector()
        result = detector.detect_feature_drift(ref_features, cur_features, feature_names)
        
        # Verificar que se usan los nombres personalizados
        assert "valor_facturado" in result["psi_per_feature"]
        assert "codigo_cups" in result["drift_status_per_feature"]
        assert len(result["psi_per_feature"]) == 3
        
        # Verificar estructura de top_drift_features
        for feature_info in result["top_drift_features"]:
            assert feature_info["name"] in feature_names
            assert "psi" in feature_info
            assert "status" in feature_info

    def test_detect_feature_drift_dimension_mismatch(self):
        """Test error cuando dimensiones de features no coinciden."""
        ref_features = np.random.normal(0, 1, (100, 5)).astype(np.float32)
        cur_features = np.random.normal(0, 1, (100, 7)).astype(np.float32)  # Dimensión diferente
        
        detector = DriftDetector()
        
        with pytest.raises(ValueError, match="Número de features no coincide"):
            detector.detect_feature_drift(ref_features, cur_features)

    def test_detect_embedding_drift_stable_case(self):
        """Test detección de drift estable en embeddings."""
        np.random.seed(RANDOM_STATE)
        
        # Embeddings similares
        ref_embeddings = np.random.normal(0, 1, (80, 12)).astype(np.float32)
        cur_embeddings = ref_embeddings + np.random.normal(0, 0.05, ref_embeddings.shape)
        
        detector = DriftDetector()
        result = detector.detect_embedding_drift(ref_embeddings, cur_embeddings)
        
        assert result["overall_status"] in ["stable", "warning"]  # Puede ser warning por ruido
        assert result["mean_psi"] < 0.3
        assert "n_components_analyzed" in result
        assert result["total_variance_explained"] > 0.5

    def test_detect_embedding_drift_alert_case(self):
        """Test detección de drift con alert en embeddings."""
        np.random.seed(RANDOM_STATE)
        
        # Embeddings muy diferentes
        ref_embeddings = np.random.normal(0, 1, (60, 10)).astype(np.float32)
        cur_embeddings = np.random.normal(2, 1.5, (60, 10)).astype(np.float32)
        
        detector = DriftDetector(alert_threshold=0.15)
        result = detector.detect_embedding_drift(ref_embeddings, cur_embeddings)
        
        assert result["overall_status"] == "alert"
        assert result["max_psi"] > 0.15
        assert len(result["top_drift_components"]) <= 3
        
        # Verificar estructura de componentes problemáticos
        for comp_info in result["top_drift_components"]:
            assert "component" in comp_info
            assert "psi" in comp_info
            assert "status" in comp_info
            assert "variance_explained" in comp_info

    def test_detect_embedding_drift_error_handling(self):
        """Test manejo de errores en detección de embeddings."""
        detector = DriftDetector()
        
        # Simular error en compute_embedding_psi
        with patch('src.monitoring.drift_detector.compute_embedding_psi') as mock_compute:
            mock_compute.side_effect = Exception("Test error")
            
            ref_embeddings = np.random.normal(0, 1, (50, 8)).astype(np.float32)
            cur_embeddings = np.random.normal(0, 1, (50, 8)).astype(np.float32)
            
            result = detector.detect_embedding_drift(ref_embeddings, cur_embeddings)
            
            assert result["overall_status"] == "error"
            assert "error" in result
            assert result["mean_psi"] == 0.0

    def test_detect_comprehensive_drift_features_only(self):
        """Test análisis comprehensivo solo con features."""
        np.random.seed(RANDOM_STATE)
        
        ref_features = np.random.normal(0, 1, (100, 6)).astype(np.float32)
        cur_features = np.random.normal(0.3, 1, (100, 6)).astype(np.float32)
        
        detector = DriftDetector()
        result = detector.detect_comprehensive_drift(ref_features, cur_features)
        
        assert "feature_drift" in result
        assert result["embedding_drift"] is None
        assert result["overall_status"] in ["stable", "warning", "alert"]
        assert "recommendations" in result
        assert len(result["recommendations"]) > 0
        assert "timestamp" in result

    def test_detect_comprehensive_drift_with_embeddings(self):
        """Test análisis comprehensivo con features y embeddings."""
        np.random.seed(RANDOM_STATE)
        
        # Features con drift moderado
        ref_features = np.random.normal(0, 1, (80, 4)).astype(np.float32)
        cur_features = np.random.normal(0.5, 1, (80, 4)).astype(np.float32)
        
        # Embeddings con drift alto
        ref_embeddings = np.random.normal(0, 1, (80, 8)).astype(np.float32)
        cur_embeddings = np.random.normal(2, 1, (80, 8)).astype(np.float32)
        
        detector = DriftDetector(alert_threshold=0.15)
        result = detector.detect_comprehensive_drift(
            ref_features, cur_features,
            ref_embeddings, cur_embeddings
        )
        
        assert result["feature_drift"] is not None
        assert result["embedding_drift"] is not None
        assert result["overall_status"] == "alert"  # Embeddings tienen drift alto
        
        # Verificar summary
        summary = result["summary"]
        assert "requires_retraining" in summary
        assert summary["requires_retraining"] is True  # Alert requiere reentrenamiento
        assert "embedding_max_psi" in summary
        assert summary["embedding_max_psi"] > 0

    def test_comprehensive_drift_recommendations_alert(self):
        """Test recomendaciones para nivel alert."""
        np.random.seed(RANDOM_STATE)
        
        # Crear drift severo
        ref_features = np.random.normal(0, 1, (50, 3)).astype(np.float32)
        cur_features = np.random.normal(5, 2, (50, 3)).astype(np.float32)
        
        detector = DriftDetector(alert_threshold=0.10)
        result = detector.detect_comprehensive_drift(ref_features, cur_features)
        
        recommendations = result["recommendations"]
        
        # Verificar recomendaciones de alert
        rec_text = " ".join(recommendations).lower()
        assert "acción requerida" in rec_text or "reentrenamiento" in rec_text
        assert any("🚨" in rec for rec in recommendations)

    def test_comprehensive_drift_recommendations_stable(self):
        """Test recomendaciones para nivel stable."""
        np.random.seed(RANDOM_STATE)
        
        # Crear datos muy similares
        ref_features = np.random.normal(0, 1, (50, 3)).astype(np.float32)
        cur_features = ref_features + np.random.normal(0, 0.01, ref_features.shape)
        
        detector = DriftDetector()
        result = detector.detect_comprehensive_drift(ref_features, cur_features)
        
        recommendations = result["recommendations"]
        
        # Verificar recomendaciones de stable
        rec_text = " ".join(recommendations).lower()
        assert "estable" in rec_text or "continuar" in rec_text
        assert any("✅" in rec for rec in recommendations)

    def test_get_configuration_method(self):
        """Test método get_configuration."""
        detector = DriftDetector(
            warning_threshold=0.12,
            alert_threshold=0.22,
            n_pca_components=7,
            n_bins=8
        )
        
        config = detector.get_configuration()
        
        assert config["warning_threshold"] == 0.12
        assert config["alert_threshold"] == 0.22
        assert config["n_pca_components"] == 7
        assert config["n_bins"] == 8
        assert "psi_thresholds" in config
        assert config["psi_thresholds"]["warning"] == 0.12

    def test_determine_overall_status_method(self):
        """Test método _determine_overall_status."""
        detector = DriftDetector()
        
        # Casos de prueba
        assert detector._determine_overall_status(["stable", "stable", "stable"]) == "stable"
        assert detector._determine_overall_status(["stable", "warning", "stable"]) == "warning"
        assert detector._determine_overall_status(["stable", "warning", "alert"]) == "alert"
        assert detector._determine_overall_status(["alert", "alert", "alert"]) == "alert"

    def test_generate_recommendations_method(self):
        """Test método _generate_recommendations."""
        detector = DriftDetector()
        
        # Mock feature_drift y embedding_drift
        feature_drift_alert = {
            "drift_counts": {"alert": 8, "warning": 2, "stable": 0},
            "max_psi": 0.35
        }
        
        embedding_drift_alert = {
            "max_psi": 0.45,
            "mean_psi": 0.25
        }
        
        recommendations = detector._generate_recommendations(
            "alert", feature_drift_alert, embedding_drift_alert
        )
        
        assert len(recommendations) > 0
        assert any("ACCIÓN REQUERIDA" in rec for rec in recommendations)
        assert any("reentrenamiento" in rec.lower() for rec in recommendations)

    def test_psi_calculation_robustness(self):
        """Test robustez del cálculo PSI con datos problemáticos."""
        # Datos con outliers extremos
        ref_data = np.concatenate([
            np.random.normal(0, 1, 95),
            np.array([100, -100, 1000, -1000, 50])  # Outliers
        ])
        cur_data = np.concatenate([
            np.random.normal(0.2, 1, 95),
            np.array([110, -90, 900, -900, 60])  # Outliers similares
        ])
        
        psi = compute_psi(ref_data, cur_data)
        
        assert np.isfinite(psi), "PSI debe ser finito incluso con outliers"
        assert psi >= 0, "PSI debe ser no negativo con outliers"
        assert psi < 50, f"PSI no debe ser excesivamente alto con outliers: {psi}"

    def test_embedding_psi_small_components(self):
        """Test PSI de embeddings con pocos componentes PCA."""
        np.random.seed(RANDOM_STATE)
        
        # Embeddings de alta dimensión pero pocos samples (fuerza pocos componentes PCA)
        ref_embeddings = np.random.normal(0, 1, (10, 50)).astype(np.float32)
        cur_embeddings = np.random.normal(0.5, 1, (10, 50)).astype(np.float32)
        
        result = compute_embedding_psi(ref_embeddings, cur_embeddings, n_components=5)
        
        # Debe usar menos componentes debido a limitación de samples
        assert result["n_components_used"] <= 9  # min(50, 10-1)
        assert len(result["psi_per_component"]) == result["n_components_used"]
        assert result["total_variance_explained"] > 0