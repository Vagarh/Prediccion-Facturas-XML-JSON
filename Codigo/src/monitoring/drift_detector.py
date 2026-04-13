# -*- coding: utf-8 -*-
"""Detector de concept drift usando Population Stability Index (PSI).

Monitorea cambios en las distribuciones de features y embeddings para detectar
concept drift que pueda afectar el rendimiento del modelo de auditoría.

Cumplimiento:
    - PSI para features numéricas y embeddings
    - Reducción dimensional con PCA para embeddings
    - Umbrales configurables para warnings y alerts
    - Logging detallado de métricas de drift
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from scipy import stats

from src.config import (
    PSI_DRIFT_WARN_THRESHOLD,
    PSI_DRIFT_ALERT_THRESHOLD,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)

# Umbrales PSI estándar de la industria
PSI_THRESHOLDS = {
    "stable": 0.10,      # PSI < 0.10: Sin cambio significativo
    "warning": PSI_DRIFT_WARN_THRESHOLD,  # 0.10 ≤ PSI < 0.20: Cambio moderado
    "alert": PSI_DRIFT_ALERT_THRESHOLD,   # PSI ≥ 0.20: Cambio significativo
}


def compute_psi(
    reference: np.ndarray, 
    current: np.ndarray, 
    n_bins: int = 10
) -> float:
    """Calcula Population Stability Index entre dos distribuciones.

    El PSI mide la diferencia entre dos distribuciones usando la fórmula:
    PSI = Σ (actual_% - expected_%) * ln(actual_% / expected_%)

    Args:
        reference: Array de referencia (baseline).
        current: Array actual para comparar.
        n_bins: Número de bins para discretización.

    Returns:
        Valor PSI (0 = idénticas, >0.20 = cambio significativo).

    Raises:
        ValueError: Si los arrays están vacíos o tienen dimensiones incompatibles.
    """
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("Arrays no pueden estar vacíos")
    
    if reference.ndim != 1 or current.ndim != 1:
        raise ValueError("Arrays deben ser unidimensionales")
    
    # Determinar bins basados en la distribución de referencia
    # Usar percentiles para bins más robustos
    try:
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(reference, percentiles)
        
        # Asegurar que los edges son únicos
        bin_edges = np.unique(bin_edges)
        
        # Si hay muy pocos valores únicos, usar bins simples
        if len(bin_edges) < 3:
            bin_edges = np.linspace(
                min(reference.min(), current.min()),
                max(reference.max(), current.max()),
                n_bins + 1
            )
    
    except Exception as e:
        logger.warning("Error calculando percentiles, usando bins lineales: %s", e)
        bin_edges = np.linspace(
            min(reference.min(), current.min()),
            max(reference.max(), current.max()),
            n_bins + 1
        )
    
    # Calcular histogramas
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)
    
    # Convertir a proporciones
    ref_props = ref_counts / len(reference)
    cur_props = cur_counts / len(current)
    
    # Evitar divisiones por cero agregando pequeña constante
    epsilon = 1e-8
    ref_props = np.maximum(ref_props, epsilon)
    cur_props = np.maximum(cur_props, epsilon)
    
    # Calcular PSI
    psi_values = (cur_props - ref_props) * np.log(cur_props / ref_props)
    psi = np.sum(psi_values)
    
    logger.debug(
        "PSI calculado: %.4f (bins=%d, ref_samples=%d, cur_samples=%d)",
        psi, len(bin_edges) - 1, len(reference), len(current)
    )
    
    return float(psi)


def compute_embedding_psi(
    ref_embeddings: np.ndarray, 
    cur_embeddings: np.ndarray,
    n_components: int = 10,
    n_bins: int = 10
) -> dict[str, float]:
    """Calcula PSI para embeddings usando reducción dimensional PCA.

    Args:
        ref_embeddings: Embeddings de referencia (n_samples, embedding_dim).
        cur_embeddings: Embeddings actuales (n_samples, embedding_dim).
        n_components: Número de componentes PCA para reducción dimensional.
        n_bins: Número de bins para PSI.

    Returns:
        Diccionario con métricas PSI:
        - mean_psi: PSI promedio across componentes
        - max_psi: PSI máximo de cualquier componente
        - psi_per_component: PSI por cada componente PCA
        - explained_variance_ratio: Varianza explicada por componentes

    Raises:
        ValueError: Si las dimensiones de embeddings no coinciden.
    """
    if ref_embeddings.shape[1] != cur_embeddings.shape[1]:
        raise ValueError(
            f"Dimensiones de embeddings no coinciden: "
            f"{ref_embeddings.shape[1]} vs {cur_embeddings.shape[1]}"
        )
    
    if ref_embeddings.shape[0] == 0 or cur_embeddings.shape[0] == 0:
        raise ValueError("Embeddings no pueden estar vacíos")
    
    embedding_dim = ref_embeddings.shape[1]
    n_components = min(n_components, embedding_dim, ref_embeddings.shape[0] - 1)
    
    logger.debug(
        "Calculando PSI para embeddings: ref=%s, cur=%s, n_components=%d",
        ref_embeddings.shape, cur_embeddings.shape, n_components
    )
    
    # Aplicar PCA para reducción dimensional
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    
    # Fit PCA en embeddings de referencia
    ref_pca = pca.fit_transform(ref_embeddings)
    
    # Transform embeddings actuales usando el mismo PCA
    cur_pca = pca.transform(cur_embeddings)
    
    # Calcular PSI para cada componente PCA
    psi_per_component = []
    for i in range(n_components):
        try:
            component_psi = compute_psi(ref_pca[:, i], cur_pca[:, i], n_bins=n_bins)
            psi_per_component.append(component_psi)
        except Exception as e:
            logger.warning("Error calculando PSI para componente %d: %s", i, e)
            psi_per_component.append(0.0)
    
    psi_per_component = np.array(psi_per_component)
    
    # Métricas agregadas
    mean_psi = float(np.mean(psi_per_component))
    max_psi = float(np.max(psi_per_component))
    
    result = {
        "mean_psi": mean_psi,
        "max_psi": max_psi,
        "psi_per_component": psi_per_component.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "n_components_used": n_components,
        "total_variance_explained": float(np.sum(pca.explained_variance_ratio_)),
    }
    
    logger.info(
        "PSI embeddings: mean=%.4f, max=%.4f, variance_explained=%.3f",
        mean_psi, max_psi, result["total_variance_explained"]
    )
    
    return result


def check_drift(psi_value: float) -> str:
    """Determina el nivel de drift basado en el valor PSI.

    Args:
        psi_value: Valor PSI calculado.

    Returns:
        Nivel de drift: "stable", "warning", o "alert".
    """
    if psi_value < PSI_THRESHOLDS["stable"]:
        return "stable"
    elif psi_value < PSI_THRESHOLDS["warning"]:
        return "warning"
    else:
        return "alert"


class DriftDetector:
    """Detector de concept drift para el pipeline de auditoría prestacional."""

    def __init__(
        self,
        warning_threshold: float = PSI_DRIFT_WARN_THRESHOLD,
        alert_threshold: float = PSI_DRIFT_ALERT_THRESHOLD,
        n_pca_components: int = 10,
        n_bins: int = 10,
    ) -> None:
        """Inicializa el detector de drift.

        Args:
            warning_threshold: Umbral PSI para warnings.
            alert_threshold: Umbral PSI para alerts.
            n_pca_components: Componentes PCA para embeddings.
            n_bins: Número de bins para discretización PSI.
        """
        self.warning_threshold = warning_threshold
        self.alert_threshold = alert_threshold
        self.n_pca_components = n_pca_components
        self.n_bins = n_bins
        
        # Actualizar umbrales globales
        PSI_THRESHOLDS["warning"] = warning_threshold
        PSI_THRESHOLDS["alert"] = alert_threshold
        
        logger.info(
            "DriftDetector inicializado: warning=%.3f, alert=%.3f, pca_components=%d",
            warning_threshold, alert_threshold, n_pca_components
        )

    def detect_feature_drift(
        self,
        reference_features: np.ndarray,
        current_features: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Detecta drift en features numéricas.

        Args:
            reference_features: Features de referencia (n_samples, n_features).
            current_features: Features actuales (n_samples, n_features).
            feature_names: Nombres de las features (opcional).

        Returns:
            Diccionario con resultados de detección de drift.
        """
        if reference_features.shape[1] != current_features.shape[1]:
            raise ValueError("Número de features no coincide")
        
        n_features = reference_features.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        logger.info(
            "Detectando drift en %d features: ref=%d samples, cur=%d samples",
            n_features, len(reference_features), len(current_features)
        )
        
        # Calcular PSI por feature
        psi_per_feature = []
        drift_status_per_feature = []
        
        for i in range(n_features):
            try:
                feature_psi = compute_psi(
                    reference_features[:, i],
                    current_features[:, i],
                    n_bins=self.n_bins
                )
                psi_per_feature.append(feature_psi)
                drift_status_per_feature.append(check_drift(feature_psi))
                
            except Exception as e:
                logger.warning("Error calculando PSI para feature %s: %s", feature_names[i], e)
                psi_per_feature.append(0.0)
                drift_status_per_feature.append("stable")
        
        # Métricas agregadas
        psi_per_feature = np.array(psi_per_feature)
        mean_psi = float(np.mean(psi_per_feature))
        max_psi = float(np.max(psi_per_feature))
        
        # Conteo por nivel de drift
        drift_counts = {
            "stable": drift_status_per_feature.count("stable"),
            "warning": drift_status_per_feature.count("warning"),
            "alert": drift_status_per_feature.count("alert"),
        }
        
        # Estado general
        overall_status = self._determine_overall_status(drift_status_per_feature)
        
        # Features más problemáticas
        top_drift_indices = np.argsort(psi_per_feature)[-5:][::-1]  # Top 5
        top_drift_features = [
            {
                "name": feature_names[i],
                "psi": float(psi_per_feature[i]),
                "status": drift_status_per_feature[i]
            }
            for i in top_drift_indices
        ]
        
        result = {
            "overall_status": overall_status,
            "mean_psi": mean_psi,
            "max_psi": max_psi,
            "drift_counts": drift_counts,
            "psi_per_feature": {
                feature_names[i]: float(psi_per_feature[i])
                for i in range(n_features)
            },
            "drift_status_per_feature": {
                feature_names[i]: drift_status_per_feature[i]
                for i in range(n_features)
            },
            "top_drift_features": top_drift_features,
            "thresholds": {
                "warning": self.warning_threshold,
                "alert": self.alert_threshold
            },
        }
        
        logger.info(
            "Drift features: status=%s, mean_psi=%.4f, max_psi=%.4f, alerts=%d",
            overall_status, mean_psi, max_psi, drift_counts["alert"]
        )
        
        return result

    def detect_embedding_drift(
        self,
        reference_embeddings: np.ndarray,
        current_embeddings: np.ndarray,
    ) -> dict[str, Any]:
        """Detecta drift en embeddings semánticos.

        Args:
            reference_embeddings: Embeddings de referencia.
            current_embeddings: Embeddings actuales.

        Returns:
            Diccionario con resultados de detección de drift en embeddings.
        """
        logger.info(
            "Detectando drift en embeddings: ref=%s, cur=%s",
            reference_embeddings.shape, current_embeddings.shape
        )
        
        try:
            # Calcular PSI para embeddings con PCA
            psi_results = compute_embedding_psi(
                reference_embeddings,
                current_embeddings,
                n_components=self.n_pca_components,
                n_bins=self.n_bins
            )
            
            # Determinar estado de drift
            mean_status = check_drift(psi_results["mean_psi"])
            max_status = check_drift(psi_results["max_psi"])
            
            # Estado general (usar el más severo)
            overall_status = max_status if max_status != "stable" else mean_status
            
            # Componentes más problemáticos
            psi_components = np.array(psi_results["psi_per_component"])
            top_component_indices = np.argsort(psi_components)[-3:][::-1]  # Top 3
            
            top_drift_components = [
                {
                    "component": int(i),
                    "psi": float(psi_components[i]),
                    "status": check_drift(psi_components[i]),
                    "variance_explained": float(psi_results["explained_variance_ratio"][i])
                }
                for i in top_component_indices
            ]
            
            result = {
                "overall_status": overall_status,
                "mean_psi": psi_results["mean_psi"],
                "max_psi": psi_results["max_psi"],
                "mean_status": mean_status,
                "max_status": max_status,
                "n_components_analyzed": psi_results["n_components_used"],
                "total_variance_explained": psi_results["total_variance_explained"],
                "psi_per_component": psi_results["psi_per_component"],
                "explained_variance_ratio": psi_results["explained_variance_ratio"],
                "top_drift_components": top_drift_components,
                "thresholds": {
                    "warning": self.warning_threshold,
                    "alert": self.alert_threshold
                },
            }
            
            logger.info(
                "Drift embeddings: status=%s, mean_psi=%.4f, max_psi=%.4f, variance=%.3f",
                overall_status, psi_results["mean_psi"], psi_results["max_psi"],
                psi_results["total_variance_explained"]
            )
            
            return result
            
        except Exception as e:
            logger.error("Error detectando drift en embeddings: %s", e)
            return {
                "overall_status": "error",
                "error": str(e),
                "mean_psi": 0.0,
                "max_psi": 0.0,
            }

    def detect_comprehensive_drift(
        self,
        reference_features: np.ndarray,
        current_features: np.ndarray,
        reference_embeddings: np.ndarray | None = None,
        current_embeddings: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Detecta drift comprehensivo en features y embeddings.

        Args:
            reference_features: Features numéricas de referencia.
            current_features: Features numéricas actuales.
            reference_embeddings: Embeddings de referencia (opcional).
            current_embeddings: Embeddings actuales (opcional).
            feature_names: Nombres de features (opcional).

        Returns:
            Diccionario con análisis completo de drift.
        """
        logger.info("Iniciando detección comprehensiva de drift")
        
        # Detectar drift en features numéricas
        feature_drift = self.detect_feature_drift(
            reference_features, current_features, feature_names
        )
        
        # Detectar drift en embeddings si están disponibles
        embedding_drift = None
        if (reference_embeddings is not None and 
            current_embeddings is not None):
            embedding_drift = self.detect_embedding_drift(
                reference_embeddings, current_embeddings
            )
        
        # Determinar estado general
        overall_status = feature_drift["overall_status"]
        if embedding_drift and embedding_drift["overall_status"] == "alert":
            overall_status = "alert"
        elif embedding_drift and embedding_drift["overall_status"] == "warning" and overall_status == "stable":
            overall_status = "warning"
        
        # Recomendaciones basadas en el nivel de drift
        recommendations = self._generate_recommendations(
            overall_status, feature_drift, embedding_drift
        )
        
        result = {
            "timestamp": self._get_timestamp(),
            "overall_status": overall_status,
            "feature_drift": feature_drift,
            "embedding_drift": embedding_drift,
            "recommendations": recommendations,
            "summary": {
                "features_with_alerts": feature_drift["drift_counts"]["alert"],
                "features_with_warnings": feature_drift["drift_counts"]["warning"],
                "max_feature_psi": feature_drift["max_psi"],
                "embedding_max_psi": embedding_drift["max_psi"] if embedding_drift else None,
                "requires_retraining": overall_status == "alert",
            }
        }
        
        logger.info(
            "Drift comprehensivo: status=%s, feature_alerts=%d, embedding_psi=%.4f",
            overall_status, 
            feature_drift["drift_counts"]["alert"],
            embedding_drift["max_psi"] if embedding_drift else 0.0
        )
        
        return result

    def _determine_overall_status(self, drift_statuses: list[str]) -> str:
        """Determina estado general basado en estados individuales."""
        if "alert" in drift_statuses:
            return "alert"
        elif "warning" in drift_statuses:
            return "warning"
        else:
            return "stable"

    def _generate_recommendations(
        self,
        overall_status: str,
        feature_drift: dict[str, Any],
        embedding_drift: dict[str, Any] | None,
    ) -> list[str]:
        """Genera recomendaciones basadas en el nivel de drift detectado."""
        recommendations = []
        
        if overall_status == "alert":
            recommendations.extend([
                "🚨 ACCIÓN REQUERIDA: Concept drift significativo detectado",
                "Programar reentrenamiento del modelo inmediatamente",
                "Revisar calidad de datos recientes",
                "Validar pipeline de preprocesamiento",
            ])
            
            if feature_drift["drift_counts"]["alert"] > 5:
                recommendations.append(
                    f"Investigar {feature_drift['drift_counts']['alert']} features con drift severo"
                )
            
            if embedding_drift and embedding_drift["max_psi"] > 0.30:
                recommendations.append(
                    "Drift severo en embeddings: revisar preprocesamiento de texto"
                )
        
        elif overall_status == "warning":
            recommendations.extend([
                "⚠️ MONITOREO: Drift moderado detectado",
                "Aumentar frecuencia de monitoreo",
                "Preparar datos para reentrenamiento preventivo",
                "Revisar features con mayor drift",
            ])
            
            if embedding_drift and embedding_drift["mean_psi"] > 0.15:
                recommendations.append(
                    "Monitorear embeddings: posible cambio en patrones textuales"
                )
        
        else:
            recommendations.extend([
                "✅ ESTABLE: No se detectó drift significativo",
                "Continuar monitoreo regular",
                "Mantener modelo actual en producción",
            ])
        
        return recommendations

    def _get_timestamp(self) -> str:
        """Obtiene timestamp actual en formato ISO."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"

    def get_configuration(self) -> dict[str, Any]:
        """Obtiene configuración actual del detector.

        Returns:
            Diccionario con parámetros de configuración.
        """
        return {
            "warning_threshold": self.warning_threshold,
            "alert_threshold": self.alert_threshold,
            "n_pca_components": self.n_pca_components,
            "n_bins": self.n_bins,
            "psi_thresholds": PSI_THRESHOLDS.copy(),
        }