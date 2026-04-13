# -*- coding: utf-8 -*-
"""Tests unitarios para ExplanationGenerator.

Verifica funcionalidad específica del generador de explicaciones:
- Narrativas para clases Glosada/Devuelta incluyen causales Res. 2284/2023
- Narrativas no contienen términos técnicos de ML
- Fallback SHAP se activa cuando Ollama no responde
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from requests.exceptions import RequestException, Timeout
import json

from src.explainability.explanation_generator import (
    ExplanationGenerator,
    FEATURE_TO_CAUSAL,
    EXPLANATION_TEMPLATES
)
from src.privacy.guard import PrivacyGuard
from src.config import EXPLANATION_MAX_SECONDS


class TestExplanationGenerator:
    """Tests unitarios para ExplanationGenerator."""

    def setup_method(self):
        """Configuración inicial para cada test."""
        # Mock del PrivacyGuard
        self.mock_privacy_guard = Mock(spec=PrivacyGuard)
        self.mock_privacy_guard.verify_local_only.return_value = None
        self.mock_privacy_guard.check_ollama_available.return_value = True
        self.mock_privacy_guard.log_inference.return_value = None

    def test_initialization_with_ollama_available(self):
        """Test inicialización cuando Ollama está disponible."""
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="llama3.2",
            timeout_s=10.0
        )
        
        assert generator.llm_model == "llama3.2"
        assert generator.timeout_s == 10.0
        assert generator.ollama_available is True
        
        # Verificar que se llamaron los métodos de verificación
        self.mock_privacy_guard.verify_local_only.assert_called_once()
        self.mock_privacy_guard.check_ollama_available.assert_called_once()

    def test_initialization_with_ollama_unavailable(self):
        """Test inicialización cuando Ollama no está disponible."""
        self.mock_privacy_guard.check_ollama_available.return_value = False
        
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="mistral",
            timeout_s=5.0
        )
        
        assert generator.ollama_available is False
        assert generator.llm_model == "mistral"

    def test_initialization_with_privacy_violation(self):
        """Test inicialización cuando hay violación de privacidad."""
        self.mock_privacy_guard.verify_local_only.side_effect = Exception("Host externo detectado")
        
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model"
        )
        
        assert generator.ollama_available is False

    def test_generate_fallback_explanation_glosada_includes_causals(self):
        """Test que explicación fallback para Glosada incluye causales Res. 2284/2023."""
        self.mock_privacy_guard.check_ollama_available.return_value = False
        
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model"
        )
        
        # SHAP values con features que tienen causales mapeados
        shap_values = {
            "valor_facturado": 0.15,  # SO0101
            "codigo_cups": -0.12,     # FA0101
            "numero_autorizacion": 0.08  # DE16
        }
        
        feature_values = {
            "valor_facturado": 150000,
            "codigo_cups": "890201",
            "numero_autorizacion": "AUT123456"
        }
        
        result = generator.generate(
            prediction=1,  # Glosada
            confidence=0.85,
            shap_values=shap_values,
            feature_values=feature_values
        )
        
        # Verificar que incluye causales específicos
        assert "SO0101" in result or "Facturación por servicios no prestados" in result
        assert "FA0101" in result or "servicios no incluidos en el plan" in result
        assert "DE16" in result or "Falta de autorización" in result
        
        # Verificar estructura de explicación
        assert "GLOSADA" in result
        assert "Confianza: 85.0%" in result
        assert len(result.strip()) > 100  # Explicación sustancial

    def test_generate_fallback_explanation_devuelta_includes_causals(self):
        """Test que explicación fallback para Devuelta incluye causales Res. 2284/2023."""
        self.mock_privacy_guard.check_ollama_available.return_value = False
        
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model"
        )
        
        # SHAP values con diferentes causales
        shap_values = {
            "valor_unitario": 0.20,      # TA0101
            "tipo_documento": -0.15,     # DE01
            "fecha_autorizacion": 0.10   # DE16
        }
        
        feature_values = {
            "valor_unitario": 50000,
            "tipo_documento": "FACTURA",
            "fecha_autorizacion": "2024-01-15"
        }
        
        result = generator.generate(
            prediction=2,  # Devuelta
            confidence=0.92,
            shap_values=shap_values,
            feature_values=feature_values
        )
        
        # Verificar causales específicos
        assert "TA0101" in result or "tarifas superiores" in result
        assert "DE01" in result or "Ausencia de la factura" in result
        assert "DE16" in result or "Falta de autorización" in result
        
        # Verificar estructura
        assert "DEVUELTA" in result
        assert "Confianza: 92.0%" in result

    def test_generate_explanation_no_ml_terms(self):
        """Test que explicaciones no contienen términos técnicos de ML."""
        self.mock_privacy_guard.check_ollama_available.return_value = False
        
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model"
        )
        
        # Features con nombres que podrían contener términos ML
        shap_values = {
            "shap_importance": 0.1,
            "embedding_similarity": -0.2,
            "model_confidence": 0.15
        }
        
        feature_values = {
            "shap_importance": 0.85,
            "embedding_similarity": 0.72,
            "model_confidence": 0.91
        }
        
        result = generator.generate(
            prediction=1,
            confidence=0.80,
            shap_values=shap_values,
            feature_values=feature_values
        )
        
        # Verificar que no contiene términos ML
        ml_terms = ["shap", "embedding", "feature importance", "machine learning", "modelo", "algoritmo"]
        result_lower = result.lower()
        
        for term in ml_terms:
            assert term not in result_lower, f"Explicación no debe contener '{term}'"
        
        # Verificar que contiene términos apropiados
        assert "factor de análisis" in result_lower or "factor" in result_lower

    def test_fallback_activation_when_ollama_not_responding(self):
        """Test que fallback SHAP se activa cuando Ollama no responde."""
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model",
            timeout_s=1.0
        )
        
        # Mock requests.post para simular fallo de conexión
        with patch('requests.post') as mock_post:
            mock_post.side_effect = RequestException("Connection refused")
            
            result = generator.generate(
                prediction=1,
                confidence=0.75,
                shap_values={"test_feature": 0.1},
                feature_values={"test_feature": "test_value"}
            )
            
            # Verificar que se usó fallback
            assert isinstance(result, str)
            assert len(result.strip()) > 0
            assert "GLOSADA" in result
            assert "test_feature" in result
            
            # Verificar que se intentó llamar a Ollama
            mock_post.assert_called_once()

    def test_fallback_activation_on_timeout(self):
        """Test que fallback se activa cuando se excede timeout."""
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model",
            timeout_s=0.1  # Timeout muy corto
        )
        
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Timeout("Request timed out")
            
            result = generator.generate(
                prediction=2,
                confidence=0.88,
                shap_values={"timeout_feature": -0.2},
                feature_values={"timeout_feature": 12345}
            )
            
            # Verificar fallback exitoso
            assert "DEVUELTA" in result
            assert "timeout_feature" in result
            assert len(result.strip()) > 50

    def test_successful_llm_generation(self):
        """Test generación exitosa con LLM."""
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="llama3.2",
            timeout_s=5.0
        )
        
        # Mock respuesta exitosa de Ollama
        mock_response = {
            "response": "La factura presenta glosas debido a inconsistencias en la facturación. "
                       "Los factores principales incluyen el valor facturado que excede los límites "
                       "establecidos y problemas con la autorización requerida."
        }
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status.return_value = None
            
            result = generator.generate(
                prediction=1,
                confidence=0.82,
                shap_values={"valor_facturado": 0.15, "numero_autorizacion": 0.08},
                feature_values={"valor_facturado": 200000, "numero_autorizacion": "AUT789"}
            )
            
            # Verificar que se usó respuesta LLM
            assert "glosas debido a inconsistencias" in result
            assert "GLOSADA" in result
            assert "Confianza: 82.0%" in result
            
            # Verificar que se llamó a log_inference
            self.mock_privacy_guard.log_inference.assert_called_once()

    def test_empty_llm_response_fallback(self):
        """Test fallback cuando LLM retorna respuesta vacía."""
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model"
        )
        
        # Mock respuesta vacía de Ollama
        mock_response = {"response": ""}
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status.return_value = None
            
            result = generator.generate(
                prediction=0,
                confidence=0.95,
                shap_values={"feature1": 0.05},
                feature_values={"feature1": "value1"}
            )
            
            # Debe usar fallback por respuesta vacía
            assert "AUDITADA" in result
            assert "feature1" in result
            assert len(result.strip()) > 0

    def test_get_top_factors_ordering(self):
        """Test que _get_top_factors ordena correctamente por importancia."""
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model"
        )
        
        shap_values = {
            "low_impact": 0.02,
            "high_impact": -0.25,
            "medium_impact": 0.15,
            "zero_impact": 0.0
        }
        
        feature_values = {
            "low_impact": "low",
            "high_impact": "high", 
            "medium_impact": "medium",
            "zero_impact": "zero"
        }
        
        top_factors = generator._get_top_factors(shap_values, feature_values, n=3)
        
        # Verificar orden por valor absoluto (importancia)
        assert len(top_factors) == 3
        assert top_factors[0][0] == "high_impact"  # |−0.25| = 0.25
        assert top_factors[1][0] == "medium_impact"  # |0.15| = 0.15
        assert top_factors[2][0] == "low_impact"  # |0.02| = 0.02
        
        # Verificar estructura de tuplas
        for feature_name, shap_value, feature_value in top_factors:
            assert isinstance(feature_name, str)
            assert isinstance(shap_value, float)
            assert feature_value == feature_values[feature_name]

    def test_get_status_method(self):
        """Test método get_status retorna información correcta."""
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="qwen2.5",
            timeout_s=7.5
        )
        
        status = generator.get_status()
        
        # Verificar estructura del status
        assert isinstance(status, dict)
        assert status["ollama_available"] is True
        assert status["llm_model"] == "qwen2.5"
        assert status["timeout_s"] == 7.5
        assert "ollama_base_url" in status
        assert status["feature_mappings_count"] == len(FEATURE_TO_CAUSAL)

    def test_test_generation_method_success(self):
        """Test método test_generation con generación exitosa."""
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model"
        )
        
        # Mock generación exitosa
        with patch.object(generator, 'generate') as mock_generate:
            mock_generate.return_value = "Test explanation generated successfully"
            
            result = generator.test_generation()
            
            # Verificar resultado exitoso
            assert result["success"] is True
            assert result["explanation"] == "Test explanation generated successfully"
            assert result["used_llm"] is True
            assert result["explanation_length"] > 0
            assert "elapsed_time_s" in result

    def test_test_generation_method_failure(self):
        """Test método test_generation con fallo en generación."""
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model"
        )
        
        # Mock fallo en generación
        with patch.object(generator, 'generate') as mock_generate:
            mock_generate.side_effect = Exception("Test error")
            
            result = generator.test_generation()
            
            # Verificar manejo de error
            assert result["success"] is False
            assert "error" in result
            assert result["used_llm"] is False
            assert "elapsed_time_s" in result

    def test_format_explanation_ml_terms_filtering(self):
        """Test que _format_explanation filtra términos ML correctamente."""
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model"
        )
        
        # Texto con términos ML
        text_with_ml_terms = """
        Esta predicción se basa en valores SHAP y embeddings del modelo.
        El algoritmo de machine learning identificó features importantes.
        """
        
        result = generator._format_explanation(text_with_ml_terms, 1, 0.85)
        
        # Verificar filtrado
        result_lower = result.lower()
        assert "shap" not in result_lower
        assert "embedding" not in result_lower
        assert "machine learning" not in result_lower
        assert "algoritmo" not in result_lower
        assert "modelo" not in result_lower
        
        # Verificar reemplazo
        assert "factor de análisis" in result_lower

    def test_build_llm_prompt_structure(self):
        """Test que _build_llm_prompt genera estructura correcta."""
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model"
        )
        
        top_factors = [
            ("valor_facturado", 0.15, 150000),
            ("codigo_cups", -0.12, "890201"),
            ("numero_autorizacion", 0.08, "AUT123")
        ]
        
        prompt = generator._build_llm_prompt(1, 0.85, top_factors)
        
        # Verificar elementos del prompt
        assert "GLOSADA" in prompt
        assert "85.0%" in prompt
        assert "valor_facturado" in prompt
        assert "SO0101" in prompt  # Causal mapeado
        assert "auditoría prestacional" in prompt
        assert "Resolución 2284/2023" in prompt
        assert "machine learning" not in prompt.lower()

    def test_causal_mapping_coverage(self):
        """Test que el mapeo de causales cubre features importantes."""
        # Verificar que FEATURE_TO_CAUSAL tiene mappings para features clave
        important_features = [
            "valor_facturado", "codigo_cups", "numero_autorizacion",
            "valor_unitario", "tipo_documento", "diagnostico_principal"
        ]
        
        for feature in important_features:
            assert feature in FEATURE_TO_CAUSAL, f"Feature '{feature}' debe tener mapeo de causal"
            
            causal = FEATURE_TO_CAUSAL[feature]
            assert len(causal) > 10, f"Causal para '{feature}' debe ser descriptivo"
            assert any(code in causal for code in ["SO", "FA", "TA", "DE"]), f"Causal debe incluir código Res. 2284"

    def test_explanation_templates_completeness(self):
        """Test que las plantillas de explicación están completas."""
        # Verificar que hay plantillas para las 3 clases
        for class_id in [0, 1, 2]:
            assert class_id in EXPLANATION_TEMPLATES, f"Falta plantilla para clase {class_id}"
            
            template = EXPLANATION_TEMPLATES[class_id]
            assert "intro" in template, f"Plantilla {class_id} debe tener 'intro'"
            assert "factors" in template, f"Plantilla {class_id} debe tener 'factors'"
            assert "conclusion" in template, f"Plantilla {class_id} debe tener 'conclusion'"
            
            # Verificar que los textos no están vacíos
            for key, text in template.items():
                assert len(text.strip()) > 10, f"Texto '{key}' en plantilla {class_id} muy corto"

    def test_privacy_guard_integration(self):
        """Test integración correcta con PrivacyGuard."""
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model"
        )
        
        # Simular generación exitosa
        with patch('requests.post') as mock_post:
            mock_response = {"response": "Test explanation"}
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status.return_value = None
            
            generator.generate(
                prediction=1,
                confidence=0.8,
                shap_values={"test": 0.1},
                feature_values={"test": "value"}
            )
            
            # Verificar llamadas a PrivacyGuard
            self.mock_privacy_guard.verify_local_only.assert_called()
            self.mock_privacy_guard.log_inference.assert_called_once()
            
            # Verificar parámetros de log_inference
            call_args = self.mock_privacy_guard.log_inference.call_args
            assert call_args[1]["model"] == "test-model"
            assert "input_hash" in call_args[1]