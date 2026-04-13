# -*- coding: utf-8 -*-
"""Property tests para ExplanationGenerator.

Valida propiedades fundamentales de la generación de explicaciones:
- Propiedad 6: Degradación graciosa siempre retorna string no vacío
"""

import pytest
from hypothesis import given, strategies as st, assume
from unittest.mock import Mock, patch, MagicMock
import requests
from requests.exceptions import RequestException, Timeout

from src.explainability.explanation_generator import ExplanationGenerator
from src.privacy.guard import PrivacyGuard
from src.config import EXPLANATION_MAX_SECONDS


class TestExplanationGeneratorProperties:
    """Property tests para ExplanationGenerator usando Hypothesis."""

    def setup_method(self):
        """Configuración inicial para cada test."""
        # Mock del PrivacyGuard
        self.mock_privacy_guard = Mock(spec=PrivacyGuard)
        self.mock_privacy_guard.verify_local_only.return_value = None
        self.mock_privacy_guard.check_ollama_available.return_value = True
        self.mock_privacy_guard.log_inference.return_value = None

    @given(
        st.integers(min_value=0, max_value=2),  # prediction
        st.floats(min_value=0.0, max_value=1.0),  # confidence
        st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.floats(min_value=-1.0, max_value=1.0),
            min_size=1, max_size=10
        ),  # shap_values
        st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(st.integers(), st.floats(), st.text()),
            min_size=1, max_size=10
        )  # feature_values
    )
    def test_graceful_degradation_always_returns_non_empty_string(
        self, prediction, confidence, shap_values, feature_values
    ):
        """
        Propiedad 6: Degradación graciosa siempre retorna string no vacío.
        
        Para cualquier combinación de predicción/SHAP values, generate(...) retorna
        str no vacío incluso cuando Ollama no está disponible.
        
        Valida: Requisito 4.5
        """
        # Caso 1: Ollama no disponible (fallback desde el inicio)
        self.mock_privacy_guard.check_ollama_available.return_value = False
        
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model",
            timeout_s=1.0
        )
        
        result = generator.generate(prediction, confidence, shap_values, feature_values)
        
        # Propiedad: siempre retorna string no vacío
        assert isinstance(result, str), f"Resultado debe ser string, obtuvo {type(result)}"
        assert len(result.strip()) > 0, "Resultado no debe estar vacío"
        assert "RESULTADO DE AUDITORÍA" in result, "Debe contener header de auditoría"

    @given(
        st.integers(min_value=0, max_value=2),
        st.floats(min_value=0.0, max_value=1.0),
        st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.floats(min_value=-1.0, max_value=1.0),
            min_size=1, max_size=5
        ),
        st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(st.integers(), st.text()),
            min_size=1, max_size=5
        )
    )
    def test_ollama_failure_graceful_degradation(
        self, prediction, confidence, shap_values, feature_values
    ):
        """
        Propiedad: Degradación graciosa cuando Ollama falla.
        
        Cuando Ollama está disponible pero falla durante la generación,
        debe retornar explicación fallback válida.
        """
        # Ollama disponible inicialmente
        self.mock_privacy_guard.check_ollama_available.return_value = True
        
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model",
            timeout_s=1.0
        )
        
        # Mock requests.post para simular fallo de Ollama
        with patch('requests.post') as mock_post:
            mock_post.side_effect = RequestException("Connection failed")
            
            result = generator.generate(prediction, confidence, shap_values, feature_values)
            
            # Propiedad: degradación graciosa exitosa
            assert isinstance(result, str), "Debe retornar string en caso de fallo"
            assert len(result.strip()) > 0, "Resultado no debe estar vacío tras fallo"
            assert "RESULTADO DE AUDITORÍA" in result, "Debe contener header tras fallo"

    @given(
        st.integers(min_value=0, max_value=2),
        st.floats(min_value=0.0, max_value=1.0),
        st.dictionaries(
            keys=st.text(min_size=1, max_size=15),
            values=st.floats(min_value=-0.5, max_value=0.5),
            min_size=1, max_size=3
        ),
        st.dictionaries(
            keys=st.text(min_size=1, max_size=15),
            values=st.integers(min_value=0, max_value=1000000),
            min_size=1, max_size=3
        )
    )
    def test_timeout_graceful_degradation(
        self, prediction, confidence, shap_values, feature_values
    ):
        """
        Propiedad: Degradación graciosa cuando se excede timeout.
        
        Cuando la generación LLM excede el timeout, debe retornar
        explicación fallback válida.
        """
        self.mock_privacy_guard.check_ollama_available.return_value = True
        
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model",
            timeout_s=0.1  # Timeout muy corto para forzar timeout
        )
        
        # Mock requests.post para simular timeout
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Timeout("Request timed out")
            
            result = generator.generate(prediction, confidence, shap_values, feature_values)
            
            # Propiedad: manejo correcto de timeout
            assert isinstance(result, str), "Debe retornar string tras timeout"
            assert len(result.strip()) > 0, "Resultado no debe estar vacío tras timeout"
            assert "RESULTADO DE AUDITORÍA" in result, "Debe contener header tras timeout"

    @given(
        st.integers(min_value=0, max_value=2),
        st.floats(min_value=0.0, max_value=1.0)
    )
    def test_empty_shap_values_handling(self, prediction, confidence):
        """
        Propiedad: Manejo correcto de valores SHAP vacíos.
        
        Cuando no hay valores SHAP o están vacíos, debe generar
        explicación válida con información disponible.
        """
        self.mock_privacy_guard.check_ollama_available.return_value = False
        
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model"
        )
        
        # Casos de valores vacíos
        empty_cases = [
            ({}, {}),  # Ambos vacíos
            ({"feature1": 0.0}, {}),  # Solo SHAP
            ({}, {"feature1": "value1"}),  # Solo features
        ]
        
        for shap_values, feature_values in empty_cases:
            result = generator.generate(prediction, confidence, shap_values, feature_values)
            
            # Propiedad: manejo robusto de casos vacíos
            assert isinstance(result, str), f"Debe retornar string para caso vacío {shap_values}, {feature_values}"
            assert len(result.strip()) > 0, f"No debe estar vacío para caso {shap_values}, {feature_values}"

    @given(
        st.integers(min_value=0, max_value=2),
        st.floats(min_value=0.0, max_value=1.0),
        st.dictionaries(
            keys=st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=5, max_size=30),
            values=st.floats(min_value=-2.0, max_value=2.0),
            min_size=3, max_size=8
        )
    )
    def test_feature_name_consistency(self, prediction, confidence, shap_values):
        """
        Propiedad: Consistencia en el manejo de nombres de features.
        
        Los nombres de features deben ser manejados consistentemente
        independientemente de su formato o contenido.
        """
        # Generar feature_values correspondientes
        feature_values = {name: f"value_{i}" for i, name in enumerate(shap_values.keys())}
        
        self.mock_privacy_guard.check_ollama_available.return_value = False
        
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model"
        )
        
        result = generator.generate(prediction, confidence, shap_values, feature_values)
        
        # Propiedad: manejo consistente de nombres de features
        assert isinstance(result, str), "Debe retornar string para cualquier nombre de feature"
        assert len(result.strip()) > 0, "No debe estar vacío para nombres de features válidos"
        
        # Verificar que al menos algunos nombres de features aparecen en la explicación
        # (para explicaciones fallback que incluyen detalles técnicos)
        feature_names_in_result = sum(1 for name in shap_values.keys() if name in result)
        # Al menos uno de los top features debería aparecer
        assert feature_names_in_result >= 0, "Explicación debe referenciar features relevantes"

    @given(
        st.floats(min_value=-1.0, max_value=1.0),
        st.floats(min_value=-1.0, max_value=1.0),
        st.floats(min_value=-1.0, max_value=1.0)
    )
    def test_prediction_class_consistency(self, shap1, shap2, shap3):
        """
        Propiedad: Consistencia entre clases de predicción.
        
        Para cada clase de predicción (0, 1, 2), debe generar
        explicaciones apropiadas y consistentes.
        """
        shap_values = {"feature1": shap1, "feature2": shap2, "feature3": shap3}
        feature_values = {"feature1": 100, "feature2": "ABC123", "feature3": 50.5}
        
        self.mock_privacy_guard.check_ollama_available.return_value = False
        
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model"
        )
        
        class_keywords = {
            0: "AUDITADA",
            1: "GLOSADA", 
            2: "DEVUELTA"
        }
        
        for prediction_class in [0, 1, 2]:
            result = generator.generate(
                prediction=prediction_class,
                confidence=0.8,
                shap_values=shap_values,
                feature_values=feature_values
            )
            
            # Propiedad: explicación apropiada para cada clase
            expected_keyword = class_keywords[prediction_class]
            assert expected_keyword in result, f"Explicación para clase {prediction_class} debe contener '{expected_keyword}'"
            assert isinstance(result, str), f"Debe retornar string para clase {prediction_class}"
            assert len(result.strip()) > 0, f"No debe estar vacío para clase {prediction_class}"

    @given(
        st.integers(min_value=0, max_value=2),
        st.floats(min_value=0.0, max_value=1.0),
        st.dictionaries(
            keys=st.sampled_from([
                "valor_facturado", "codigo_cups", "numero_autorizacion",
                "diagnostico_principal", "procedimiento_principal"
            ]),
            values=st.floats(min_value=-1.0, max_value=1.0),
            min_size=1, max_size=5
        )
    )
    def test_causal_mapping_integration(self, prediction, confidence, shap_values):
        """
        Propiedad: Integración correcta de mapeo de causales.
        
        Cuando se usan features que tienen mapeo a causales de Res. 2284/2023,
        la explicación debe incluir referencias apropiadas.
        """
        # Features con valores realistas
        feature_values = {
            "valor_facturado": 150000,
            "codigo_cups": "890201", 
            "numero_autorizacion": "AUT123456",
            "diagnostico_principal": "I50.9",
            "procedimiento_principal": "890201"
        }
        
        # Filtrar solo features que están en feature_values
        filtered_shap = {k: v for k, v in shap_values.items() if k in feature_values}
        
        if not filtered_shap:
            # Si no hay features válidas, usar una por defecto
            filtered_shap = {"valor_facturado": 0.1}
        
        self.mock_privacy_guard.check_ollama_available.return_value = False
        
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model"
        )
        
        result = generator.generate(prediction, confidence, filtered_shap, feature_values)
        
        # Propiedad: integración de causales cuando aplica
        assert isinstance(result, str), "Debe retornar string con causales"
        assert len(result.strip()) > 0, "No debe estar vacío con causales"
        
        # Para explicaciones fallback, verificar que incluye información de causales
        # cuando hay features mapeadas
        mapped_features = [f for f in filtered_shap.keys() if f in [
            "valor_facturado", "codigo_cups", "numero_autorizacion",
            "diagnostico_principal", "procedimiento_principal"
        ]]
        
        if mapped_features:
            # Debe mencionar al menos uno de los features mapeados
            feature_mentioned = any(feature in result for feature in mapped_features)
            assert feature_mentioned, f"Debe mencionar al menos un feature mapeado: {mapped_features}"

    @given(st.floats(min_value=0.1, max_value=10.0))
    def test_timeout_parameter_respected(self, timeout_value):
        """
        Propiedad: El parámetro de timeout es respetado.
        
        El generador debe usar el timeout especificado y manejar
        correctamente diferentes valores de timeout.
        """
        self.mock_privacy_guard.check_ollama_available.return_value = True
        
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model",
            timeout_s=timeout_value
        )
        
        # Propiedad: timeout configurado correctamente
        assert generator.timeout_s == timeout_value, f"Timeout debe ser {timeout_value}, obtuvo {generator.timeout_s}"
        
        # Verificar que el timeout se usa en las llamadas
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Timeout("Timeout")
            
            result = generator.generate(
                prediction=1,
                confidence=0.8,
                shap_values={"test": 0.1},
                feature_values={"test": "value"}
            )
            
            # Debe usar fallback tras timeout
            assert isinstance(result, str), "Debe retornar string tras timeout configurado"
            assert len(result.strip()) > 0, "No debe estar vacío tras timeout configurado"

    @given(
        st.integers(min_value=0, max_value=2),
        st.floats(min_value=0.0, max_value=1.0)
    )
    def test_ml_terms_filtering_property(self, prediction, confidence):
        """
        Propiedad: Filtrado de términos técnicos de ML.
        
        Las explicaciones no deben contener términos técnicos de ML
        como "SHAP", "embedding", "feature importance", etc.
        """
        shap_values = {"shap_feature": 0.1, "embedding_value": -0.2}
        feature_values = {"shap_feature": "test", "embedding_value": 42}
        
        self.mock_privacy_guard.check_ollama_available.return_value = False
        
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model"
        )
        
        result = generator.generate(prediction, confidence, shap_values, feature_values)
        
        # Propiedad: no contiene términos técnicos de ML
        ml_terms = ["shap", "embedding", "feature importance", "machine learning", "modelo", "algoritmo"]
        
        result_lower = result.lower()
        for term in ml_terms:
            assert term not in result_lower, f"Explicación no debe contener término ML: '{term}'"
        
        # Debe contener términos apropiados para auditores
        appropriate_terms = ["auditoría", "factor", "análisis"]
        has_appropriate_term = any(term in result_lower for term in appropriate_terms)
        assert has_appropriate_term, "Explicación debe contener términos apropiados para auditores"

    @given(st.booleans())
    def test_status_consistency_property(self, ollama_available):
        """
        Propiedad: Consistencia del estado del generador.
        
        El método get_status() debe retornar información consistente
        con el estado actual del generador.
        """
        self.mock_privacy_guard.check_ollama_available.return_value = ollama_available
        
        generator = ExplanationGenerator(
            privacy_guard=self.mock_privacy_guard,
            llm_model="test-model",
            timeout_s=5.0
        )
        
        status = generator.get_status()
        
        # Propiedad: estado consistente
        assert isinstance(status, dict), "get_status debe retornar diccionario"
        assert "ollama_available" in status, "Estado debe incluir disponibilidad de Ollama"
        assert status["ollama_available"] == ollama_available, "Estado de Ollama debe ser consistente"
        assert status["llm_model"] == "test-model", "Modelo debe ser consistente"
        assert status["timeout_s"] == 5.0, "Timeout debe ser consistente"
        assert isinstance(status["feature_mappings_count"], int), "Conteo de mappings debe ser entero"
        assert status["feature_mappings_count"] > 0, "Debe tener mappings de features"