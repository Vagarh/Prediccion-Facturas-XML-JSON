# -*- coding: utf-8 -*-
"""Tests unitarios para NLP Pipeline.

Valida el preprocesamiento de texto clínico en español:
- preprocess_text(None) retorna ""
- stopwords médicos son eliminados
- función es idempotente (aplicar dos veces = mismo resultado)
- load_nlp_pipeline() carga modelo con caché singleton
"""

import pytest
from unittest.mock import Mock, patch
import spacy

from src.nlp.pipeline import (
    load_nlp_pipeline,
    preprocess_text,
    MEDICAL_STOPWORDS
)


class TestNLPPipeline:
    """Suite de tests para NLP Pipeline."""

    def test_preprocess_text_with_none_returns_empty_string(self):
        """Test que preprocess_text(None) retorna cadena vacía."""
        result = preprocess_text(None)
        assert result == ""

    def test_preprocess_text_with_empty_string_returns_empty_string(self):
        """Test que preprocess_text("") retorna cadena vacía."""
        result = preprocess_text("")
        assert result == ""

    def test_preprocess_text_with_whitespace_returns_empty_string(self):
        """Test que preprocess_text con solo espacios retorna cadena vacía."""
        whitespace_inputs = ["   ", "\t", "\n", "  \t\n  "]
        
        for input_text in whitespace_inputs:
            result = preprocess_text(input_text)
            assert result == ""

    @patch('src.nlp.pipeline.load_nlp_pipeline')
    def test_preprocess_text_removes_medical_stopwords(self, mock_load_nlp):
        """Test que stopwords médicos son eliminados."""
        # Mock del pipeline spaCy
        mock_nlp = Mock()
        mock_doc = Mock()
        
        # Simular tokens con stopwords médicos y palabras normales
        mock_token1 = Mock()
        mock_token1.lemma_ = "paciente"
        mock_token1.is_stop = False
        mock_token1.is_alpha = True
        mock_token1.text = "paciente"
        
        mock_token2 = Mock()
        mock_token2.lemma_ = "hipertensión"
        mock_token2.is_stop = False
        mock_token2.is_alpha = True
        mock_token2.text = "hipertensión"
        
        mock_token3 = Mock()
        mock_token3.lemma_ = "presenta"
        mock_token3.is_stop = False
        mock_token3.is_alpha = True
        mock_token3.text = "presenta"
        
        mock_doc.__iter__ = Mock(return_value=iter([mock_token1, mock_token2, mock_token3]))
        mock_nlp.return_value = mock_doc
        mock_load_nlp.return_value = mock_nlp
        
        # Verificar que "paciente" y "presenta" están en MEDICAL_STOPWORDS
        assert "paciente" in MEDICAL_STOPWORDS
        assert "presenta" in MEDICAL_STOPWORDS
        
        result = preprocess_text("El paciente presenta hipertensión")
        
        # Solo debe quedar "hipertensión" (sin stopwords médicos)
        assert "paciente" not in result
        assert "presenta" not in result
        assert "hipertensión" in result

    @patch('src.nlp.pipeline.load_nlp_pipeline')
    def test_preprocess_text_converts_to_lowercase(self, mock_load_nlp):
        """Test que el texto se convierte a minúsculas."""
        mock_nlp = Mock()
        mock_doc = Mock()
        
        mock_token = Mock()
        mock_token.lemma_ = "diabetes"
        mock_token.is_stop = False
        mock_token.is_alpha = True
        mock_token.text = "DIABETES"
        
        mock_doc.__iter__ = Mock(return_value=iter([mock_token]))
        mock_nlp.return_value = mock_doc
        mock_load_nlp.return_value = mock_nlp
        
        result = preprocess_text("DIABETES MELLITUS")
        
        # Debe estar en minúsculas
        assert result.islower()
        assert "diabetes" in result

    @patch('src.nlp.pipeline.load_nlp_pipeline')
    def test_preprocess_text_removes_non_alpha_tokens(self, mock_load_nlp):
        """Test que tokens no alfabéticos son eliminados."""
        mock_nlp = Mock()
        mock_doc = Mock()
        
        # Token alfabético válido
        mock_token1 = Mock()
        mock_token1.lemma_ = "diabetes"
        mock_token1.is_stop = False
        mock_token1.is_alpha = True
        mock_token1.text = "diabetes"
        
        # Token no alfabético (número)
        mock_token2 = Mock()
        mock_token2.lemma_ = "123"
        mock_token2.is_stop = False
        mock_token2.is_alpha = False
        mock_token2.text = "123"
        
        # Token no alfabético (puntuación)
        mock_token3 = Mock()
        mock_token3.lemma_ = "."
        mock_token3.is_stop = False
        mock_token3.is_alpha = False
        mock_token3.text = "."
        
        mock_doc.__iter__ = Mock(return_value=iter([mock_token1, mock_token2, mock_token3]))
        mock_nlp.return_value = mock_doc
        mock_load_nlp.return_value = mock_nlp
        
        result = preprocess_text("diabetes 123 .")
        
        # Solo debe quedar el token alfabético
        assert "diabetes" in result
        assert "123" not in result
        assert "." not in result

    @patch('src.nlp.pipeline.load_nlp_pipeline')
    def test_preprocess_text_is_idempotent(self, mock_load_nlp):
        """Test que la función es idempotente (aplicar dos veces = mismo resultado)."""
        mock_nlp = Mock()
        mock_doc = Mock()
        
        mock_token = Mock()
        mock_token.lemma_ = "diabetes"
        mock_token.is_stop = False
        mock_token.is_alpha = True
        mock_token.text = "diabetes"
        
        mock_doc.__iter__ = Mock(return_value=iter([mock_token]))
        mock_nlp.return_value = mock_doc
        mock_load_nlp.return_value = mock_nlp
        
        input_text = "Diabetes mellitus tipo 2"
        
        # Aplicar preprocesamiento una vez
        result1 = preprocess_text(input_text)
        
        # Aplicar preprocesamiento al resultado anterior
        result2 = preprocess_text(result1)
        
        # Deben ser idénticos
        assert result1 == result2

    @patch('spacy.load')
    def test_load_nlp_pipeline_loads_spanish_model(self, mock_spacy_load):
        """Test que load_nlp_pipeline() carga el modelo en español."""
        mock_nlp = Mock()
        mock_spacy_load.return_value = mock_nlp
        
        result = load_nlp_pipeline()
        
        # Verificar que se cargó el modelo correcto
        mock_spacy_load.assert_called_once_with("es_core_news_md")
        assert result == mock_nlp

    @patch('spacy.load')
    def test_load_nlp_pipeline_singleton_cache(self, mock_spacy_load):
        """Test que load_nlp_pipeline() usa caché singleton."""
        mock_nlp = Mock()
        mock_spacy_load.return_value = mock_nlp
        
        # Limpiar caché si existe
        if hasattr(load_nlp_pipeline, '_cached_nlp'):
            delattr(load_nlp_pipeline, '_cached_nlp')
        
        # Primera llamada
        result1 = load_nlp_pipeline()
        
        # Segunda llamada
        result2 = load_nlp_pipeline()
        
        # spacy.load debe haberse llamado solo una vez
        mock_spacy_load.assert_called_once()
        
        # Ambos resultados deben ser el mismo objeto
        assert result1 is result2

    @patch('spacy.load')
    def test_load_nlp_pipeline_handles_model_not_found(self, mock_spacy_load):
        """Test que load_nlp_pipeline() maneja error cuando modelo no está instalado."""
        # Simular error de modelo no encontrado
        mock_spacy_load.side_effect = OSError("Can't find model 'es_core_news_md'")
        
        with pytest.raises(OSError) as exc_info:
            load_nlp_pipeline()
        
        assert "es_core_news_md" in str(exc_info.value)

    def test_medical_stopwords_contains_expected_terms(self):
        """Test que MEDICAL_STOPWORDS contiene términos médicos esperados."""
        expected_stopwords = [
            "paciente", "presenta", "refiere", "diagnostico", "tratamiento",
            "medico", "clinico", "hospital", "consulta", "revision"
        ]
        
        for stopword in expected_stopwords:
            assert stopword in MEDICAL_STOPWORDS, f"'{stopword}' debe estar en MEDICAL_STOPWORDS"

    def test_medical_stopwords_is_list_of_strings(self):
        """Test que MEDICAL_STOPWORDS es una lista de strings."""
        assert isinstance(MEDICAL_STOPWORDS, list)
        assert len(MEDICAL_STOPWORDS) > 0
        
        for stopword in MEDICAL_STOPWORDS:
            assert isinstance(stopword, str)
            assert len(stopword) > 0

    @patch('src.nlp.pipeline.load_nlp_pipeline')
    def test_preprocess_text_with_real_medical_text(self, mock_load_nlp):
        """Test de integración con texto médico realista."""
        mock_nlp = Mock()
        mock_doc = Mock()
        
        # Simular procesamiento de texto médico real
        tokens = [
            ("hipertensión", True, False),  # palabra médica válida
            ("arterial", True, False),      # palabra médica válida
            ("esencial", True, False),      # palabra médica válida
        ]
        
        mock_tokens = []
        for lemma, is_alpha, is_stop in tokens:
            mock_token = Mock()
            mock_token.lemma_ = lemma
            mock_token.is_alpha = is_alpha
            mock_token.is_stop = is_stop
            mock_token.text = lemma
            mock_tokens.append(mock_token)
        
        mock_doc.__iter__ = Mock(return_value=iter(mock_tokens))
        mock_nlp.return_value = mock_doc
        mock_load_nlp.return_value = mock_nlp
        
        result = preprocess_text("El paciente presenta hipertensión arterial esencial")
        
        # Debe contener términos médicos pero no stopwords
        assert "hipertensión" in result
        assert "arterial" in result
        assert "esencial" in result
        assert "paciente" not in result  # stopword médico
        assert "presenta" not in result  # stopword médico

    @patch('src.nlp.pipeline.load_nlp_pipeline')
    def test_preprocess_text_handles_special_characters(self, mock_load_nlp):
        """Test que preprocess_text maneja caracteres especiales correctamente."""
        mock_nlp = Mock()
        mock_doc = Mock()
        
        # Solo token alfabético válido
        mock_token = Mock()
        mock_token.lemma_ = "diabetes"
        mock_token.is_alpha = True
        mock_token.is_stop = False
        mock_token.text = "diabetes"
        
        mock_doc.__iter__ = Mock(return_value=iter([mock_token]))
        mock_nlp.return_value = mock_doc
        mock_load_nlp.return_value = mock_nlp
        
        # Texto con caracteres especiales
        result = preprocess_text("Diabetes (tipo-2) - 50% casos")
        
        # Solo debe quedar la palabra válida
        assert "diabetes" in result
        assert "(" not in result
        assert ")" not in result
        assert "-" not in result
        assert "%" not in result