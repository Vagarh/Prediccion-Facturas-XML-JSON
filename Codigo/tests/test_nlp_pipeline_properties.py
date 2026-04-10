# -*- coding: utf-8 -*-
"""Property tests para NLP Pipeline.

Valida propiedades fundamentales del preprocesamiento de texto:
- Propiedad 1: Idempotencia del preprocesamiento
- Para cualquier texto de entrada, preprocess_text(preprocess_text(t)) == preprocess_text(t)
"""

import pytest
from hypothesis import given, strategies as st, assume
from unittest.mock import Mock, patch

from src.nlp.pipeline import preprocess_text


class TestNLPPipelineProperties:
    """Property tests para NLP Pipeline usando Hypothesis."""

    @given(st.text())
    @patch('src.nlp.pipeline.load_nlp_pipeline')
    def test_preprocess_text_idempotence_property(self, text, mock_load_nlp):
        """
        Propiedad 1: Idempotencia del preprocesamiento.
        
        Para cualquier texto de entrada, aplicar preprocess_text dos veces
        debe producir el mismo resultado que aplicarlo una vez.
        
        Valida: Requisito 2.3
        """
        # Mock del pipeline spaCy para comportamiento determinístico
        mock_nlp = Mock()
        mock_doc = Mock()
        
        # Simular procesamiento determinístico
        # Para simplificar, convertimos a minúsculas y eliminamos espacios extra
        processed_tokens = []
        if text and isinstance(text, str):
            # Simular tokenización básica
            words = text.lower().split()
            for word in words:
                if word.isalpha() and len(word) > 0:
                    mock_token = Mock()
                    mock_token.lemma_ = word
                    mock_token.is_alpha = True
                    mock_token.is_stop = False
                    mock_token.text = word
                    processed_tokens.append(mock_token)
        
        mock_doc.__iter__ = Mock(return_value=iter(processed_tokens))
        mock_nlp.return_value = mock_doc
        mock_load_nlp.return_value = mock_nlp
        
        # Aplicar preprocesamiento una vez
        result1 = preprocess_text(text)
        
        # Aplicar preprocesamiento al resultado anterior
        result2 = preprocess_text(result1)
        
        # Propiedad: f(f(x)) = f(x) (idempotencia)
        assert result1 == result2, f"Idempotencia fallida: '{text}' -> '{result1}' -> '{result2}'"

    @given(st.one_of(st.none(), st.just(""), st.text().filter(lambda x: not x.strip())))
    @patch('src.nlp.pipeline.load_nlp_pipeline')
    def test_preprocess_text_empty_input_property(self, empty_input, mock_load_nlp):
        """
        Propiedad: Inputs vacíos siempre producen cadena vacía.
        
        Para cualquier input que sea None, cadena vacía, o solo whitespace,
        el resultado debe ser cadena vacía.
        """
        # No necesitamos mock del pipeline para inputs vacíos
        result = preprocess_text(empty_input)
        assert result == "", f"Input vacío '{empty_input}' debe producir cadena vacía, obtuvo '{result}'"

    @given(st.text(min_size=1).filter(lambda x: x.strip()))
    @patch('src.nlp.pipeline.load_nlp_pipeline')
    def test_preprocess_text_output_is_string_property(self, text, mock_load_nlp):
        """
        Propiedad: El output siempre es un string.
        
        Para cualquier input válido, preprocess_text debe retornar un string.
        """
        # Mock básico del pipeline
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([]))
        mock_nlp.return_value = mock_doc
        mock_load_nlp.return_value = mock_nlp
        
        result = preprocess_text(text)
        assert isinstance(result, str), f"Output debe ser string, obtuvo {type(result)}"

    @given(st.text().filter(lambda x: any(c.isalpha() for c in x)))
    @patch('src.nlp.pipeline.load_nlp_pipeline')
    def test_preprocess_text_lowercase_property(self, text, mock_load_nlp):
        """
        Propiedad: El output siempre está en minúsculas.
        
        Para cualquier texto que contenga letras, el resultado debe estar
        completamente en minúsculas.
        """
        # Mock que simula conversión a minúsculas
        mock_nlp = Mock()
        mock_doc = Mock()
        
        # Extraer solo caracteres alfabéticos y convertir a minúsculas
        alpha_chars = [c.lower() for c in text if c.isalpha()]
        if alpha_chars:
            # Simular un token en minúsculas
            mock_token = Mock()
            mock_token.lemma_ = ''.join(alpha_chars[:5])  # Limitar longitud
            mock_token.is_alpha = True
            mock_token.is_stop = False
            mock_token.text = mock_token.lemma_
            mock_doc.__iter__ = Mock(return_value=iter([mock_token]))
        else:
            mock_doc.__iter__ = Mock(return_value=iter([]))
        
        mock_nlp.return_value = mock_doc
        mock_load_nlp.return_value = mock_nlp
        
        result = preprocess_text(text)
        
        # Si hay resultado, debe estar en minúsculas
        if result:
            assert result.islower(), f"Output debe estar en minúsculas: '{result}'"

    @given(st.text(min_size=1, max_size=100))
    @patch('src.nlp.pipeline.load_nlp_pipeline')
    def test_preprocess_text_deterministic_property(self, text, mock_load_nlp):
        """
        Propiedad: El preprocesamiento es determinístico.
        
        Para el mismo input, múltiples llamadas deben producir el mismo output.
        """
        # Mock determinístico
        mock_nlp = Mock()
        mock_doc = Mock()
        
        # Crear tokens determinísticos basados en el input
        tokens = []
        words = text.lower().split()[:3]  # Limitar a 3 palabras para eficiencia
        for word in words:
            if word.isalpha():
                mock_token = Mock()
                mock_token.lemma_ = word
                mock_token.is_alpha = True
                mock_token.is_stop = False
                mock_token.text = word
                tokens.append(mock_token)
        
        mock_doc.__iter__ = Mock(return_value=iter(tokens))
        mock_nlp.return_value = mock_doc
        mock_load_nlp.return_value = mock_nlp
        
        # Múltiples llamadas con el mismo input
        result1 = preprocess_text(text)
        result2 = preprocess_text(text)
        result3 = preprocess_text(text)
        
        assert result1 == result2 == result3, f"Preprocesamiento no determinístico para '{text}'"

    @given(st.text().filter(lambda x: not any(c.isalpha() for c in x) and x.strip()))
    @patch('src.nlp.pipeline.load_nlp_pipeline')
    def test_preprocess_text_no_alpha_property(self, non_alpha_text, mock_load_nlp):
        """
        Propiedad: Texto sin caracteres alfabéticos produce cadena vacía.
        
        Para texto que solo contiene números, símbolos, o puntuación,
        el resultado debe ser cadena vacía.
        """
        # Mock que filtra tokens no alfabéticos
        mock_nlp = Mock()
        mock_doc = Mock()
        
        # Simular tokens no alfabéticos que se filtran
        mock_token = Mock()
        mock_token.lemma_ = "123"
        mock_token.is_alpha = False  # No alfabético
        mock_token.is_stop = False
        mock_token.text = "123"
        
        mock_doc.__iter__ = Mock(return_value=iter([mock_token]))
        mock_nlp.return_value = mock_doc
        mock_load_nlp.return_value = mock_nlp
        
        result = preprocess_text(non_alpha_text)
        
        # Sin caracteres alfabéticos válidos, debe ser cadena vacía
        assert result == "", f"Texto sin letras '{non_alpha_text}' debe producir cadena vacía, obtuvo '{result}'"

    @given(st.text(min_size=1).filter(lambda x: len(x.split()) > 1))
    @patch('src.nlp.pipeline.load_nlp_pipeline')
    def test_preprocess_text_length_property(self, text, mock_load_nlp):
        """
        Propiedad: El output nunca es más largo que el input procesado.
        
        Después del preprocesamiento (eliminación de stopwords, puntuación, etc.),
        el resultado no debe ser más largo que el input original.
        """
        # Mock que simula reducción de longitud
        mock_nlp = Mock()
        mock_doc = Mock()
        
        # Simular procesamiento que reduce el texto
        words = text.split()[:2]  # Tomar máximo 2 palabras
        tokens = []
        for word in words:
            if word.isalpha() and len(word) > 2:  # Filtrar palabras cortas
                mock_token = Mock()
                mock_token.lemma_ = word.lower()
                mock_token.is_alpha = True
                mock_token.is_stop = False
                mock_token.text = word.lower()
                tokens.append(mock_token)
        
        mock_doc.__iter__ = Mock(return_value=iter(tokens))
        mock_nlp.return_value = mock_doc
        mock_load_nlp.return_value = mock_nlp
        
        result = preprocess_text(text)
        
        # El resultado no debe ser más largo que el input original
        assert len(result) <= len(text), f"Output más largo que input: '{text}' -> '{result}'"

    @given(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')), min_size=1, max_size=50))
    @patch('src.nlp.pipeline.load_nlp_pipeline')
    def test_preprocess_text_alpha_only_property(self, alpha_text, mock_load_nlp):
        """
        Propiedad: Texto solo alfabético produce output válido o vacío.
        
        Para texto que solo contiene letras, el resultado debe ser
        una cadena válida (posiblemente vacía si todas son stopwords).
        """
        assume(len(alpha_text.strip()) > 0)  # Asegurar que no está vacío
        
        # Mock que procesa texto alfabético
        mock_nlp = Mock()
        mock_doc = Mock()
        
        # Simular procesamiento de texto alfabético
        words = alpha_text.lower().split()[:3]  # Limitar para eficiencia
        tokens = []
        for word in words:
            if len(word) > 1:  # Evitar palabras de una letra
                mock_token = Mock()
                mock_token.lemma_ = word
                mock_token.is_alpha = True
                mock_token.is_stop = False  # Asumir que no es stopword
                mock_token.text = word
                tokens.append(mock_token)
        
        mock_doc.__iter__ = Mock(return_value=iter(tokens))
        mock_nlp.return_value = mock_doc
        mock_load_nlp.return_value = mock_nlp
        
        result = preprocess_text(alpha_text)
        
        # El resultado debe ser string válido
        assert isinstance(result, str), f"Output debe ser string para input alfabético '{alpha_text}'"
        
        # Si no está vacío, debe contener solo caracteres válidos
        if result:
            assert all(c.isalpha() or c.isspace() for c in result), f"Output contiene caracteres inválidos: '{result}'"