# -*- coding: utf-8 -*-
"""Property tests para NLP Pipeline.

Valida propiedades fundamentales del preprocesamiento de texto:
- Propiedad 1: Idempotencia del preprocesamiento
"""

import pytest
from hypothesis import given, settings, strategies as st, assume
from unittest.mock import Mock, patch
import src.nlp.pipeline as nlp_module

from src.nlp.pipeline import preprocess_text


def _make_mock_nlp(tokens_lemmas: list[str]) -> Mock:
    """Crea un mock de spaCy NLP que retorna los tokens dados."""
    mock_nlp = Mock()
    mock_doc = Mock()
    mock_tokens = []
    for lemma in tokens_lemmas:
        t = Mock()
        t.lemma_ = lemma
        t.is_alpha = True
        t.is_stop = False
        t.text = lemma
        mock_tokens.append(t)
    mock_doc.__iter__ = Mock(return_value=iter(mock_tokens))
    mock_nlp.return_value = mock_doc
    return mock_nlp


def _preprocess_with_mock(text, tokens_lemmas=None):
    """Ejecuta preprocess_text con un mock de spaCy."""
    if tokens_lemmas is None:
        tokens_lemmas = []
    original = nlp_module._nlp_instance
    nlp_module._nlp_instance = None
    try:
        with patch('spacy.load', return_value=_make_mock_nlp(tokens_lemmas)):
            return preprocess_text(text)
    finally:
        nlp_module._nlp_instance = original


class TestNLPPipelineProperties:
    """Property tests para NLP Pipeline usando Hypothesis."""

    @given(st.text())
    @settings(max_examples=30)
    def test_preprocess_text_idempotence_property(self, text):
        """Propiedad 1: Idempotencia del preprocesamiento.

        preprocess_text(preprocess_text(t)) == preprocess_text(t)
        Valida: Requisito 2.3
        """
        # Tokens determinísticos basados en el input
        words = text.lower().split() if text else []
        tokens = [w for w in words if w.isalpha()][:5]

        result1 = _preprocess_with_mock(text, tokens)
        # Segunda aplicación: los tokens del resultado son las mismas palabras
        result2 = _preprocess_with_mock(result1, result1.split() if result1 else [])

        assert result1 == result2, f"Idempotencia fallida: '{text}' -> '{result1}' -> '{result2}'"

    @given(st.one_of(st.none(), st.just(""), st.text().filter(lambda x: not x.strip())))
    @settings(max_examples=30)
    def test_preprocess_text_empty_input_property(self, empty_input):
        """Propiedad: Inputs vacíos siempre producen cadena vacía."""
        result = preprocess_text(empty_input)
        assert result == "", f"Input vacío debe producir '', obtuvo '{result}'"

    @given(st.text(min_size=1).filter(lambda x: x.strip()))
    @settings(max_examples=30)
    def test_preprocess_text_output_is_string_property(self, text):
        """Propiedad: El output siempre es un string."""
        result = _preprocess_with_mock(text, [])
        assert isinstance(result, str)

    @given(st.text().filter(lambda x: any(c.isalpha() for c in x)))
    @settings(max_examples=30)
    def test_preprocess_text_lowercase_property(self, text):
        """Propiedad: El output no contiene letras ASCII en mayúsculas."""
        alpha_chars = [c.lower() for c in text if c.isalpha()]
        tokens = [''.join(alpha_chars[:5])] if alpha_chars else []
        result = _preprocess_with_mock(text, tokens)
        if result:
            # Solo verificar caracteres ASCII que tienen distinción de caso
            assert not any(c.isupper() and c.isascii() for c in result), \
                f"Output contiene mayúsculas ASCII: '{result}'"

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=30)
    def test_preprocess_text_deterministic_property(self, text):
        """Propiedad: El preprocesamiento es determinístico."""
        words = text.lower().split()
        tokens = [w for w in words if w.isalpha()][:3]

        result1 = _preprocess_with_mock(text, tokens)
        result2 = _preprocess_with_mock(text, tokens)
        result3 = _preprocess_with_mock(text, tokens)

        assert result1 == result2 == result3

    @given(st.text().filter(lambda x: not any(c.isalpha() for c in x) and x.strip()))
    @settings(max_examples=30)
    def test_preprocess_text_no_alpha_property(self, non_alpha_text):
        """Propiedad: Texto sin caracteres alfabéticos produce cadena vacía."""
        result = _preprocess_with_mock(non_alpha_text, [])
        assert result == "", f"Texto sin letras debe producir '', obtuvo '{result}'"

    @given(st.text(min_size=1).filter(lambda x: len(x.split()) > 1))
    @settings(max_examples=30)
    def test_preprocess_text_length_property(self, text):
        """Propiedad: El output nunca es más largo que el input."""
        words = text.split()[:2]
        tokens = [w.lower() for w in words if w.isalpha() and len(w) > 2]
        result = _preprocess_with_mock(text, tokens)
        assert len(result) <= len(text)

    @given(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll')),
        min_size=1, max_size=50
    ))
    @settings(max_examples=30)
    def test_preprocess_text_alpha_only_property(self, alpha_text):
        """Propiedad: Texto solo alfabético produce output válido o vacío."""
        assume(len(alpha_text.strip()) > 0)
        words = alpha_text.lower().split()[:3]
        tokens = [w for w in words if len(w) > 1]
        result = _preprocess_with_mock(alpha_text, tokens)
        assert isinstance(result, str)
        if result:
            # Verificar que no hay dígitos ni símbolos de puntuación ASCII
            assert not any(c.isdigit() or (not c.isalpha() and not c.isspace() and c.isascii()) for c in result)
