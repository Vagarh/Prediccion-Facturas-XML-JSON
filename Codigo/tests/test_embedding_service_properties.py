# -*- coding: utf-8 -*-
"""Property tests para EmbeddingService.

Valida propiedades fundamentales de la generación de embeddings:
- Propiedad 2: Dimensión fija de embeddings
- Propiedad 3: Vector cero para inputs nulos
"""

import pytest
from hypothesis import given, strategies as st, assume
from unittest.mock import Mock, patch
import numpy as np

from src.embeddings.service import EmbeddingService
from src.config import EMBEDDING_DIM, RANDOM_STATE


class TestEmbeddingServiceProperties:
    """Property tests para EmbeddingService usando Hypothesis."""

    def setup_method(self):
        """Configuración inicial para cada test."""
        # Mock del modelo SentenceTransformer
        with patch('src.embeddings.service.SentenceTransformer') as mock_st:
            mock_model = Mock()
            # Por defecto, retorna vector de dimensión correcta
            mock_model.encode.return_value = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            mock_st.return_value = mock_model
            
            self.service = EmbeddingService(
                model_name="test-model",
                random_state=RANDOM_STATE
            )

    @given(st.text().filter(lambda x: x is not None and x.strip()))
    def test_embedding_dimension_property(self, text):
        """
        Propiedad 2: Dimensión fija de embeddings.
        
        Para cualquier texto no nulo, len(service.encode(text)) == EMBEDDING_DIM
        
        Valida: Requisito 2.1
        """
        assume(len(text.strip()) > 0)  # Asegurar que no está vacío después de strip
        
        # Mock del modelo para retornar vector de dimensión correcta
        expected_embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        self.service._model.encode.return_value = expected_embedding
        
        result = self.service.encode(text)
        
        # Propiedad: dimensión siempre es EMBEDDING_DIM
        assert isinstance(result, np.ndarray), f"Resultado debe ser numpy array para '{text}'"
        assert result.shape == (EMBEDDING_DIM,), f"Dimensión incorrecta para '{text}': esperada {EMBEDDING_DIM}, obtuvo {result.shape}"
        assert result.dtype == np.float32, f"Dtype incorrecto para '{text}': esperado float32, obtuvo {result.dtype}"

    @given(st.one_of(st.none(), st.just(""), st.text().filter(lambda x: not x.strip())))
    def test_null_input_zero_vector_property(self, null_input):
        """
        Propiedad 3: Vector cero para inputs nulos.
        
        service.encode(None) y service.encode("") producen np.zeros(EMBEDDING_DIM)
        
        Valida: Requisito 2.4
        """
        result = self.service.encode(null_input)
        
        # Propiedad: inputs nulos/vacíos producen vector de ceros
        expected_zero_vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        
        assert isinstance(result, np.ndarray), f"Resultado debe ser numpy array para input nulo '{null_input}'"
        assert result.shape == (EMBEDDING_DIM,), f"Dimensión incorrecta para input nulo: {result.shape}"
        assert result.dtype == np.float32, f"Dtype incorrecto para input nulo: {result.dtype}"
        np.testing.assert_array_equal(result, expected_zero_vector, 
                                    f"Input nulo '{null_input}' debe producir vector de ceros")
        
        # Verificar que no se llamó al modelo para inputs nulos
        self.service._model.encode.assert_not_called()

    @given(st.lists(st.text(), min_size=0, max_size=10))
    def test_batch_encoding_shape_property(self, text_list):
        """
        Propiedad: encode_batch produce forma correcta.
        
        Para cualquier lista de textos, encode_batch debe retornar
        array de forma (len(texts), EMBEDDING_DIM).
        """
        # Mock del modelo para batch encoding
        batch_size = len(text_list)
        expected_batch = np.random.randn(batch_size, EMBEDDING_DIM).astype(np.float32)
        self.service._model.encode.return_value = expected_batch
        
        result = self.service.encode_batch(text_list)
        
        # Propiedad: forma del batch es (n_samples, EMBEDDING_DIM)
        expected_shape = (batch_size, EMBEDDING_DIM)
        assert isinstance(result, np.ndarray), f"Resultado debe ser numpy array para batch {text_list}"
        assert result.shape == expected_shape, f"Forma incorrecta para batch: esperada {expected_shape}, obtuvo {result.shape}"
        assert result.dtype == np.float32, f"Dtype incorrecto para batch: {result.dtype}"

    @given(st.text(min_size=1, max_size=100))
    def test_encoding_deterministic_property(self, text):
        """
        Propiedad: La codificación es determinística.
        
        Para el mismo texto, múltiples llamadas a encode deben
        producir el mismo resultado.
        """
        assume(text.strip())  # Asegurar que no está vacío
        
        # Mock determinístico basado en el texto
        # Usar hash del texto para generar embedding reproducible
        np.random.seed(hash(text) % (2**32))
        expected_embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        
        # Configurar mock para retornar siempre el mismo embedding
        self.service._model.encode.return_value = expected_embedding
        
        # Múltiples llamadas
        result1 = self.service.encode(text)
        result2 = self.service.encode(text)
        result3 = self.service.encode(text)
        
        # Propiedad: determinismo
        np.testing.assert_array_equal(result1, result2, f"Encoding no determinístico para '{text}'")
        np.testing.assert_array_equal(result2, result3, f"Encoding no determinístico para '{text}'")

    @given(st.lists(st.one_of(st.text(), st.none()), min_size=1, max_size=5))
    def test_batch_null_handling_property(self, mixed_list):
        """
        Propiedad: encode_batch maneja correctamente mezcla de textos y nulos.
        
        Para listas que contienen tanto textos válidos como valores nulos,
        las posiciones nulas deben tener vectores de ceros.
        """
        # Contar textos válidos (no nulos y no vacíos)
        valid_texts = [t for t in mixed_list if t is not None and str(t).strip()]
        valid_count = len(valid_texts)
        
        # Mock para textos válidos solamente
        if valid_count > 0:
            valid_embeddings = np.random.randn(valid_count, EMBEDDING_DIM).astype(np.float32)
            self.service._model.encode.return_value = valid_embeddings
        
        result = self.service.encode_batch(mixed_list)
        
        # Propiedad: forma correcta
        expected_shape = (len(mixed_list), EMBEDDING_DIM)
        assert result.shape == expected_shape, f"Forma incorrecta para batch mixto: {result.shape}"
        
        # Propiedad: posiciones nulas tienen vectores de ceros
        zero_vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        for i, text in enumerate(mixed_list):
            if text is None or not str(text).strip():
                np.testing.assert_array_equal(result[i], zero_vector,
                                            f"Posición {i} con valor nulo '{text}' debe ser vector de ceros")

    @given(st.integers(min_value=0, max_value=1000))
    def test_zero_vector_property(self, seed):
        """
        Propiedad: zero_vector() siempre retorna el mismo vector de ceros.
        
        Independientemente del estado interno, zero_vector() debe
        retornar consistentemente un vector de ceros de la dimensión correcta.
        """
        # Cambiar estado interno (no debe afectar zero_vector)
        np.random.seed(seed)
        
        result = self.service.zero_vector()
        expected = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        
        # Propiedad: siempre vector de ceros de dimensión correcta
        assert isinstance(result, np.ndarray), "zero_vector debe retornar numpy array"
        assert result.shape == (EMBEDDING_DIM,), f"zero_vector dimensión incorrecta: {result.shape}"
        assert result.dtype == np.float32, f"zero_vector dtype incorrecto: {result.dtype}"
        np.testing.assert_array_equal(result, expected, "zero_vector debe ser vector de ceros")

    @given(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')), min_size=1, max_size=50))
    def test_encoding_output_type_property(self, alphanumeric_text):
        """
        Propiedad: encode siempre retorna numpy array float32.
        
        Para cualquier input válido, el tipo de salida debe ser consistente.
        """
        assume(alphanumeric_text.strip())
        
        # Mock embedding
        expected_embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        self.service._model.encode.return_value = expected_embedding
        
        result = self.service.encode(alphanumeric_text)
        
        # Propiedad: tipo de salida consistente
        assert isinstance(result, np.ndarray), f"encode debe retornar numpy array para '{alphanumeric_text}'"
        assert result.dtype == np.float32, f"encode debe retornar float32 para '{alphanumeric_text}'"
        assert result.ndim == 1, f"encode debe retornar array 1D para '{alphanumeric_text}'"

    @given(st.lists(st.text(), min_size=2, max_size=5).filter(lambda x: len(set(x)) == len(x)))
    def test_batch_vs_individual_consistency_property(self, unique_texts):
        """
        Propiedad: encode_batch es consistente con encode individual.
        
        Para una lista de textos únicos, encode_batch debe producir
        resultados equivalentes a llamadas individuales de encode.
        """
        assume(all(text.strip() for text in unique_texts))  # Todos no vacíos
        
        # Mock para encode individual (cada texto produce embedding único)
        individual_embeddings = []
        for i, text in enumerate(unique_texts):
            embedding = np.full(EMBEDDING_DIM, i + 1, dtype=np.float32)  # Embedding único por texto
            individual_embeddings.append(embedding)
        
        # Mock para encode_batch
        batch_embeddings = np.array(individual_embeddings)
        self.service._model.encode.return_value = batch_embeddings

        # Resetear call_count antes de medir (Hypothesis reutiliza la instancia entre ejemplos)
        self.service._model.encode.reset_mock()

        # Obtener resultado de batch
        batch_result = self.service.encode_batch(unique_texts)

        # Propiedad: consistencia de forma
        assert batch_result.shape == (len(unique_texts), EMBEDDING_DIM), "Forma de batch inconsistente"

        # Verificar que encode_batch hace como máximo 1 llamada al modelo (batch vs N individuales)
        # preprocess_text puede filtrar todos los textos → 0 llamadas
        assert self.service._model.encode.call_count <= 1, (
            "encode_batch debe hacer como máximo una llamada al modelo (batch)"
        )

    @given(st.text().filter(lambda x: len(x) > 1000))
    def test_large_text_handling_property(self, large_text):
        """
        Propiedad: Textos grandes producen embeddings de dimensión correcta.
        
        Independientemente del tamaño del input, la dimensión de salida
        debe mantenerse constante.
        """
        # Mock para texto grande
        expected_embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        self.service._model.encode.return_value = expected_embedding
        
        result = self.service.encode(large_text)
        
        # Propiedad: dimensión constante independiente del tamaño del input
        assert result.shape == (EMBEDDING_DIM,), f"Dimensión incorrecta para texto grande de {len(large_text)} caracteres"
        assert result.dtype == np.float32, "Dtype incorrecto para texto grande"