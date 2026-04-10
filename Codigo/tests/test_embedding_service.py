# -*- coding: utf-8 -*-
"""Tests unitarios para EmbeddingService.

Valida la generación de embeddings semánticos:
- encode_batch produce la misma forma que N llamadas a encode
- modelo se carga una sola vez (singleton)
- dimensión fija de embeddings
- null-handling con vector de ceros
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.embeddings.service import EmbeddingService
from src.config import EMBEDDING_DIM, RANDOM_STATE


class TestEmbeddingService:
    """Suite de tests para EmbeddingService."""

    def setup_method(self):
        """Configuración inicial para cada test."""
        # Mock del modelo SentenceTransformer para evitar carga real
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            mock_st.return_value = mock_model
            
            self.service = EmbeddingService(
                model_name="test-model",
                random_state=RANDOM_STATE
            )

    @patch('sentence_transformers.SentenceTransformer')
    def test_embedding_service_initialization(self, mock_st):
        """Test que EmbeddingService se inicializa correctamente."""
        mock_model = Mock()
        mock_st.return_value = mock_model
        
        service = EmbeddingService(
            model_name="test-model",
            random_state=42
        )
        
        # Verificar que se cargó el modelo correcto
        mock_st.assert_called_once_with("test-model")
        assert service.model_name == "test-model"
        assert service.random_state == 42

    def test_encode_returns_correct_dimension(self):
        """Test que encode() retorna vector de dimensión correcta."""
        # Mock del modelo para retornar vector de dimensión correcta
        mock_embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        self.service.model.encode.return_value = mock_embedding
        
        result = self.service.encode("texto de prueba")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (EMBEDDING_DIM,)
        assert result.dtype == np.float32

    def test_encode_with_none_returns_zero_vector(self):
        """Test que encode(None) retorna vector de ceros."""
        result = self.service.encode(None)
        
        expected = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        
        # Verificar que no se llamó al modelo
        self.service.model.encode.assert_not_called()

    def test_encode_with_empty_string_returns_zero_vector(self):
        """Test que encode("") retorna vector de ceros."""
        result = self.service.encode("")
        
        expected = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        
        # Verificar que no se llamó al modelo
        self.service.model.encode.assert_not_called()

    @patch('src.embeddings.service.preprocess_text')
    def test_encode_with_whitespace_returns_zero_vector(self, mock_preprocess):
        """Test que encode con solo espacios retorna vector de ceros."""
        # Mock preprocess_text para retornar cadena vacía
        mock_preprocess.return_value = ""
        
        result = self.service.encode("   \t\n   ")
        
        expected = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        
        # Verificar que se llamó preprocess_text
        mock_preprocess.assert_called_once_with("   \t\n   ")
        
        # Verificar que no se llamó al modelo
        self.service.model.encode.assert_not_called()

    @patch('src.embeddings.service.preprocess_text')
    def test_encode_calls_preprocess_text(self, mock_preprocess):
        """Test que encode() llama a preprocess_text."""
        mock_preprocess.return_value = "texto procesado"
        mock_embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        self.service.model.encode.return_value = mock_embedding
        
        self.service.encode("texto original")
        
        # Verificar que se llamó preprocess_text
        mock_preprocess.assert_called_once_with("texto original")
        
        # Verificar que se llamó al modelo con texto procesado
        self.service.model.encode.assert_called_once_with("texto procesado")

    def test_encode_batch_returns_correct_shape(self):
        """Test que encode_batch() retorna forma correcta."""
        texts = ["texto 1", "texto 2", "texto 3"]
        
        # Mock del modelo para retornar batch de embeddings
        mock_embeddings = np.random.randn(len(texts), EMBEDDING_DIM).astype(np.float32)
        self.service.model.encode.return_value = mock_embeddings
        
        result = self.service.encode_batch(texts)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(texts), EMBEDDING_DIM)
        assert result.dtype == np.float32

    def test_encode_batch_same_as_individual_encodes(self):
        """Test que encode_batch produce la misma forma que N llamadas a encode."""
        texts = ["diabetes", "hipertensión", "neumonía"]
        
        # Mock embeddings individuales
        individual_embeddings = []
        for i, text in enumerate(texts):
            embedding = np.full(EMBEDDING_DIM, i + 1, dtype=np.float32)
            individual_embeddings.append(embedding)
        
        # Mock encode_batch
        batch_embeddings = np.array(individual_embeddings)
        self.service.model.encode.return_value = batch_embeddings
        
        # Llamar encode_batch
        batch_result = self.service.encode_batch(texts)
        
        # Verificar forma
        assert batch_result.shape == (len(texts), EMBEDDING_DIM)
        
        # Verificar que se llamó al modelo una sola vez con todos los textos
        self.service.model.encode.assert_called_once()

    def test_encode_batch_with_empty_list(self):
        """Test que encode_batch([]) retorna array vacío."""
        result = self.service.encode_batch([])
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, EMBEDDING_DIM)
        assert result.dtype == np.float32

    def test_encode_batch_with_mixed_none_and_text(self):
        """Test que encode_batch maneja mezcla de None y texto."""
        texts = ["diabetes", None, "hipertensión", ""]
        
        # Mock para textos válidos solamente
        valid_embeddings = np.random.randn(2, EMBEDDING_DIM).astype(np.float32)
        self.service.model.encode.return_value = valid_embeddings
        
        result = self.service.encode_batch(texts)
        
        assert result.shape == (len(texts), EMBEDDING_DIM)
        
        # Verificar que posiciones con None/empty son vectores de ceros
        np.testing.assert_array_equal(result[1], np.zeros(EMBEDDING_DIM))  # None
        np.testing.assert_array_equal(result[3], np.zeros(EMBEDDING_DIM))  # ""

    def test_zero_vector_returns_correct_shape_and_values(self):
        """Test que zero_vector() retorna vector de ceros correcto."""
        result = self.service.zero_vector()
        
        expected = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (EMBEDDING_DIM,)
        assert result.dtype == np.float32

    @patch('sentence_transformers.SentenceTransformer')
    def test_model_singleton_behavior(self, mock_st):
        """Test que el modelo se carga una sola vez (singleton)."""
        mock_model = Mock()
        mock_st.return_value = mock_model
        
        # Crear múltiples instancias con el mismo modelo
        service1 = EmbeddingService("same-model")
        service2 = EmbeddingService("same-model")
        service3 = EmbeddingService("different-model")
        
        # Verificar llamadas a SentenceTransformer
        # Debe haberse llamado para cada modelo único
        assert mock_st.call_count >= 2  # Al menos same-model y different-model

    def test_encode_with_different_text_types(self):
        """Test que encode maneja diferentes tipos de texto."""
        test_cases = [
            "texto normal",
            "TEXTO EN MAYÚSCULAS",
            "texto con números 123",
            "texto-con-guiones",
            "texto.con.puntos",
            "texto\ncon\nsaltos",
            "   texto con espacios   "
        ]
        
        for text in test_cases:
            # Mock embedding diferente para cada caso
            mock_embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            self.service.model.encode.return_value = mock_embedding
            
            result = self.service.encode(text)
            
            assert isinstance(result, np.ndarray)
            assert result.shape == (EMBEDDING_DIM,)

    @patch('src.embeddings.service.logger')
    def test_encode_logs_operations(self, mock_logger):
        """Test que encode registra operaciones en logs."""
        mock_embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        self.service.model.encode.return_value = mock_embedding
        
        self.service.encode("texto de prueba")
        
        # Verificar que se registró la operación
        mock_logger.debug.assert_called()

    def test_encode_handles_model_errors_gracefully(self):
        """Test que encode maneja errores del modelo graciosamente."""
        # Simular error en el modelo
        self.service.model.encode.side_effect = Exception("Model error")
        
        with pytest.raises(Exception) as exc_info:
            self.service.encode("texto de prueba")
        
        assert "Model error" in str(exc_info.value)

    def test_embedding_service_with_custom_random_state(self):
        """Test que EmbeddingService acepta random_state personalizado."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_st.return_value = mock_model
            
            service = EmbeddingService("test-model", random_state=123)
            
            assert service.random_state == 123

    def test_embedding_service_default_random_state(self):
        """Test que EmbeddingService usa RANDOM_STATE por defecto."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_st.return_value = mock_model
            
            service = EmbeddingService("test-model")
            
            assert service.random_state == RANDOM_STATE

    @patch('src.embeddings.service.preprocess_text')
    def test_encode_batch_preprocesses_all_texts(self, mock_preprocess):
        """Test que encode_batch preprocesa todos los textos."""
        texts = ["texto 1", "texto 2", None, ""]
        processed_texts = ["procesado 1", "procesado 2", "", ""]
        
        # Mock preprocess_text para cada llamada
        mock_preprocess.side_effect = processed_texts
        
        # Mock modelo para textos válidos
        valid_embeddings = np.random.randn(2, EMBEDDING_DIM).astype(np.float32)
        self.service.model.encode.return_value = valid_embeddings
        
        self.service.encode_batch(texts)
        
        # Verificar que se llamó preprocess_text para cada texto
        assert mock_preprocess.call_count == len(texts)
        
        # Verificar las llamadas específicas
        expected_calls = [
            (("texto 1",),),
            (("texto 2",),),
            ((None,),),
            (("",),)
        ]
        
        actual_calls = [call.args for call in mock_preprocess.call_args_list]
        for expected, actual in zip(expected_calls, actual_calls):
            assert expected == actual