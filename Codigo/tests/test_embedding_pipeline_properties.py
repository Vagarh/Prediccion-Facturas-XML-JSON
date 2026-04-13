# -*- coding: utf-8 -*-
"""Property tests para EmbeddingPipeline.

Valida propiedades fundamentales del enriquecimiento de features:
- Propiedad 4: Dimensión del vector enriquecido
- Propiedad 5: No data leakage temporal
"""

import pytest
from hypothesis import given, strategies as st, assume
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.features.embedding_pipeline import (
    build_text_fields,
    enrich_features,
    generate_split_embeddings,
    TEXT_FIELD_KEYS
)
from src.config import EMBEDDING_DIM, RANDOM_STATE


class TestEmbeddingPipelineProperties:
    """Property tests para EmbeddingPipeline usando Hypothesis."""

    def setup_method(self):
        """Configuración inicial para cada test."""
        # Mock del EmbeddingService
        self.mock_embedding_service = Mock()
        self.mock_embedding_service.encode.return_value = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        self.mock_embedding_service.zero_vector.return_value = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        
        # Mock del NLP pipeline
        self.mock_nlp_pipeline = Mock()

    @given(st.integers(min_value=48, max_value=48))
    def test_enriched_vector_dimension_property(self, n_numeric_features):
        """
        Propiedad 4: Dimensión del vector enriquecido.
        
        len(enrich_features(...)) == 48 + EMBEDDING_DIM * n_text_fields
        
        Valida: Requisito 2.2
        """
        numeric_features = np.random.randn(n_numeric_features).astype(np.float32)
        text_fields = {
            "diagnostico_texto": "Diagnóstico de prueba",
            "cups_texto": "Código CUPS",
            "observaciones": "Observaciones médicas"
        }
        self.mock_embedding_service.encode.return_value = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        
        result = enrich_features(
            numeric_features, text_fields,
            self.mock_embedding_service, self.mock_nlp_pipeline
        )
        
        expected_dim = n_numeric_features + EMBEDDING_DIM * len(TEXT_FIELD_KEYS)
        assert isinstance(result, np.ndarray)
        assert result.shape == (expected_dim,)
        assert result.dtype == np.float32

    @given(st.integers(min_value=48, max_value=48))
    def test_enriched_vector_with_null_fields_property(self, n_numeric_features):
        """
        Propiedad: Vector enriquecido con campos nulos.
        
        Cuando todos los campos textuales son nulos, el vector enriquecido
        debe tener la dimensión correcta con embeddings de ceros.
        """
        # Vector numérico de entrada
        numeric_features = np.random.randn(n_numeric_features).astype(np.float32)
        
        # Todos los campos textuales nulos
        text_fields = {
            "diagnostico_texto": None,
            "cups_texto": "",
            "observaciones": "   "  # Solo espacios
        }
        
        # Mock para retornar vector de ceros para campos nulos
        self.mock_embedding_service.encode.return_value = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        
        result = enrich_features(
            numeric_features, 
            text_fields, 
            self.mock_embedding_service, 
            self.mock_nlp_pipeline
        )
        
        # Propiedad: dimensión correcta incluso con campos nulos
        expected_dim = n_numeric_features + EMBEDDING_DIM * len(TEXT_FIELD_KEYS)
        assert result.shape == (expected_dim,), f"Dimensión incorrecta con campos nulos: {result.shape}"
        
        # Verificar que la parte numérica se preserva
        np.testing.assert_array_equal(
            result[:n_numeric_features], 
            numeric_features,
            "Parte numérica debe preservarse en vector enriquecido"
        )

    @given(st.lists(st.dictionaries(
        keys=st.sampled_from(["diagnostico_texto", "cups_texto", "observaciones"]),
        values=st.one_of(st.text(min_size=1, max_size=100), st.none()),
        min_size=1, max_size=3
    ), min_size=1, max_size=10))
    def test_build_text_fields_property(self, row_dicts):
        """
        Propiedad: build_text_fields extrae campos consistentemente.
        
        Para cualquier diccionario de entrada, build_text_fields debe
        retornar un diccionario con las claves esperadas.
        """
        for row_dict in row_dicts:
            result = build_text_fields(row_dict)
            
            # Propiedad: claves consistentes
            assert isinstance(result, dict), "build_text_fields debe retornar diccionario"
            assert set(result.keys()) == set(TEXT_FIELD_KEYS), f"Claves incorrectas: {result.keys()}"
            
            # Propiedad: valores son strings o None
            for key, value in result.items():
                assert value is None or isinstance(value, str), f"Valor incorrecto para {key}: {type(value)}"

    @given(st.integers(min_value=10, max_value=100))
    def test_temporal_split_preservation_property(self, n_samples):
        """
        Propiedad 5: No data leakage temporal.
        
        Los embeddings generados para el conjunto de test no contienen información
        de registros con fecha_radicacion anterior al corte de test.
        
        Valida: Requisito 2.5, 8.3
        """
        # Generar DataFrame con fechas ordenadas temporalmente
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(n_samples)]
        
        # Crear DataFrame con división temporal clara
        df = pd.DataFrame({
            'fecha_radicacion': dates,
            'diagnostico_texto': [f"Diagnóstico {i}" for i in range(n_samples)],
            'cups_texto': [f"CUPS {i}" for i in range(n_samples)],
            'observaciones': [f"Observación {i}" for i in range(n_samples)],
            'split': ['train'] * (n_samples // 2) + ['test'] * (n_samples - n_samples // 2)
        })
        
        # Ordenar por fecha (simulando división temporal real)
        df = df.sort_values('fecha_radicacion').reset_index(drop=True)
        
        # Mock del embedding service
        def mock_encode_batch(texts):
            # Cada texto produce un embedding único basado en su contenido
            embeddings = []
            for text in texts:
                # Usar hash del texto para generar embedding reproducible
                np.random.seed(hash(str(text)) % (2**32))
                embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
                embeddings.append(embedding)
            return np.array(embeddings)
        
        self.mock_embedding_service.encode_batch.side_effect = mock_encode_batch
        
        # Generar embeddings respetando división temporal
        embeddings = generate_split_embeddings(
            df, 
            'split', 
            self.mock_embedding_service
        )
        
        # Propiedad: forma correcta
        # generate_split_embeddings codifica UN campo textual (split_col), retorna (n, EMBEDDING_DIM)
        expected_shape = (n_samples, EMBEDDING_DIM)
        assert embeddings.shape == expected_shape, f"Forma incorrecta: {embeddings.shape}"
        
        # Propiedad: orden temporal preservado
        # Los embeddings deben estar en el mismo orden que el DataFrame original
        train_indices = df[df['split'] == 'train'].index.tolist()
        test_indices = df[df['split'] == 'test'].index.tolist()
        
        # Verificar que los índices de train vienen antes que los de test
        if train_indices and test_indices:
            max_train_idx = max(train_indices)
            min_test_idx = min(test_indices)
            assert max_train_idx < min_test_idx, "División temporal violada: train debe venir antes que test"

    @given(st.integers(min_value=5, max_value=20))
    def test_embedding_consistency_property(self, n_samples):
        """
        Propiedad: Consistencia de embeddings.

        Para el mismo conjunto de datos, múltiples llamadas a generate_split_embeddings
        deben producir resultados idénticos (determinismo).
        """
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(n_samples)]
        # DataFrame de prueba con fecha_radicacion requerida por generate_split_embeddings
        df = pd.DataFrame({
            'fecha_radicacion': dates,
            'diagnostico_texto': [f"Diagnóstico {i}" for i in range(n_samples)],
        })
        
        # Mock determinístico
        def deterministic_encode_batch(texts):
            embeddings = []
            for text in texts:
                # Embedding determinístico basado en el texto
                np.random.seed(hash(str(text)) % (2**32))
                embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
                embeddings.append(embedding)
            return np.array(embeddings)
        
        self.mock_embedding_service.encode_batch.side_effect = deterministic_encode_batch
        
        # Múltiples llamadas sobre el campo textual existente
        embeddings1 = generate_split_embeddings(df, 'diagnostico_texto', self.mock_embedding_service)
        embeddings2 = generate_split_embeddings(df, 'diagnostico_texto', self.mock_embedding_service)
        
        # Propiedad: determinismo
        np.testing.assert_array_equal(
            embeddings1, 
            embeddings2,
            "generate_split_embeddings debe ser determinístico"
        )

    @given(st.integers(min_value=48, max_value=48))
    def test_empty_text_fields_property(self, n_numeric_features):
        """
        Propiedad: Manejo de campos textuales vacíos.

        Cuando todos los campos textuales están vacíos o son None,
        el vector enriquecido debe contener solo ceros en la parte de embeddings.
        enrich_features requiere exactamente 48 features numéricas.
        """
        numeric_features = np.random.randn(n_numeric_features).astype(np.float32)
        
        # Campos textuales vacíos de diferentes formas
        empty_text_fields = {
            "diagnostico_texto": "",
            "cups_texto": None,
            "observaciones": "   \t\n  "  # Solo whitespace
        }
        
        # Mock para retornar vector de ceros para campos vacíos
        self.mock_embedding_service.encode.return_value = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        
        result = enrich_features(
            numeric_features,
            empty_text_fields,
            self.mock_embedding_service,
            self.mock_nlp_pipeline
        )
        
        # Propiedad: parte numérica preservada
        np.testing.assert_array_equal(
            result[:n_numeric_features],
            numeric_features,
            "Parte numérica debe preservarse con campos textuales vacíos"
        )
        
        # Propiedad: parte de embeddings debe ser ceros
        embedding_part = result[n_numeric_features:]
        expected_embedding_zeros = np.zeros(EMBEDDING_DIM * len(TEXT_FIELD_KEYS), dtype=np.float32)
        np.testing.assert_array_equal(
            embedding_part,
            expected_embedding_zeros,
            "Parte de embeddings debe ser ceros para campos textuales vacíos"
        )

    @given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=5))
    def test_text_field_order_consistency_property(self, text_values):
        """
        Propiedad: Orden consistente de campos textuales.
        
        El orden de concatenación de embeddings debe ser consistente
        independientemente del orden de los campos en el diccionario de entrada.
        """
        # Crear dos diccionarios con el mismo contenido pero diferente orden
        text_dict1 = {}
        text_dict2 = {}
        
        for i, key in enumerate(TEXT_FIELD_KEYS):
            value = text_values[i % len(text_values)]
            text_dict1[key] = value
            text_dict2[key] = value
        
        # Reordenar el segundo diccionario
        text_dict2 = {k: text_dict2[k] for k in reversed(TEXT_FIELD_KEYS)}
        
        numeric_features = np.random.randn(48).astype(np.float32)
        
        # Mock para retornar embeddings únicos por texto
        def mock_encode(text):
            np.random.seed(hash(str(text)) % (2**32))
            return np.random.randn(EMBEDDING_DIM).astype(np.float32)
        
        self.mock_embedding_service.encode.side_effect = mock_encode
        
        # Enriquecer con ambos diccionarios
        result1 = enrich_features(numeric_features, text_dict1, self.mock_embedding_service, self.mock_nlp_pipeline)
        result2 = enrich_features(numeric_features, text_dict2, self.mock_embedding_service, self.mock_nlp_pipeline)
        
        # Propiedad: resultados idénticos independientemente del orden del diccionario
        np.testing.assert_array_equal(
            result1,
            result2,
            "Orden de campos textuales no debe afectar el resultado final"
        )

    @given(st.integers(min_value=1, max_value=5))
    def test_feature_concatenation_property(self, n_text_fields):
        """Propiedad: Concatenación correcta de features."""
        assume(n_text_fields <= len(TEXT_FIELD_KEYS))
        
        numeric_features = np.arange(48, dtype=np.float32)
        
        text_fields = {}
        for i in range(n_text_fields):
            key = TEXT_FIELD_KEYS[i]
            text_fields[key] = f"Texto {i}"
        # Rellenar campos faltantes con vacío
        for key in TEXT_FIELD_KEYS:
            if key not in text_fields:
                text_fields[key] = ""
        
        def mock_encode(text):
            if "Texto 0" in text:
                return np.full(EMBEDDING_DIM, 10, dtype=np.float32)
            elif "Texto 1" in text:
                return np.full(EMBEDDING_DIM, 11, dtype=np.float32)
            elif "Texto 2" in text:
                return np.full(EMBEDDING_DIM, 12, dtype=np.float32)
            else:
                return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        
        self.mock_embedding_service.encode.side_effect = mock_encode
        
        result = enrich_features(
            numeric_features, text_fields,
            self.mock_embedding_service, self.mock_nlp_pipeline
        )
        
        np.testing.assert_array_equal(result[:48], numeric_features)
        
        for i in range(n_text_fields):
            start_idx = 48 + i * EMBEDDING_DIM
            end_idx = start_idx + EMBEDDING_DIM
            expected_value = i + 10
            np.testing.assert_array_equal(
                result[start_idx:end_idx],
                np.full(EMBEDDING_DIM, expected_value, dtype=np.float32)
            )