# -*- coding: utf-8 -*-
"""Tests unitarios para Embedding Pipeline.

Valida el enriquecimiento de features con embeddings semánticos:
- enrich_features con campos nulos produce vector de longitud correcta
- orden temporal se preserva en generate_split_embeddings
- build_text_fields extrae campos correctamente
- serialización/deserialización NPZ funciona correctamente
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from src.features.embedding_pipeline import (
    build_text_fields,
    enrich_features,
    generate_split_embeddings,
    save_embeddings,
    load_embeddings,
    TEXT_FIELD_KEYS
)
from src.config import EMBEDDING_DIM


class TestEmbeddingPipeline:
    """Suite de tests para Embedding Pipeline."""

    def test_build_text_fields_extracts_all_fields(self):
        """Test que build_text_fields extrae todos los campos textuales."""
        row = {
            "diagnostico_texto": "hipertensión arterial",
            "cups_texto": "890201 - consulta médica",
            "observaciones": "paciente estable",
            "valor_factura": 150000,  # campo no textual
            "other_field": "ignored"   # campo no esperado
        }
        
        result = build_text_fields(row)
        
        # Verificar que se extraen exactamente los campos esperados
        assert len(result) == len(TEXT_FIELD_KEYS)
        assert result["diagnostico_texto"] == "hipertensión arterial"
        assert result["cups_texto"] == "890201 - consulta médica"
        assert result["observaciones"] == "paciente estable"

    def test_build_text_fields_handles_missing_fields(self):
        """Test que build_text_fields maneja campos faltantes."""
        row = {
            "diagnostico_texto": "diabetes mellitus",
            # cups_texto faltante
            # observaciones faltante
            "valor_factura": 200000
        }
        
        result = build_text_fields(row)
        
        # Campos faltantes deben ser cadena vacía
        assert result["diagnostico_texto"] == "diabetes mellitus"
        assert result["cups_texto"] == ""
        assert result["observaciones"] == ""

    def test_build_text_fields_handles_none_values(self):
        """Test que build_text_fields maneja valores None."""
        row = {
            "diagnostico_texto": None,
            "cups_texto": None,
            "observaciones": "observación válida"
        }
        
        result = build_text_fields(row)
        
        # Valores None deben convertirse a cadena vacía
        assert result["diagnostico_texto"] == ""
        assert result["cups_texto"] == ""
        assert result["observaciones"] == "observación válida"

    def test_build_text_fields_handles_non_string_values(self):
        """Test que build_text_fields maneja valores no string."""
        row = {
            "diagnostico_texto": 123,  # número
            "cups_texto": [],          # lista
            "observaciones": {"key": "value"}  # diccionario
        }
        
        result = build_text_fields(row)
        
        # Valores no string deben convertirse a cadena vacía
        assert result["diagnostico_texto"] == ""
        assert result["cups_texto"] == ""
        assert result["observaciones"] == ""

    def test_build_text_fields_handles_empty_strings(self):
        """Test que build_text_fields maneja cadenas vacías."""
        row = {
            "diagnostico_texto": "",
            "cups_texto": "   ",  # solo espacios
            "observaciones": "\t\n"  # solo whitespace
        }
        
        result = build_text_fields(row)
        
        # Cadenas vacías o solo whitespace deben mantenerse como están
        # (el preprocesamiento se hace en otro lugar)
        assert result["diagnostico_texto"] == ""
        assert result["cups_texto"] == "   "
        assert result["observaciones"] == "\t\n"

    def test_enrich_features_with_valid_numeric_features(self):
        """Test que enrich_features funciona con features numéricas válidas."""
        # Vector de 48 features numéricas
        numeric_features = np.random.randn(48).astype(np.float32)
        
        # Campos textuales
        text_fields = {
            "diagnostico_texto": "diabetes",
            "cups_texto": "consulta",
            "observaciones": "estable"
        }
        
        # Mock del embedding service
        mock_embedding_service = Mock()
        mock_embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        mock_embedding_service.encode.return_value = mock_embedding
        
        # Mock del pipeline NLP
        mock_nlp_pipeline = Mock()
        
        result = enrich_features(
            numeric_features, text_fields, mock_embedding_service, mock_nlp_pipeline
        )
        
        # Verificar dimensión del vector enriquecido
        expected_length = 48 + EMBEDDING_DIM * len(TEXT_FIELD_KEYS)
        assert result.shape == (expected_length,)
        assert result.dtype == np.float32
        
        # Verificar que se llamó encode para cada campo textual
        assert mock_embedding_service.encode.call_count == len(TEXT_FIELD_KEYS)

    def test_enrich_features_with_incorrect_numeric_dimension_raises_error(self):
        """Test que enrich_features lanza error con dimensión incorreta."""
        # Vector con dimensión incorrecta (no 48)
        numeric_features = np.random.randn(47).astype(np.float32)
        
        text_fields = {"diagnostico_texto": "", "cups_texto": "", "observaciones": ""}
        mock_embedding_service = Mock()
        mock_nlp_pipeline = Mock()
        
        with pytest.raises(ValueError) as exc_info:
            enrich_features(
                numeric_features, text_fields, mock_embedding_service, mock_nlp_pipeline
            )
        
        assert "48 features numéricas" in str(exc_info.value)

    def test_enrich_features_with_all_null_fields_produces_zero_embeddings(self):
        """Test que enrich_features con campos nulos produce vector de longitud correcta."""
        numeric_features = np.random.randn(48).astype(np.float32)
        
        # Todos los campos textuales vacíos
        text_fields = {
            "diagnostico_texto": "",
            "cups_texto": "",
            "observaciones": ""
        }
        
        # Mock embedding service que retorna vector de ceros para campos vacíos
        mock_embedding_service = Mock()
        zero_embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        mock_embedding_service.encode.return_value = zero_embedding
        
        mock_nlp_pipeline = Mock()
        
        result = enrich_features(
            numeric_features, text_fields, mock_embedding_service, mock_nlp_pipeline
        )
        
        # Verificar dimensión correcta
        expected_length = 48 + EMBEDDING_DIM * len(TEXT_FIELD_KEYS)
        assert result.shape == (expected_length,)
        
        # Verificar que la parte de embeddings son ceros
        embeddings_start = 48
        embeddings_end = 48 + EMBEDDING_DIM * len(TEXT_FIELD_KEYS)
        embeddings_part = result[embeddings_start:embeddings_end]
        
        expected_embeddings = np.zeros(EMBEDDING_DIM * len(TEXT_FIELD_KEYS), dtype=np.float32)
        np.testing.assert_array_equal(embeddings_part, expected_embeddings)

    def test_enrich_features_preserves_numeric_features(self):
        """Test que enrich_features preserva las features numéricas originales."""
        # Vector específico de features numéricas
        numeric_features = np.arange(48, dtype=np.float32)
        
        text_fields = {"diagnostico_texto": "", "cups_texto": "", "observaciones": ""}
        
        mock_embedding_service = Mock()
        mock_embedding_service.encode.return_value = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        
        mock_nlp_pipeline = Mock()
        
        result = enrich_features(
            numeric_features, text_fields, mock_embedding_service, mock_nlp_pipeline
        )
        
        # Verificar que las primeras 48 posiciones son las features originales
        np.testing.assert_array_equal(result[:48], numeric_features)

    def test_generate_split_embeddings_preserves_temporal_order(self):
        """Test que generate_split_embeddings preserva el orden temporal."""
        # DataFrame con fechas desordenadas
        df = pd.DataFrame({
            "fecha_radicacion": pd.to_datetime([
                "2025-06-03",
                "2025-06-01",  # Más antigua
                "2025-06-02"
            ]),
            "diagnostico_texto": [
                "diabetes",
                "hipertensión",
                "neumonía"
            ]
        })
        
        # Mock embedding service
        mock_embedding_service = Mock()
        # Embeddings únicos para verificar orden
        embeddings = np.array([
            np.full(EMBEDDING_DIM, 1.0, dtype=np.float32),  # hipertensión (fecha más antigua)
            np.full(EMBEDDING_DIM, 2.0, dtype=np.float32),  # neumonía
            np.full(EMBEDDING_DIM, 3.0, dtype=np.float32)   # diabetes (fecha más reciente)
        ])
        mock_embedding_service.encode_batch.return_value = embeddings
        
        result = generate_split_embeddings(df, "diagnostico_texto", mock_embedding_service)
        
        # Verificar forma
        assert result.shape == (3, EMBEDDING_DIM)
        
        # Verificar que se llamó encode_batch con textos en orden temporal
        mock_embedding_service.encode_batch.assert_called_once()
        called_texts = mock_embedding_service.encode_batch.call_args[0][0]
        
        # Orden esperado: hipertensión (2025-06-01), neumonía (2025-06-02), diabetes (2025-06-03)
        expected_order = ["hipertensión", "neumonía", "diabetes"]
        assert called_texts == expected_order

    def test_generate_split_embeddings_handles_missing_fecha_radicacion(self):
        """Test que generate_split_embeddings maneja fecha_radicacion faltante."""
        df = pd.DataFrame({
            "diagnostico_texto": ["diabetes", "hipertensión"]
            # fecha_radicacion faltante
        })
        
        mock_embedding_service = Mock()
        
        with pytest.raises(KeyError) as exc_info:
            generate_split_embeddings(df, "diagnostico_texto", mock_embedding_service)
        
        assert "fecha_radicacion" in str(exc_info.value)

    def test_generate_split_embeddings_handles_missing_split_col(self):
        """Test que generate_split_embeddings maneja columna textual faltante."""
        df = pd.DataFrame({
            "fecha_radicacion": pd.to_datetime(["2025-06-01", "2025-06-02"])
            # diagnostico_texto faltante
        })
        
        mock_embedding_service = Mock()
        
        with pytest.raises(KeyError) as exc_info:
            generate_split_embeddings(df, "diagnostico_texto", mock_embedding_service)
        
        assert "diagnostico_texto" in str(exc_info.value)

    def test_generate_split_embeddings_handles_null_values(self):
        """Test que generate_split_embeddings maneja valores nulos."""
        df = pd.DataFrame({
            "fecha_radicacion": pd.to_datetime(["2025-06-01", "2025-06-02", "2025-06-03"]),
            "diagnostico_texto": ["diabetes", None, "hipertensión"]
        })
        
        mock_embedding_service = Mock()
        mock_embeddings = np.random.randn(3, EMBEDDING_DIM).astype(np.float32)
        mock_embedding_service.encode_batch.return_value = mock_embeddings
        
        result = generate_split_embeddings(df, "diagnostico_texto", mock_embedding_service)
        
        assert result.shape == (3, EMBEDDING_DIM)
        
        # Verificar que se llamó con valores None convertidos a ""
        called_texts = mock_embedding_service.encode_batch.call_args[0][0]
        expected_texts = ["diabetes", "", "hipertensión"]
        assert called_texts == expected_texts

    def test_save_and_load_embeddings_roundtrip(self):
        """Test que save_embeddings y load_embeddings funcionan correctamente."""
        # Crear embeddings de prueba
        original_embeddings = np.random.randn(100, EMBEDDING_DIM).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_embeddings"
            
            # Guardar embeddings
            save_embeddings(original_embeddings, file_path)
            
            # Verificar que el archivo se creó
            assert (file_path.parent / f"{file_path.name}.npz").exists()
            
            # Cargar embeddings
            loaded_embeddings = load_embeddings(file_path)
            
            # Verificar que son idénticos
            np.testing.assert_array_equal(original_embeddings, loaded_embeddings)
            assert loaded_embeddings.shape == original_embeddings.shape
            assert loaded_embeddings.dtype == original_embeddings.dtype

    def test_save_embeddings_with_custom_key(self):
        """Test que save_embeddings funciona con clave personalizada."""
        embeddings = np.random.randn(50, EMBEDDING_DIM).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "custom_key_test"
            custom_key = "custom_embeddings"
            
            save_embeddings(embeddings, file_path, key=custom_key)
            
            # Cargar con clave personalizada
            loaded_embeddings = load_embeddings(file_path, key=custom_key)
            
            np.testing.assert_array_equal(embeddings, loaded_embeddings)

    def test_load_embeddings_with_nonexistent_file_raises_error(self):
        """Test que load_embeddings lanza error con archivo inexistente."""
        nonexistent_path = Path("/nonexistent/path/embeddings")
        
        with pytest.raises(FileNotFoundError) as exc_info:
            load_embeddings(nonexistent_path)
        
        assert str(nonexistent_path) + ".npz" in str(exc_info.value)

    def test_load_embeddings_with_nonexistent_key_raises_error(self):
        """Test que load_embeddings lanza error con clave inexistente."""
        import gc
        embeddings = np.random.randn(10, EMBEDDING_DIM).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_key_error"
            save_embeddings(embeddings, file_path)
            
            with pytest.raises(KeyError) as exc_info:
                load_embeddings(file_path, key="nonexistent_key")
            
            gc.collect()  # Forzar cierre de archivos en Windows
        
        assert "nonexistent_key" in str(exc_info.value)

    def test_save_embeddings_creates_parent_directories(self):
        """Test que save_embeddings crea directorios padre si no existen."""
        embeddings = np.random.randn(10, EMBEDDING_DIM).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "nested" / "directories" / "embeddings"
            
            # El directorio padre no existe
            assert not nested_path.parent.exists()
            
            save_embeddings(embeddings, nested_path)
            
            # Verificar que se crearon los directorios y el archivo
            assert nested_path.parent.exists()
            assert (nested_path.parent / f"{nested_path.name}.npz").exists()

    def test_text_field_keys_constant(self):
        """Test que TEXT_FIELD_KEYS contiene los campos esperados."""
        expected_keys = ["diagnostico_texto", "cups_texto", "observaciones"]
        
        assert TEXT_FIELD_KEYS == expected_keys
        assert len(TEXT_FIELD_KEYS) == 3
        
        for key in expected_keys:
            assert key in TEXT_FIELD_KEYS

    def test_enrich_features_calls_encode_in_correct_order(self):
        """Test que enrich_features llama encode en el orden correcto."""
        numeric_features = np.random.randn(48).astype(np.float32)
        text_fields = {
            "diagnostico_texto": "diabetes",
            "cups_texto": "consulta",
            "observaciones": "estable"
        }
        
        mock_embedding_service = Mock()
        mock_embedding_service.encode.return_value = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        mock_nlp_pipeline = Mock()
        
        enrich_features(numeric_features, text_fields, mock_embedding_service, mock_nlp_pipeline)
        
        # Verificar que se llamó encode para cada campo en el orden de TEXT_FIELD_KEYS
        actual_calls = [call.args[0] for call in mock_embedding_service.encode.call_args_list]
        expected_calls = ["diabetes", "consulta", "estable"]
        assert actual_calls == expected_calls

    def test_generate_split_embeddings_with_empty_dataframe(self):
        """Test que generate_split_embeddings maneja DataFrame vacío."""
        df = pd.DataFrame({
            "fecha_radicacion": pd.to_datetime([]),
            "diagnostico_texto": []
        })
        
        mock_embedding_service = Mock()
        mock_embedding_service.encode_batch.return_value = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        
        result = generate_split_embeddings(df, "diagnostico_texto", mock_embedding_service)
        
        assert result.shape == (0, EMBEDDING_DIM)
        assert result.dtype == np.float32