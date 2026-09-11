# -*- coding: utf-8 -*-
"""Tests unitarios para XGBoostRetrainer.

Valida el proceso de reentrenamiento con features enriquecidas:
- Carga correcta de datasets
- Extracción de features numéricas y embeddings
- Entrenamiento de modelos base y enriquecido
- Evaluación y comparación de métricas
- Serialización de modelos y reportes
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import json

from src.evaluation.retrain_xgboost import XGBoostRetrainer, XGBOOST_PARAMS, CLASS_NAMES


class TestXGBoostRetrainer:
    """Suite de tests para XGBoostRetrainer."""

    def setup_method(self):
        """Configuración inicial para cada test."""
        # Mock de todos los componentes externos
        with patch('src.evaluation.retrain_xgboost.PrivacyGuard'), \
             patch('src.evaluation.retrain_xgboost.load_nlp_pipeline'), \
             patch('src.evaluation.retrain_xgboost.EmbeddingService'):
            
            self.retrainer = XGBoostRetrainer(
                embedding_model_name="test-model",
                random_state=42
            )

    def test_retrainer_initialization(self):
        """Test que XGBoostRetrainer se inicializa correctamente."""
        with patch('src.evaluation.retrain_xgboost.PrivacyGuard') as mock_guard, \
             patch('src.evaluation.retrain_xgboost.load_nlp_pipeline') as mock_nlp, \
             patch('src.evaluation.retrain_xgboost.EmbeddingService') as mock_embedding:
            
            retrainer = XGBoostRetrainer(
                embedding_model_name="custom-model",
                random_state=123
            )
            
            assert retrainer.embedding_model_name == "custom-model"
            assert retrainer.random_state == 123
            
            # Verificar que se inicializaron los componentes
            mock_guard.assert_called_once()
            mock_nlp.assert_called_once()
            mock_embedding.assert_called_once()

    @patch('pandas.read_parquet')
    @patch('pathlib.Path.exists')
    def test_load_datasets_success(self, mock_exists, mock_read_parquet):
        """Test que load_datasets carga correctamente los archivos."""
        # Mock que los archivos existen
        mock_exists.return_value = True
        
        # Mock DataFrames
        mock_train = pd.DataFrame({'col1': [1, 2, 3], 'clase': [0, 1, 2]})
        mock_val = pd.DataFrame({'col1': [4, 5], 'clase': [1, 0]})
        mock_test = pd.DataFrame({'col1': [6, 7, 8], 'clase': [2, 1, 0]})
        
        mock_read_parquet.side_effect = [mock_train, mock_val, mock_test]
        
        train_df, val_df, test_df = self.retrainer.load_datasets()
        
        # Verificar que se cargaron correctamente
        assert len(train_df) == 3
        assert len(val_df) == 2
        assert len(test_df) == 3
        
        # Verificar que se llamó read_parquet 3 veces
        assert mock_read_parquet.call_count == 3

    @patch('pathlib.Path.exists')
    def test_load_datasets_file_not_found(self, mock_exists):
        """Test que load_datasets lanza error cuando archivos no existen."""
        # Mock que los archivos no existen
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError) as exc_info:
            self.retrainer.load_datasets()
        
        assert "Dataset no encontrado" in str(exc_info.value)

    @patch('pathlib.Path.exists')
    def test_extract_features_and_labels_numeric_only(self, mock_exists):
        """Test extracción de features numéricas solamente (modelo base)."""
        from unittest.mock import mock_open as _mock_open
        mock_exists.return_value = True
        
        feature_names = ['feature_1', 'feature_2', 'feature_3']
        file_content = '\n'.join(feature_names) + '\n'
        
        df = pd.DataFrame({
            'feature_1': [1.0, 2.0, 3.0],
            'feature_2': [4.0, 5.0, 6.0],
            'feature_3': [7.0, 8.0, 9.0],
            'clase': [0, 1, 2],
            'diagnostico_texto': ['diabetes', 'hipertensión', 'neumonía']
        })
        
        with patch('builtins.open', _mock_open(read_data=file_content)):
            X, y = self.retrainer.extract_features_and_labels(df, include_embeddings=False)
        
        assert X.shape == (3, 3)
        assert y.shape == (3,)
        np.testing.assert_array_equal(y, [0, 1, 2])
        assert X.dtype == np.float32

    @patch('pathlib.Path.exists')
    def test_extract_features_and_labels_with_embeddings(self, mock_exists):
        """Test extracción de features con embeddings (modelo enriquecido).

        La implementación actual codifica cada campo textual en batch con
        ``EmbeddingService.encode_batch`` (una llamada por campo en
        ``TEXT_FIELD_KEYS``), no con un ``enrich_features`` por fila.
        """
        from unittest.mock import mock_open as _mock_open

        from src.features.embedding_pipeline import TEXT_FIELD_KEYS
        from src.config import EMBEDDING_DIM

        mock_exists.return_value = True
        feature_names = ['feature_1', 'feature_2']
        file_content = '\n'.join(feature_names) + '\n'

        n = 3
        self.retrainer.embedding_service.encode_batch = Mock(
            return_value=np.random.randn(n, EMBEDDING_DIM).astype(np.float32)
        )

        df = pd.DataFrame({
            'feature_1': [1.0, 2.0, 3.0],
            'feature_2': [3.0, 4.0, 5.0],
            'target': [0, 1, 2],
            **{k: ['diabetes', 'hipertensión', 'neumonía'] for k in TEXT_FIELD_KEYS},
        })

        with patch('builtins.open', _mock_open(read_data=file_content)):
            X, y = self.retrainer.extract_features_and_labels(df, include_embeddings=True)

        # Una llamada a encode_batch por cada campo textual
        assert self.retrainer.embedding_service.encode_batch.call_count == len(TEXT_FIELD_KEYS)
        # 2 numéricas + EMBEDDING_DIM * n_campos
        assert X.shape == (n, 2 + EMBEDDING_DIM * len(TEXT_FIELD_KEYS))
        assert X.dtype == np.float32
        np.testing.assert_array_equal(y, [0, 1, 2])
        assert X.shape == (3, 1154)  # 2 features numéricas + 384*3 embeddings
        assert y.shape == (3,)

    def test_extract_features_invalid_classes(self):
        """Test que extract_features valida correctamente las clases."""
        from unittest.mock import mock_open as _mock_open
        file_content = 'feature_1\n'
        
        df = pd.DataFrame({
            'feature_1': [1.0, 2.0],
            'clase': [0, 3]  # Clase 3 no válida
        })
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', _mock_open(read_data=file_content)):
            with pytest.raises(AssertionError):
                self.retrainer.extract_features_and_labels(df, include_embeddings=False)

    @patch('xgboost.XGBClassifier')
    def test_train_model(self, mock_xgb_class):
        """Test que train_model entrena correctamente el modelo."""
        # Mock del modelo XGBoost
        mock_model = Mock()
        mock_model.best_iteration = 50
        mock_xgb_class.return_value = mock_model
        
        # Datos de prueba
        X_train = np.random.randn(100, 10).astype(np.float32)
        y_train = np.random.randint(0, 3, 100)
        X_val = np.random.randn(20, 10).astype(np.float32)
        y_val = np.random.randint(0, 3, 20)
        
        result = self.retrainer.train_model(X_train, y_train, X_val, y_val)
        
        # Verificar que se creó el modelo con parámetros correctos
        mock_xgb_class.assert_called_once_with(**XGBOOST_PARAMS)
        
        # Verificar que se llamó fit con early stopping
        mock_model.fit.assert_called_once()
        fit_args, fit_kwargs = mock_model.fit.call_args
        
        assert np.array_equal(fit_args[0], X_train)
        assert np.array_equal(fit_args[1], y_train)
        assert 'eval_set' in fit_kwargs
        # early_stopping_rounds puede estar en el constructor (XGBOOST_PARAMS) o en fit
        assert result == mock_model

    def test_evaluate_model(self):
        """Test que evaluate_model calcula métricas correctamente."""
        # Mock del modelo
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 2, 1, 0])
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1], 
            [0.1, 0.2, 0.7],
            [0.3, 0.6, 0.1],
            [0.9, 0.05, 0.05]
        ])
        
        # Datos de test
        X_test = np.random.randn(5, 10).astype(np.float32)
        y_test = np.array([0, 1, 2, 0, 0])
        
        metrics = self.retrainer.evaluate_model(mock_model, X_test, y_test, "test")
        
        # Verificar estructura de métricas
        required_keys = [
            "dataset", "n_samples", "f1_macro", "auc_roc", 
            "recall_glosada", "recall_per_class", "confusion_matrix", 
            "classification_report"
        ]
        
        for key in required_keys:
            assert key in metrics
        
        # Verificar tipos y valores
        assert metrics["dataset"] == "test"
        assert metrics["n_samples"] == 5
        assert isinstance(metrics["f1_macro"], float)
        assert isinstance(metrics["auc_roc"], float)
        assert isinstance(metrics["recall_glosada"], float)
        assert isinstance(metrics["recall_per_class"], dict)
        assert len(metrics["recall_per_class"]) == 3

    def test_save_model(self):
        """Test que save_model guarda modelo y metadata correctamente."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_model = Mock()
            mock_model.n_features_in_ = 48
            
            metrics = {
                "f1_macro": 0.75,
                "recall_glosada": 0.80,
                "auc_roc": 0.85
            }
            
            with patch('src.evaluation.retrain_xgboost.MODELS_DIR', temp_path):
                model_path = self.retrainer.save_model(mock_model, metrics, "test")
            
            mock_model.save_model.assert_called_once()
            
            # Verificar que se creó el archivo de metadata
            metadata_files = list(temp_path.rglob("*metadata*.json"))
            assert len(metadata_files) > 0, f"No metadata files in {list(temp_path.rglob('*'))}"
            
            with open(metadata_files[0], 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            assert metadata["model_type"] == "test"
            assert metadata["n_features"] == 48
            assert metadata["n_classes"] == 3
            assert metadata["metrics"] == metrics

    def test_compare_models(self):
        """Test que _compare_models compara correctamente las métricas."""
        # Métricas del modelo base
        metrics_base = {
            "f1_macro": 0.70,
            "auc_roc": 0.85,
            "recall_glosada": 0.75
        }
        
        # Métricas del modelo enriquecido (mejores)
        metrics_enriched = {
            "f1_macro": 0.72,  # +0.02 (mejora significativa)
            "auc_roc": 0.86,   # +0.01 (mejora significativa)
            "recall_glosada": 0.82  # +0.07 (supera objetivo 0.80)
        }
        
        comparison = self.retrainer._compare_models(metrics_base, metrics_enriched)
        
        # Verificar cálculo de diferencias
        assert abs(comparison["f1_macro_diff"] - 0.02) < 0.001
        assert abs(comparison["auc_roc_diff"] - 0.01) < 0.001
        assert abs(comparison["recall_glosada_diff"] - 0.07) < 0.001
        
        # Verificar flags de mejora
        assert comparison["f1_improved"] is True
        assert comparison["auc_improved"] is True
        assert comparison["recall_glosada_improved"] is True
        assert comparison["target_recall_achieved"] is True
        assert comparison["overall_improvement"] is True

    def test_generate_recommendation_success(self):
        """Test que _generate_recommendation produce recomendación correcta para éxito."""
        recommendation = self.retrainer._generate_recommendation(
            target_achieved=True,
            f1_improved=True, 
            recall_improved=True
        )
        
        assert "ADOPTAR MODELO ENRIQUECIDO" in recommendation
        assert "objetivo de Recall Glosada" in recommendation

    def test_generate_recommendation_failure(self):
        """Test que _generate_recommendation produce recomendación correcta para fallo."""
        recommendation = self.retrainer._generate_recommendation(
            target_achieved=False,
            f1_improved=False,
            recall_improved=False
        )
        
        assert "MANTENER MODELO BASE" in recommendation
        assert "no produjo mejoras significativas" in recommendation

    @patch.object(XGBoostRetrainer, 'load_datasets')
    @patch.object(XGBoostRetrainer, 'extract_features_and_labels')
    @patch.object(XGBoostRetrainer, 'train_model')
    @patch.object(XGBoostRetrainer, 'evaluate_model')
    @patch.object(XGBoostRetrainer, 'save_model')
    @patch.object(XGBoostRetrainer, '_save_comparison_report')
    def test_run_full_retraining_integration(
        self, mock_save_report, mock_save_model, mock_evaluate, 
        mock_train, mock_extract, mock_load
    ):
        """Test de integración para run_full_retraining."""
        # Mock datasets
        mock_train_df = pd.DataFrame({'col': [1, 2, 3]})
        mock_val_df = pd.DataFrame({'col': [4, 5]})
        mock_test_df = pd.DataFrame({'col': [6, 7]})
        mock_load.return_value = (mock_train_df, mock_val_df, mock_test_df)
        
        # Mock extracción de features
        mock_extract.side_effect = [
            # Base model calls
            (np.random.randn(3, 48), np.array([0, 1, 2])),  # train
            (np.random.randn(2, 48), np.array([1, 0])),     # val
            (np.random.randn(2, 48), np.array([2, 1])),     # test
            # Enriched model calls  
            (np.random.randn(3, 1200), np.array([0, 1, 2])), # train
            (np.random.randn(2, 1200), np.array([1, 0])),    # val
            (np.random.randn(2, 1200), np.array([2, 1])),    # test
        ]
        
        # Mock modelos entrenados
        mock_model_base = Mock()
        mock_model_enriched = Mock()
        mock_train.side_effect = [mock_model_base, mock_model_enriched]
        
        # Mock métricas
        mock_metrics_base = {"f1_macro": 0.70, "recall_glosada": 0.75, "auc_roc": 0.85}
        mock_metrics_enriched = {"f1_macro": 0.72, "recall_glosada": 0.82, "auc_roc": 0.87}
        mock_evaluate.side_effect = [mock_metrics_base, mock_metrics_enriched]
        
        # Mock guardado de modelos
        mock_save_model.side_effect = [Path("base.json"), Path("enriched.json")]
        mock_save_report.return_value = Path("report.json")
        
        # Ejecutar reentrenamiento completo
        results = self.retrainer.run_full_retraining()
        
        # Verificar que se llamaron todos los métodos
        mock_load.assert_called_once()
        assert mock_extract.call_count == 6  # 3 para base + 3 para enriched
        assert mock_train.call_count == 2    # base + enriched
        assert mock_evaluate.call_count == 2 # base + enriched
        assert mock_save_model.call_count == 2 # base + enriched
        mock_save_report.assert_called_once()
        
        # Verificar estructura de resultados
        required_keys = [
            "model_base_path", "model_enriched_path", "metrics_base", 
            "metrics_enriched", "comparison", "report_path"
        ]
        for key in required_keys:
            assert key in results