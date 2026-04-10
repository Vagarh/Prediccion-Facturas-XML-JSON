# -*- coding: utf-8 -*-
"""Tests unitarios para LLMEvaluator.

Valida la evaluación experimental de candidatos LLM:
- run_full_evaluation produce JSON con todas las claves requeridas
- resultado negativo se documenta correctamente
- métricas de negocio se calculan correctamente
- separabilidad semántica funciona
- latencia se mide apropiadamente
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import json
import tempfile
from pathlib import Path

from src.evaluation.llm_evaluator import (
    LLMEvaluator,
    BASELINE_RECALL_GLOSADA,
    CLASS_GLOSADA,
    CLASS_NAMES
)


class TestLLMEvaluator:
    """Suite de tests para LLMEvaluator."""

    def setup_method(self):
        """Configuración inicial para cada test."""
        # Mock del embedding service
        self.mock_embedding_service = Mock()
        self.evaluator = LLMEvaluator(
            embedding_service=self.mock_embedding_service,
            random_state=42
        )

    def test_llm_evaluator_initialization(self):
        """Test que LLMEvaluator se inicializa correctamente."""
        mock_service = Mock()
        evaluator = LLMEvaluator(embedding_service=mock_service, random_state=123)
        
        assert evaluator.embedding_service == mock_service
        assert evaluator.random_state == 123

    def test_evaluate_candidate_with_predictions_array(self):
        """Test que evaluate_candidate funciona con array de predicciones."""
        model_name = "test_model"
        y_pred = np.array([0, 1, 2, 1, 0, 1])  # Predicciones
        y_test = np.array([0, 1, 2, 0, 0, 1])  # Etiquetas verdaderas
        
        result = self.evaluator.evaluate_candidate(model_name, y_pred, y_test)
        
        # Verificar estructura del resultado
        required_keys = [
            "model_name", "recall_glosada", "f1_macro", "auc_roc",
            "recall_per_class", "confusion_matrix", "exceeds_baseline"
        ]
        for key in required_keys:
            assert key in result
        
        # Verificar tipos de datos
        assert result["model_name"] == model_name
        assert isinstance(result["recall_glosada"], float)
        assert isinstance(result["f1_macro"], float)
        assert isinstance(result["auc_roc"], float)
        assert isinstance(result["recall_per_class"], dict)
        assert isinstance(result["confusion_matrix"], list)
        assert isinstance(result["exceeds_baseline"], bool)

    def test_evaluate_candidate_with_sklearn_model(self):
        """Test que evaluate_candidate funciona con modelo sklearn."""
        # Mock modelo sklearn
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 2, 1, 0])
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.3, 0.6, 0.1],
            [0.9, 0.05, 0.05]
        ])
        
        X_test = np.random.randn(5, 10)  # Features
        y_test = np.array([0, 1, 2, 0, 0])
        
        result = self.evaluator.evaluate_candidate(
            "sklearn_model", X_test, y_test, model=mock_model
        )
        
        # Verificar que se llamaron los métodos del modelo
        mock_model.predict.assert_called_once_with(X_test)
        mock_model.predict_proba.assert_called_once_with(X_test)
        
        # Verificar resultado
        assert result["model_name"] == "sklearn_model"
        assert "recall_glosada" in result

    def test_evaluate_candidate_without_predict_proba(self):
        """Test que evaluate_candidate maneja modelos sin predict_proba."""
        # Mock modelo sin predict_proba
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 2])
        del mock_model.predict_proba  # Simular que no tiene predict_proba
        
        X_test = np.random.randn(3, 10)
        y_test = np.array([0, 1, 2])
        
        result = self.evaluator.evaluate_candidate(
            "no_proba_model", X_test, y_test, model=mock_model
        )
        
        # Debe funcionar sin errores
        assert result["model_name"] == "no_proba_model"
        assert "auc_roc" in result

    def test_evaluate_candidate_calculates_recall_glosada_correctly(self):
        """Test que evaluate_candidate calcula Recall Glosada correctamente."""
        # Datos de prueba donde sabemos el resultado esperado
        y_pred = np.array([1, 1, 0, 1, 2])  # 3 predicciones de clase Glosada (1)
        y_test = np.array([1, 1, 1, 0, 2])  # 3 casos reales de clase Glosada (1)
        
        # Recall Glosada = TP / (TP + FN) = 2 / 3 = 0.6667
        result = self.evaluator.evaluate_candidate("test", y_pred, y_test)
        
        expected_recall_glosada = 2.0 / 3.0  # 2 TP, 1 FN
        assert abs(result["recall_glosada"] - expected_recall_glosada) < 0.001

    def test_evaluate_candidate_exceeds_baseline_logic(self):
        """Test que evaluate_candidate determina correctamente si supera baseline."""
        # Caso que supera baseline
        y_pred_good = np.array([1] * 10)  # Todas predicciones Glosada
        y_test_good = np.array([1] * 10)  # Todas etiquetas Glosada (Recall = 1.0)
        
        result_good = self.evaluator.evaluate_candidate("good_model", y_pred_good, y_test_good)
        assert result_good["recall_glosada"] > BASELINE_RECALL_GLOSADA
        assert result_good["exceeds_baseline"] is True
        
        # Caso que no supera baseline
        y_pred_bad = np.array([0] * 10)  # Todas predicciones Auditada
        y_test_bad = np.array([1] * 10)  # Todas etiquetas Glosada (Recall = 0.0)
        
        result_bad = self.evaluator.evaluate_candidate("bad_model", y_pred_bad, y_test_bad)
        assert result_bad["recall_glosada"] <= BASELINE_RECALL_GLOSADA
        assert result_bad["exceeds_baseline"] is False

    def test_evaluate_candidate_handles_dimension_mismatch(self):
        """Test que evaluate_candidate maneja desajuste de dimensiones."""
        y_pred = np.array([0, 1, 2])
        y_test = np.array([0, 1])  # Diferente longitud
        
        with pytest.raises(ValueError) as exc_info:
            self.evaluator.evaluate_candidate("test", y_pred, y_test)
        
        assert "Dimensiones incompatibles" in str(exc_info.value)

    def test_measure_embedding_separability_with_good_separation(self):
        """Test que measure_embedding_separability funciona con buena separación."""
        # Crear embeddings bien separados por clase
        n_samples_per_class = 50
        embedding_dim = 10
        
        # Clase 0: centrada en (1, 0, 0, ...)
        class_0 = np.random.randn(n_samples_per_class, embedding_dim) * 0.1
        class_0[:, 0] += 1.0
        
        # Clase 1: centrada en (0, 1, 0, ...)
        class_1 = np.random.randn(n_samples_per_class, embedding_dim) * 0.1
        class_1[:, 1] += 1.0
        
        # Clase 2: centrada en (0, 0, 1, ...)
        class_2 = np.random.randn(n_samples_per_class, embedding_dim) * 0.1
        class_2[:, 2] += 1.0
        
        embeddings = np.vstack([class_0, class_1, class_2])
        labels = np.array([0] * n_samples_per_class + [1] * n_samples_per_class + [2] * n_samples_per_class)
        
        score = self.evaluator.measure_embedding_separability(embeddings, labels)
        
        # Con buena separación, el score debe ser positivo
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0
        assert score > 0  # Esperamos separación positiva

    def test_measure_embedding_separability_with_insufficient_samples(self):
        """Test que measure_embedding_separability maneja muestras insuficientes."""
        # Solo una muestra
        embeddings = np.random.randn(1, 10)
        labels = np.array([0])
        
        score = self.evaluator.measure_embedding_separability(embeddings, labels)
        assert score == 0.0

    def test_measure_embedding_separability_with_dimension_mismatch(self):
        """Test que measure_embedding_separability maneja desajuste de dimensiones."""
        embeddings = np.random.randn(10, 5)
        labels = np.array([0, 1, 2])  # Diferente longitud
        
        with pytest.raises(ValueError) as exc_info:
            self.evaluator.measure_embedding_separability(embeddings, labels)
        
        assert "embeddings tiene" in str(exc_info.value)

    def test_measure_latency_returns_correct_format(self):
        """Test que measure_latency retorna formato correcto."""
        # Mock del embedding service para simular latencia
        self.mock_embedding_service.encode.return_value = np.random.randn(384)
        
        result = self.evaluator.measure_latency("test_model", sample_size=10)
        
        # Verificar estructura del resultado
        assert isinstance(result, dict)
        assert "p50_ms" in result
        assert "p95_ms" in result
        assert isinstance(result["p50_ms"], float)
        assert isinstance(result["p95_ms"], float)
        assert result["p50_ms"] >= 0.0
        assert result["p95_ms"] >= 0.0
        assert result["p95_ms"] >= result["p50_ms"]  # p95 >= p50

    def test_measure_latency_calls_encode_correct_number_of_times(self):
        """Test que measure_latency llama encode el número correcto de veces."""
        sample_size = 5
        self.mock_embedding_service.encode.return_value = np.random.randn(384)
        
        self.evaluator.measure_latency("test_model", sample_size=sample_size)
        
        # Verificar que se llamó encode sample_size veces
        assert self.mock_embedding_service.encode.call_count == sample_size

    @patch('src.evaluation.llm_evaluator._save_report')
    def test_run_full_evaluation_produces_complete_report(self, mock_save_report):
        """Test que run_full_evaluation produce JSON con todas las claves requeridas."""
        candidates = ["model1", "model2"]
        
        # Mock datos de test
        X_test = np.array([0, 1, 2, 1, 0])  # Predicciones
        y_test = np.array([0, 1, 2, 0, 0])  # Etiquetas verdaderas
        
        # Mock embeddings por candidato
        embeddings_per_candidate = {
            "model1": np.random.randn(5, 10),
            "model2": np.random.randn(5, 10)
        }
        
        # Mock del embedding service para latencia
        self.mock_embedding_service.encode.return_value = np.random.randn(384)
        
        result = self.evaluator.run_full_evaluation(
            candidates=candidates,
            X_test=X_test,
            y_test=y_test,
            embeddings_per_candidate=embeddings_per_candidate
        )
        
        # Verificar estructura del reporte
        required_top_keys = ["timestamp", "baseline_recall_glosada", "candidates", "recommendation"]
        for key in required_top_keys:
            assert key in result
        
        # Verificar candidatos
        assert len(result["candidates"]) == len(candidates)
        for candidate in candidates:
            assert candidate in result["candidates"]
            candidate_data = result["candidates"][candidate]
            
            # Verificar estructura de cada candidato
            assert "metrics" in candidate_data
            assert "separability" in candidate_data
            assert "latency" in candidate_data

        # Verificar recomendación
        recommendation = result["recommendation"]
        assert "selected_model" in recommendation
        assert "justification" in recommendation
        assert "negative_result" in recommendation
        
        # Verificar que se guardó el reporte
        mock_save_report.assert_called_once_with(result)

    @patch('src.evaluation.llm_evaluator._save_report')
    def test_run_full_evaluation_negative_result_documented(self, mock_save_report):
        """Test que resultado negativo se documenta correctamente."""
        candidates = ["weak_model"]
        
        # Datos que producen Recall Glosada bajo (no supera baseline)
        X_test = np.array([0, 0, 0, 0, 0])  # Todas predicciones Auditada
        y_test = np.array([1, 1, 1, 1, 1])  # Todas etiquetas Glosada (Recall = 0.0)
        
        self.mock_embedding_service.encode.return_value = np.random.randn(384)
        
        result = self.evaluator.run_full_evaluation(
            candidates=candidates,
            X_test=X_test,
            y_test=y_test
        )
        
        # Verificar resultado negativo
        recommendation = result["recommendation"]
        assert recommendation["negative_result"] is True
        assert recommendation["selected_model"] is None
        assert "ningún candidato" in recommendation["justification"].lower()
        assert "baseline" in recommendation["justification"].lower()

    @patch('src.evaluation.llm_evaluator._save_report')
    def test_run_full_evaluation_positive_result_documented(self, mock_save_report):
        """Test que resultado positivo se documenta correctamente."""
        candidates = ["strong_model"]
        
        # Datos que producen Recall Glosada alto (supera baseline)
        X_test = np.array([1, 1, 1, 1, 1])  # Todas predicciones Glosada
        y_test = np.array([1, 1, 1, 1, 1])  # Todas etiquetas Glosada (Recall = 1.0)
        
        self.mock_embedding_service.encode.return_value = np.random.randn(384)
        
        result = self.evaluator.run_full_evaluation(
            candidates=candidates,
            X_test=X_test,
            y_test=y_test
        )
        
        # Verificar resultado positivo
        recommendation = result["recommendation"]
        assert recommendation["negative_result"] is False
        assert recommendation["selected_model"] == "strong_model"
        assert "supera" in recommendation["justification"].lower()

    def test_run_full_evaluation_handles_missing_data_gracefully(self):
        """Test que run_full_evaluation maneja datos faltantes graciosamente."""
        candidates = ["model1"]
        
        # Sin datos de test (solo latencia)
        result = self.evaluator.run_full_evaluation(candidates=candidates)
        
        # Debe funcionar sin errores
        assert "candidates" in result
        assert "model1" in result["candidates"]
        
        candidate_data = result["candidates"]["model1"]
        assert candidate_data["metrics"] is None  # Sin datos de test
        assert candidate_data["separability"] is None  # Sin embeddings
        assert candidate_data["latency"] is not None  # Latencia siempre se mide

    def test_run_full_evaluation_handles_exceptions_gracefully(self):
        """Test que run_full_evaluation maneja excepciones graciosamente."""
        candidates = ["error_model"]
        
        # Mock embedding service que lanza excepción
        self.mock_embedding_service.encode.side_effect = Exception("Test error")
        
        # No debe lanzar excepción, debe manejar el error
        result = self.evaluator.run_full_evaluation(candidates=candidates)
        
        assert "candidates" in result
        assert "error_model" in result["candidates"]

    def test_constants_are_correct(self):
        """Test que las constantes están definidas correctamente."""
        assert isinstance(BASELINE_RECALL_GLOSADA, float)
        assert BASELINE_RECALL_GLOSADA == 0.7261
        
        assert isinstance(CLASS_GLOSADA, int)
        assert CLASS_GLOSADA == 1
        
        assert isinstance(CLASS_NAMES, list)
        assert len(CLASS_NAMES) == 3
        assert CLASS_NAMES == ["Auditada", "Glosada", "Devuelta"]

    @patch('src.evaluation.llm_evaluator._generate_sample_texts')
    def test_measure_latency_uses_sample_texts(self, mock_generate_texts):
        """Test que measure_latency usa textos de muestra apropiados."""
        sample_texts = ["diabetes", "hipertensión", "neumonía"]
        mock_generate_texts.return_value = sample_texts
        
        self.mock_embedding_service.encode.return_value = np.random.randn(384)
        
        self.evaluator.measure_latency("test_model", sample_size=3)
        
        # Verificar que se generaron textos de muestra
        mock_generate_texts.assert_called_once_with(3, random_state=42)
        
        # Verificar que se llamó encode para cada texto
        assert self.mock_embedding_service.encode.call_count == 3

    def test_evaluate_candidate_with_invalid_predictions_shape(self):
        """Test que evaluate_candidate maneja forma inválida de predicciones."""
        # Predicciones 2D cuando se espera 1D (sin modelo)
        X_test = np.random.randn(5, 3)  # 2D
        y_test = np.array([0, 1, 2, 0, 1])
        
        with pytest.raises(ValueError) as exc_info:
            self.evaluator.evaluate_candidate("test", X_test, y_test, model=None)
        
        assert "1-D" in str(exc_info.value)

    def test_run_full_evaluation_with_models_dict(self):
        """Test que run_full_evaluation funciona con diccionario de modelos."""
        candidates = ["model1"]
        
        # Mock modelo
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1, 1, 1])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]] * 3)
        
        models = {"model1": mock_model}
        X_test = np.random.randn(3, 10)
        y_test = np.array([1, 1, 1])
        
        self.mock_embedding_service.encode.return_value = np.random.randn(384)
        
        with patch('src.evaluation.llm_evaluator._save_report'):
            result = self.evaluator.run_full_evaluation(
                candidates=candidates,
                X_test=X_test,
                y_test=y_test,
                models=models
            )
        
        # Verificar que se usó el modelo
        mock_model.predict.assert_called_once_with(X_test)
        
        # Verificar resultado
        assert result["candidates"]["model1"]["metrics"] is not None