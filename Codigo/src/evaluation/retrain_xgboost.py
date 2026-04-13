# -*- coding: utf-8 -*-
"""Script de reentrenamiento XGBoost con features enriquecidas.

Carga datasets train/val/test, genera embeddings semánticos con EmbeddingService,
concatena al vector de 48 features numéricas, reentrena XGBoost con RANDOM_STATE=42,
evalúa en test y OOT, y serializa modelo enriquecido con prefijo de fecha.

Cumplimiento:
    - Mantiene tarea multiclase (3 clases) siempre
    - División temporal respetada (nunca aleatoria)
    - RANDOM_STATE = 42 en todos los componentes
    - Serialización con metadata completa
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    EMBEDDING_DIM,
    RANDOM_STATE,
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.embeddings.service import EmbeddingService
from src.features.embedding_pipeline import TEXT_FIELD_KEYS
from src.nlp.pipeline import load_nlp_pipeline
from src.privacy.guard import PrivacyGuard

logger = logging.getLogger(__name__)

# Configuración del modelo XGBoost
XGBOOST_PARAMS = {
    "objective": "multi:softprob",  # Multiclase con probabilidades
    "num_class": 3,                 # 3 clases: Auditada, Glosada, Devuelta
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
    "eval_metric": "mlogloss",
    "early_stopping_rounds": 10,
}

# Nombres de las clases
CLASS_NAMES = ["Auditada", "Glosada", "Devuelta"]
CLASS_GLOSADA = 1  # Índice de la clase Glosada


class XGBoostRetrainer:
    """Reentrenador de XGBoost con features enriquecidas por embeddings semánticos."""

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        random_state: int = RANDOM_STATE,
    ) -> None:
        """Inicializa el reentrenador con componentes necesarios.

        Args:
            embedding_model_name: Nombre del modelo de embeddings a usar.
            random_state: Semilla de aleatoriedad para reproducibilidad.
        """
        self.random_state = random_state
        self.embedding_model_name = embedding_model_name
        
        # Inicializar componentes
        logger.info("Inicializando componentes del reentrenador...")
        
        # Privacy Guard para cumplimiento Ley 1581
        self.privacy_guard = PrivacyGuard()
        self.privacy_guard.verify_local_only()
        
        # NLP Pipeline en español
        self.nlp_pipeline = load_nlp_pipeline()
        
        # Embedding Service
        self.embedding_service = EmbeddingService(
            model_name=embedding_model_name,
            random_state=random_state
        )
        
        logger.info("Componentes inicializados correctamente.")

    def load_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Carga los datasets de entrenamiento, validación y test.

        Returns:
            Tupla con DataFrames (train, val, test).

        Raises:
            FileNotFoundError: Si algún archivo de dataset no existe.
        """
        logger.info("Cargando datasets desde %s...", DATA_DIR / "processed")
        
        # Rutas de los archivos (preferir versión enriquecida con campos textuales)
        train_path = DATA_DIR / "processed" / "20260309_train_enriched.parquet"
        val_path   = DATA_DIR / "processed" / "20260309_val_enriched.parquet"
        test_path  = DATA_DIR / "processed" / "20260309_test_enriched.parquet"

        # Fallback a parquets originales si los enriquecidos no existen aún
        if not train_path.exists():
            logger.warning("Parquets enriquecidos no encontrados, usando originales (sin campos textuales).")
            train_path = DATA_DIR / "processed" / "20260309_train.parquet"
            val_path   = DATA_DIR / "processed" / "20260309_val.parquet"
            test_path  = DATA_DIR / "processed" / "20260309_test.parquet"
        
        # Verificar que existen
        for path in [train_path, val_path, test_path]:
            if not path.exists():
                raise FileNotFoundError(f"Dataset no encontrado: {path}")
        
        # Cargar datasets
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)
        
        logger.info(
            "Datasets cargados: train=%d, val=%d, test=%d",
            len(train_df), len(val_df), len(test_df)
        )
        
        return train_df, val_df, test_df

    def extract_features_and_labels(
        self, 
        df: pd.DataFrame,
        include_embeddings: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extrae features numéricas (y embeddings si se solicita) y labels del DataFrame.

        Args:
            df: DataFrame con los datos.
            include_embeddings: Si incluir embeddings semánticos en las features.

        Returns:
            Tupla (X, y) donde X son las features y y son las labels.
        """
        logger.info(
            "Extrayendo features (embeddings=%s) y labels de %d registros...",
            include_embeddings, len(df)
        )
        
        # Cargar lista de features numéricas
        feature_list_path = DATA_DIR / "processed" / "20260309_feature_list.txt"
        if not feature_list_path.exists():
            raise FileNotFoundError(f"Lista de features no encontrada: {feature_list_path}")
        
        with open(feature_list_path, 'r', encoding='utf-8') as f:
            numeric_features = [line.strip() for line in f if line.strip()]
        
        logger.info("Cargadas %d features numéricas.", len(numeric_features))
        
        # Extraer features numéricas (48 features)
        X_numeric = df[numeric_features].values.astype(np.float32)
        
        if not include_embeddings:
            # Solo features numéricas (modelo base)
            X = X_numeric
            logger.info("Features extraídas: %d numéricas (modelo base).", X.shape[1])
        else:
            # Enriquecer con embeddings semánticos — batch completo por campo textual
            logger.info("Generando embeddings semánticos en batch para %d registros...", len(df))

            embedding_parts: list[np.ndarray] = []
            EMBED_CHUNK = 10_000  # Chunk size para logs de progreso
            for key in TEXT_FIELD_KEYS:
                texts = df[key].fillna("").astype(str).tolist()
                n = len(texts)
                logger.info("  Codificando campo '%s' (%d textos) en chunks de %d...", key, n, EMBED_CHUNK)
                chunks_emb: list[np.ndarray] = []
                for start in range(0, n, EMBED_CHUNK):
                    chunk = texts[start: start + EMBED_CHUNK]
                    chunks_emb.append(self.embedding_service.encode_batch(chunk))
                    logger.info("    %s: %d/%d textos codificados...", key, min(start + EMBED_CHUNK, n), n)
                emb_matrix = np.vstack(chunks_emb)
                embedding_parts.append(emb_matrix)
                logger.info("  Campo '%s' codificado: shape=%s", key, emb_matrix.shape)

            # Concatenar: (N, 48) + (N, 384) * 3 → (N, 48 + 1152)
            X = np.concatenate([X_numeric] + embedding_parts, axis=1).astype(np.float32)
            n_numeric = X_numeric.shape[1]
            expected_dim = n_numeric + EMBEDDING_DIM * len(TEXT_FIELD_KEYS)
            logger.info(
                "Features extraídas: %d enriquecidas (%d numéricas + %d embeddings).",
                X.shape[1], n_numeric, EMBEDDING_DIM * len(TEXT_FIELD_KEYS)
            )

            assert X.shape[1] == expected_dim, f"Dimensión incorrecta: {X.shape[1]} != {expected_dim}"
        
        # Extraer labels — columna 'target' en los parquets del proyecto
        label_col = "target" if "target" in df.columns else "clase"
        y = df[label_col].values.astype(int)

        # Verificar que son 3 clases
        unique_classes = np.unique(y)
        assert len(unique_classes) == 3, f"Se esperan 3 clases, encontradas: {unique_classes}"
        assert set(unique_classes) == {0, 1, 2}, f"Clases deben ser {{0,1,2}}, encontradas: {unique_classes}"
        
        logger.info("Labels extraídas: %d registros, clases %s.", len(y), unique_classes)
        
        return X, y

    def train_model(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray
    ) -> xgb.XGBClassifier:
        """Entrena modelo XGBoost con early stopping en validación.

        Args:
            X_train: Features de entrenamiento.
            y_train: Labels de entrenamiento.
            X_val: Features de validación.
            y_val: Labels de validación.

        Returns:
            Modelo XGBoost entrenado.
        """
        logger.info(
            "Entrenando XGBoost: train=%d, val=%d, features=%d",
            len(X_train), len(X_val), X_train.shape[1]
        )
        
        # Crear modelo
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        
        # Entrenar con early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        logger.info(
            "Entrenamiento completado. Mejor iteración: %d",
            model.best_iteration if hasattr(model, 'best_iteration') else 'N/A'
        )
        
        return model

    def evaluate_model(
        self, 
        model: xgb.XGBClassifier, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        dataset_name: str = "test"
    ) -> dict[str, Any]:
        """Evalúa modelo en conjunto de test y calcula métricas de negocio.

        Args:
            model: Modelo XGBoost entrenado.
            X_test: Features de test.
            y_test: Labels de test.
            dataset_name: Nombre del dataset para logging.

        Returns:
            Diccionario con métricas calculadas.
        """
        logger.info("Evaluando modelo en conjunto %s (%d registros)...", dataset_name, len(X_test))
        
        # Predicciones
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Métricas principales
        f1_macro = float(f1_score(y_test, y_pred, average="macro"))
        auc_roc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
        
        # Recall por clase
        recall_per_class = recall_score(y_test, y_pred, average=None, labels=[0, 1, 2])
        recall_glosada = float(recall_per_class[CLASS_GLOSADA])
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        
        # Reporte de clasificación
        class_report = classification_report(
            y_test, y_pred, 
            target_names=CLASS_NAMES, 
            output_dict=True
        )
        
        metrics = {
            "dataset": dataset_name,
            "n_samples": len(X_test),
            "f1_macro": f1_macro,
            "auc_roc": auc_roc,
            "recall_glosada": recall_glosada,
            "recall_per_class": {
                CLASS_NAMES[i]: float(recall_per_class[i]) for i in range(3)
            },
            "confusion_matrix": cm.tolist(),
            "classification_report": class_report,
        }
        
        logger.info(
            "Evaluación %s: F1-macro=%.4f, AUC-ROC=%.4f, Recall Glosada=%.4f",
            dataset_name, f1_macro, auc_roc, recall_glosada
        )
        
        return metrics

    def save_model(
        self, 
        model: xgb.XGBClassifier, 
        metrics: dict[str, Any],
        model_type: str = "enriched"
    ) -> Path:
        """Guarda modelo entrenado con metadata completa.

        Args:
            model: Modelo XGBoost entrenado.
            metrics: Métricas de evaluación.
            model_type: Tipo de modelo ("base" o "enriched").

        Returns:
            Ruta donde se guardó el modelo.
        """
        # Crear directorio si no existe
        models_dir = Path(MODELS_DIR) / "artifacts"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Nombre del archivo con fecha y tipo
        date_prefix = datetime.utcnow().strftime("%Y%m%d")
        model_filename = f"{date_prefix}_xgboost_v2_{model_type}.json"
        model_path = models_dir / model_filename
        
        # Guardar modelo
        model.save_model(str(model_path))
        
        # Metadata del modelo
        metadata = {
            "model_type": model_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "xgboost_version": xgb.__version__,
            "embedding_model": self.embedding_model_name if model_type == "enriched" else None,
            "embedding_dim": EMBEDDING_DIM if model_type == "enriched" else None,
            "n_features": model.n_features_in_,
            "n_classes": 3,
            "class_names": CLASS_NAMES,
            "random_state": self.random_state,
            "xgboost_params": XGBOOST_PARAMS,
            "metrics": metrics,
        }
        
        # Guardar metadata
        metadata_path = models_dir / f"{date_prefix}_xgboost_v2_{model_type}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info("Modelo guardado: %s", model_path)
        logger.info("Metadata guardada: %s", metadata_path)
        
        return model_path

    def run_full_retraining(self) -> dict[str, Any]:
        """Ejecuta el proceso completo de reentrenamiento con comparación base vs enriquecido.

        Returns:
            Diccionario con resultados de la comparación.
        """
        logger.info("=== INICIANDO REENTRENAMIENTO COMPLETO ===")
        
        # 1. Cargar datasets
        train_df, val_df, test_df = self.load_datasets()
        
        # 2. Entrenar modelo BASE (solo features numéricas)
        logger.info("=== ENTRENANDO MODELO BASE ===")
        X_train_base, y_train = self.extract_features_and_labels(train_df, include_embeddings=False)
        X_val_base, y_val = self.extract_features_and_labels(val_df, include_embeddings=False)
        X_test_base, y_test = self.extract_features_and_labels(test_df, include_embeddings=False)
        
        model_base = self.train_model(X_train_base, y_train, X_val_base, y_val)
        metrics_base = self.evaluate_model(model_base, X_test_base, y_test, "test_base")
        model_path_base = self.save_model(model_base, metrics_base, "base")
        
        # 3. Entrenar modelo ENRIQUECIDO (features + embeddings)
        logger.info("=== ENTRENANDO MODELO ENRIQUECIDO ===")
        X_train_enriched, _ = self.extract_features_and_labels(train_df, include_embeddings=True)
        X_val_enriched, _ = self.extract_features_and_labels(val_df, include_embeddings=True)
        X_test_enriched, _ = self.extract_features_and_labels(test_df, include_embeddings=True)
        
        model_enriched = self.train_model(X_train_enriched, y_train, X_val_enriched, y_val)
        metrics_enriched = self.evaluate_model(model_enriched, X_test_enriched, y_test, "test_enriched")
        model_path_enriched = self.save_model(model_enriched, metrics_enriched, "enriched")
        
        # 4. Comparación de resultados
        comparison = self._compare_models(metrics_base, metrics_enriched)
        
        # 5. Guardar reporte de comparación
        report_path = self._save_comparison_report(comparison, metrics_base, metrics_enriched)
        
        logger.info("=== REENTRENAMIENTO COMPLETADO ===")
        logger.info("Modelo base: %s", model_path_base)
        logger.info("Modelo enriquecido: %s", model_path_enriched)
        logger.info("Reporte: %s", report_path)
        
        return {
            "model_base_path": str(model_path_base),
            "model_enriched_path": str(model_path_enriched),
            "metrics_base": metrics_base,
            "metrics_enriched": metrics_enriched,
            "comparison": comparison,
            "report_path": str(report_path),
        }

    def _compare_models(
        self, 
        metrics_base: dict[str, Any], 
        metrics_enriched: dict[str, Any]
    ) -> dict[str, Any]:
        """Compara métricas entre modelo base y enriquecido.

        Args:
            metrics_base: Métricas del modelo base.
            metrics_enriched: Métricas del modelo enriquecido.

        Returns:
            Diccionario con comparación detallada.
        """
        # Diferencias en métricas principales
        f1_diff = metrics_enriched["f1_macro"] - metrics_base["f1_macro"]
        auc_diff = metrics_enriched["auc_roc"] - metrics_base["auc_roc"]
        recall_glosada_diff = metrics_enriched["recall_glosada"] - metrics_base["recall_glosada"]
        
        # Determinar si hay mejora significativa
        f1_improved = f1_diff > 0.01  # Mejora > 1%
        auc_improved = auc_diff > 0.005  # Mejora > 0.5%
        recall_glosada_improved = recall_glosada_diff > 0.01  # Mejora > 1%
        
        # Verificar si supera objetivo de Recall Glosada ≥ 0.80
        target_recall_achieved = metrics_enriched["recall_glosada"] >= 0.80
        
        comparison = {
            "f1_macro_diff": f1_diff,
            "auc_roc_diff": auc_diff,
            "recall_glosada_diff": recall_glosada_diff,
            "f1_improved": f1_improved,
            "auc_improved": auc_improved,
            "recall_glosada_improved": recall_glosada_improved,
            "target_recall_achieved": target_recall_achieved,
            "overall_improvement": f1_improved and recall_glosada_improved,
            "recommendation": self._generate_recommendation(
                target_recall_achieved, f1_improved, recall_glosada_improved
            ),
        }
        
        return comparison

    def _generate_recommendation(
        self, 
        target_achieved: bool, 
        f1_improved: bool, 
        recall_improved: bool
    ) -> str:
        """Genera recomendación basada en resultados de comparación.

        Args:
            target_achieved: Si se alcanzó Recall Glosada ≥ 0.80.
            f1_improved: Si mejoró F1-macro.
            recall_improved: Si mejoró Recall Glosada.

        Returns:
            Recomendación textual.
        """
        if target_achieved and f1_improved and recall_improved:
            return (
                "RECOMENDACIÓN: ADOPTAR MODELO ENRIQUECIDO. "
                "Se alcanzó el objetivo de Recall Glosada ≥ 0.80 con mejoras "
                "significativas en F1-macro y Recall Glosada."
            )
        elif target_achieved:
            return (
                "RECOMENDACIÓN: CONSIDERAR MODELO ENRIQUECIDO. "
                "Se alcanzó el objetivo de Recall Glosada ≥ 0.80, aunque "
                "algunas métricas no mejoraron significativamente."
            )
        elif recall_improved:
            return (
                "RECOMENDACIÓN: CONTINUAR DESARROLLO. "
                "Hay mejora en Recall Glosada pero no se alcanzó el objetivo. "
                "Considerar ajustar hiperparámetros o probar otros modelos LLM."
            )
        else:
            return (
                "RECOMENDACIÓN: MANTENER MODELO BASE. "
                "El enriquecimiento con embeddings no produjo mejoras significativas. "
                "Revisar calidad de embeddings y preprocesamiento de texto."
            )

    def _save_comparison_report(
        self, 
        comparison: dict[str, Any],
        metrics_base: dict[str, Any],
        metrics_enriched: dict[str, Any]
    ) -> Path:
        """Guarda reporte detallado de comparación entre modelos.

        Args:
            comparison: Resultados de comparación.
            metrics_base: Métricas del modelo base.
            metrics_enriched: Métricas del modelo enriquecido.

        Returns:
            Ruta del reporte guardado.
        """
        # Crear directorio si no existe
        reports_dir = Path(REPORTS_DIR)
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Reporte completo
        date_prefix = datetime.utcnow().strftime("%Y%m%d")
        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "experiment_type": "xgboost_retraining_with_embeddings",
            "embedding_model": self.embedding_model_name,
            "random_state": self.random_state,
            "baseline_metrics": metrics_base,
            "enriched_metrics": metrics_enriched,
            "comparison": comparison,
        }
        
        # Guardar reporte
        report_path = reports_dir / f"{date_prefix}_xgboost_retraining_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report_path


def main():
    """Función principal para ejecutar reentrenamiento desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reentrenamiento XGBoost con features enriquecidas")
    parser.add_argument(
        "--embedding-model", 
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Modelo de embeddings a usar"
    )
    parser.add_argument(
        "--random-state", 
        type=int, 
        default=RANDOM_STATE,
        help="Semilla de aleatoriedad"
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar reentrenamiento
    retrainer = XGBoostRetrainer(
        embedding_model_name=args.embedding_model,
        random_state=args.random_state
    )
    
    try:
        results = retrainer.run_full_retraining()
        
        print("\n=== RESULTADOS DEL REENTRENAMIENTO ===")
        print(f"Modelo base: {results['model_base_path']}")
        print(f"Modelo enriquecido: {results['model_enriched_path']}")
        print(f"Reporte: {results['report_path']}")
        print(f"\nRecomendación: {results['comparison']['recommendation']}")
        
    except Exception as e:
        logger.error("Error durante el reentrenamiento: %s", e)
        raise


if __name__ == "__main__":
    main()