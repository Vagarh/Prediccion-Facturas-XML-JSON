# -*- coding: utf-8 -*-
"""Scheduler de reentrenamiento automático mensual del modelo XGBoost.

Evalúa PSI mensual, detecta concept drift y activa reentrenamiento
con datos recientes. Incluye backup y validación del nuevo modelo.
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    MODEL_METRICS,
    PSI_DRIFT_ALERT_THRESHOLD,
)
from src.monitoring.drift_detector import compute_psi, check_drift
from src.monitoring.model_monitor import ModelMonitor
from src.storage.model_registry import (
    save_model_version,
    set_active_version,
    get_active_version,
    list_model_versions,
)

logger = logging.getLogger(__name__)

PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
BACKUP_DIR = MODELS_DIR / "backups"


class RetrainingScheduler:
    """Orquesta el ciclo de reentrenamiento mensual del modelo XGBoost.

    Detecta concept drift vía PSI, reentrena con datos recientes,
    valida el nuevo modelo y hace rollback automático si las métricas
    son inferiores al modelo actual.

    Attributes:
        monitor: Instancia de ModelMonitor para tracking de métricas.
        min_recall_glosada: Umbral mínimo de Recall Glosada para aceptar nuevo modelo.
        months_back: Meses de datos históricos a usar en reentrenamiento.
    """

    def __init__(
        self,
        monitor: ModelMonitor | None = None,
        min_recall_glosada: float = 0.7261,
        months_back: int = 3,
    ) -> None:
        """Inicializa el scheduler.

        Args:
            monitor: Instancia de ModelMonitor. Si None, crea una nueva.
            min_recall_glosada: Recall Glosada mínimo para aceptar nuevo modelo.
            months_back: Meses de datos recientes para reentrenamiento.
        """
        self.monitor = monitor or ModelMonitor()
        self.min_recall_glosada = min_recall_glosada
        self.months_back = months_back
        logger.info(
            "RetrainingScheduler inicializado (min_recall_glosada=%.4f, months_back=%d)",
            min_recall_glosada,
            months_back,
        )

    # ------------------------------------------------------------------
    # Métodos públicos principales
    # ------------------------------------------------------------------

    def check_drift_and_retrain(self) -> dict[str, Any]:
        """Evalúa PSI mensual y activa reentrenamiento si hay drift.

        Returns:
            Diccionario con ``drift_status``, ``retrain_triggered`` y
            ``retrain_result`` (si se ejecutó).
        """
        logger.info("Iniciando evaluación de drift mensual...")

        # Cargar datos de referencia y actuales
        ref_data, cur_data = self._load_reference_and_current()
        if ref_data is None or cur_data is None:
            return {"drift_status": "sin_datos", "retrain_triggered": False}

        # Calcular PSI sobre features numéricas
        psi_values = []
        for col in ref_data.columns:
            if ref_data[col].dtype in [np.float64, np.float32, float]:
                try:
                    psi = compute_psi(ref_data[col].values, cur_data[col].values)
                    psi_values.append(psi)
                except Exception:
                    pass

        mean_psi = float(np.mean(psi_values)) if psi_values else 0.0
        drift_status = check_drift(mean_psi)

        logger.info("PSI medio=%.4f → estado=%s", mean_psi, drift_status)

        result: dict[str, Any] = {
            "drift_status": drift_status,
            "mean_psi": mean_psi,
            "retrain_triggered": False,
        }

        # Activar reentrenamiento si hay alerta de drift
        if drift_status in ("warning", "alert"):
            logger.info("Drift detectado — activando reentrenamiento automático")
            retrain_result = self.retrain_with_recent_data(self.months_back)
            result["retrain_triggered"] = True
            result["retrain_result"] = retrain_result

        return result

    def retrain_with_recent_data(self, months_back: int = 3) -> dict[str, Any]:
        """Reentrena el modelo con datos de los últimos N meses.

        Flujo: backup → carga datos → reentrenamiento → validación → activación.

        Args:
            months_back: Número de meses de datos recientes a usar.

        Returns:
            Diccionario con ``success``, ``new_version``, ``metrics`` y
            ``rollback_performed`` (bool).
        """
        logger.info("Iniciando reentrenamiento con datos de los últimos %d meses", months_back)

        # 1. Backup del modelo actual
        backup_path = self.backup_current_model()

        try:
            # 2. Cargar datos recientes
            train_df, val_df, test_df = self._load_recent_splits(months_back)

            if train_df is None:
                return {"success": False, "reason": "No hay datos suficientes para reentrenar"}

            # 3. Reentrenar
            new_model, new_metrics = self._train_model(train_df, val_df, test_df)

            # 4. Validar nuevo modelo
            validation = self.validate_new_model(new_metrics)

            if not validation["accepted"]:
                logger.warning(
                    "Nuevo modelo rechazado (Recall_Glosada=%.4f < %.4f) — rollback",
                    new_metrics.get("recall_glosada", 0),
                    self.min_recall_glosada,
                )
                return {
                    "success": False,
                    "reason": "Métricas inferiores al umbral mínimo",
                    "new_metrics": new_metrics,
                    "rollback_performed": False,
                }

            # 5. Serializar y registrar nueva versión
            date_prefix = datetime.now().strftime("%Y%m%d")
            model_id = f"{date_prefix}_xgboost_retrained"
            artifact_path = ARTIFACTS_DIR / f"{model_id}.json"
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

            new_model.save_model(str(artifact_path))

            version = f"v_retrain_{date_prefix}"
            save_model_version(
                model_id=model_id,
                version=version,
                metrics=new_metrics,
                artifact_path=artifact_path,
                notes=f"Reentrenamiento automático con {months_back} meses de datos",
            )
            set_active_version(version)

            logger.info(
                "Reentrenamiento exitoso: %s (F1=%.4f, Recall_Glosada=%.4f)",
                version,
                new_metrics.get("f1_macro", 0),
                new_metrics.get("recall_glosada", 0),
            )

            return {
                "success": True,
                "new_version": version,
                "metrics": new_metrics,
                "artifact_path": str(artifact_path),
                "rollback_performed": False,
            }

        except Exception as e:
            logger.exception("Error en reentrenamiento: %s", e)
            return {"success": False, "reason": str(e), "rollback_performed": False}

    def backup_current_model(self) -> Path | None:
        """Respalda el modelo activo antes de reentrenar.

        Returns:
            Ruta al archivo de backup, o None si no hay modelo activo.
        """
        active = get_active_version()
        if active is None:
            logger.warning("No hay versión activa para respaldar")
            return None

        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        src = Path(active["artifact_path"])
        if not src.exists():
            logger.warning("Artefacto activo no encontrado: %s", src)
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = BACKUP_DIR / f"backup_{timestamp}_{src.name}"
        shutil.copy2(src, dst)
        logger.info("Backup creado: %s", dst)
        return dst

    def validate_new_model(self, new_metrics: dict[str, float]) -> dict[str, Any]:
        """Compara métricas del nuevo modelo vs el modelo actual.

        Args:
            new_metrics: Métricas del modelo recién entrenado.

        Returns:
            Diccionario con ``accepted`` (bool), ``reason`` y comparación
            de métricas.
        """
        baseline_recall = MODEL_METRICS.get("recall_glosada", self.min_recall_glosada)
        new_recall = new_metrics.get("recall_glosada", 0.0)
        new_f1 = new_metrics.get("f1_macro", 0.0)
        baseline_f1 = MODEL_METRICS.get("f1_macro", 0.0)

        accepted = new_recall >= self.min_recall_glosada and new_f1 >= baseline_f1 * 0.95

        return {
            "accepted": accepted,
            "reason": (
                "Métricas aceptables" if accepted
                else f"Recall_Glosada={new_recall:.4f} < umbral={self.min_recall_glosada:.4f}"
            ),
            "new_metrics": new_metrics,
            "baseline_metrics": MODEL_METRICS,
            "recall_glosada_delta": new_recall - baseline_recall,
            "f1_macro_delta": new_f1 - baseline_f1,
        }

    def schedule_monthly_retrain(self) -> dict[str, Any]:
        """Punto de entrada para ejecución programada mensual.

        Combina evaluación de drift y reentrenamiento en un solo paso.

        Returns:
            Resultado completo del ciclo de reentrenamiento.
        """
        logger.info("=== Ciclo mensual de reentrenamiento iniciado ===")
        result = self.check_drift_and_retrain()

        # También verificar degradación de rendimiento
        perf_check = self.monitor.detect_performance_drop(threshold=0.05)
        result["performance_drop"] = perf_check

        if perf_check.get("drop_detected") and not result.get("retrain_triggered"):
            logger.info("Degradación de rendimiento detectada — activando reentrenamiento")
            retrain_result = self.retrain_with_recent_data(self.months_back)
            result["retrain_triggered"] = True
            result["retrain_result"] = retrain_result

        logger.info("=== Ciclo mensual completado ===")
        return result

    # ------------------------------------------------------------------
    # Métodos privados de soporte
    # ------------------------------------------------------------------

    def _load_reference_and_current(
        self,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Carga datos de referencia (train) y actuales (val/test) para PSI.

        Returns:
            Tupla (reference_df, current_df) o (None, None) si no hay datos.
        """
        try:
            train_path = PROCESSED_DIR / "20260309_train.parquet"
            val_path = PROCESSED_DIR / "20260309_val.parquet"

            if not train_path.exists() or not val_path.exists():
                logger.warning("Archivos parquet no encontrados para evaluación de drift")
                return None, None

            ref = pd.read_parquet(train_path)
            cur = pd.read_parquet(val_path)

            # Seleccionar solo columnas numéricas comunes
            numeric_cols = ref.select_dtypes(include=[np.number]).columns.tolist()
            common_cols = [c for c in numeric_cols if c in cur.columns]

            return ref[common_cols], cur[common_cols]

        except Exception as e:
            logger.error("Error cargando datos para drift: %s", e)
            return None, None

    def _load_recent_splits(
        self, months_back: int
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
        """Carga splits de entrenamiento filtrados por fecha reciente.

        Args:
            months_back: Número de meses hacia atrás desde hoy.

        Returns:
            Tupla (train_df, val_df, test_df) o (None, None, None).
        """
        try:
            train_path = PROCESSED_DIR / "20260309_train.parquet"
            val_path = PROCESSED_DIR / "20260309_val.parquet"
            test_path = PROCESSED_DIR / "20260309_test.parquet"

            if not all(p.exists() for p in [train_path, val_path, test_path]):
                return None, None, None

            train_df = pd.read_parquet(train_path)
            val_df = pd.read_parquet(val_path)
            test_df = pd.read_parquet(test_path)

            # Filtrar por fecha si la columna existe
            if "fecha_radicacion" in train_df.columns:
                cutoff = pd.Timestamp.now() - pd.DateOffset(months=months_back)
                train_df = train_df[pd.to_datetime(train_df["fecha_radicacion"]) >= cutoff]
                logger.info("Datos filtrados: %d registros desde %s", len(train_df), cutoff.date())

            return train_df, val_df, test_df

        except Exception as e:
            logger.error("Error cargando splits recientes: %s", e)
            return None, None, None

    def _train_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[Any, dict[str, float]]:
        """Reentrena XGBoost con los datos proporcionados.

        Args:
            train_df: DataFrame de entrenamiento.
            val_df: DataFrame de validación.
            test_df: DataFrame de test.

        Returns:
            Tupla (modelo_entrenado, métricas_test).
        """
        import xgboost as xgb
        from sklearn.metrics import f1_score, recall_score

        from src.features.engineering import load_feature_names

        feature_names = load_feature_names()
        target_col = "estado_auditoria"

        # Preparar matrices
        X_train = train_df[feature_names].values
        y_train = train_df[target_col].values
        X_val = val_df[feature_names].values
        y_val = val_df[target_col].values
        X_test = test_df[feature_names].values
        y_test = test_df[target_col].values

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            use_label_encoder=False,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = model.predict(X_test)
        labels = [0, 1, 2]
        recall_per_class = recall_score(y_test, y_pred, labels=labels, average=None, zero_division=0)

        metrics = {
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall_auditada": float(recall_per_class[0]),
            "recall_glosada": float(recall_per_class[1]),
            "recall_devuelta": float(recall_per_class[2]),
            "accuracy": float(np.mean(y_pred == y_test)),
        }

        logger.info("Modelo reentrenado: %s", metrics)
        return model, metrics
