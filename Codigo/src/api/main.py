# -*- coding: utf-8 -*-
"""API REST para predicción de auditoría de facturas médicas.

Expone el modelo XGBoost como servicio FastAPI con explicabilidad SHAP.
Endpoints:
    - GET  /v1/health          → Health check
    - GET  /v1/model/info      → Metadata y métricas del modelo
    - POST /v1/predict         → Predicción individual con explicación
    - POST /v1/predict/batch   → Predicciones batch

Uso:
    uvicorn src.api.main:app --reload --port 8000
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    LLMStatusResponse,
    ModelInfoResponse,
    Probabilities,
    PredictionRequest,
    PredictionResponse,
    AdminRetrainResponse,
    AdminMetricsResponse,
    AdminDriftResponse,
    AdminRollbackResponse,
)
from src.config import (
    API_VERSION,
    CLASS_NAMES,
    MODEL_DATE,
    MODEL_METRICS,
    MODEL_VERSION,
    EMBEDDING_MODEL_NAME,
    setup_logging,
)
from src.explainability.shap_explanations import (
    create_shap_explainer,
    explain_batch_predictions,
    explain_single_prediction,
)
from src.features.engineering import (
    load_feature_names,
    prepare_batch_input,
    prepare_input,
)
from src.models.predict import load_model

# Importaciones LLM (opcionales - degradación graciosa si no están disponibles)
try:
    from src.privacy.guard import PrivacyGuard
    from src.embeddings.service import EmbeddingService
    from src.nlp.pipeline import load_nlp_pipeline
    from src.explainability.explanation_generator import ExplanationGenerator
    from src.monitoring.drift_detector import DriftDetector
    from src.features.embedding_pipeline import build_text_fields, enrich_features
    LLM_COMPONENTS_AVAILABLE = True
except ImportError as e:
    LLM_COMPONENTS_AVAILABLE = False

# Importaciones de reentrenamiento (opcionales)
try:
    from src.training.retraining_scheduler import RetrainingScheduler
    from src.monitoring.model_monitor import ModelMonitor
    from src.storage.model_registry import (
        get_active_version,
        list_model_versions,
        set_active_version,
    )
    RETRAINING_AVAILABLE = True
except ImportError:
    RETRAINING_AVAILABLE = False

setup_logging()
logger = logging.getLogger(__name__)

# Zona horaria de Colombia
TZ_COLOMBIA = ZoneInfo("America/Bogota")


# ---------------------------------------------------------------------------
# Lifespan: carga de artefactos al iniciar
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga modelo, features y SHAP explainer al iniciar el servidor."""
    logger.info("Iniciando carga de artefactos del modelo...")
    t0 = time.time()

    # Cargar componentes base (siempre requeridos)
    app.state.feature_names = load_feature_names()
    app.state.model = load_model()
    app.state.explainer = create_shap_explainer(app.state.model)

    # Inicializar componentes LLM (opcional - degradación graciosa)
    app.state.llm_available = False
    app.state.privacy_guard = None
    app.state.embedding_service = None
    app.state.nlp_pipeline = None
    app.state.explanation_generator = None
    app.state.drift_detector = None

    if LLM_COMPONENTS_AVAILABLE:
        try:
            logger.info("Inicializando componentes LLM...")
            
            # Privacy Guard
            app.state.privacy_guard = PrivacyGuard()
            app.state.privacy_guard.verify_local_only()
            
            # Verificar si Ollama está disponible
            ollama_available = app.state.privacy_guard.check_ollama_available()
            
            if not ollama_available:
                logger.warning("Ollama no disponible - continuando sin componentes LLM")
            else:
                # NLP Pipeline
                app.state.nlp_pipeline = load_nlp_pipeline()
                
                # Embedding Service
                app.state.embedding_service = EmbeddingService(
                    model_name=EMBEDDING_MODEL_NAME
                )
                
                # Explanation Generator
                app.state.explanation_generator = ExplanationGenerator(
                    privacy_guard=app.state.privacy_guard
                )
                
                # Drift Detector
                app.state.drift_detector = DriftDetector()
                
                app.state.llm_available = True
                logger.info("Componentes LLM inicializados exitosamente")
                
        except Exception as e:
            logger.warning("Error inicializando componentes LLM: %s", e)
            logger.info("Continuando sin funcionalidades LLM (degradación graciosa)")
    else:
        logger.info("Componentes LLM no disponibles - continuando con funcionalidad base")

    # Inicializar componentes de reentrenamiento
    app.state.model_monitor = None
    app.state.retraining_scheduler = None
    if RETRAINING_AVAILABLE:
        try:
            app.state.model_monitor = ModelMonitor()
            app.state.retraining_scheduler = RetrainingScheduler(
                monitor=app.state.model_monitor
            )
            logger.info("Sistema de reentrenamiento inicializado")
        except Exception as e:
            logger.warning("Error inicializando sistema de reentrenamiento: %s", e)

    elapsed = time.time() - t0
    logger.info("Artefactos cargados exitosamente en %.1fs", elapsed)

    yield

    logger.info("Servidor detenido")


# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="API Predicción Auditoría — Savia Salud EPS",
    description=(
        "Servicio de clasificación predictiva de facturas médicas en tres "
        "estados (Auditada, Glosada, Devuelta) con explicabilidad SHAP. "
        "Proyecto de tesis — Universidad EAFIT."
    ),
    version=f"{API_VERSION}.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(f"/{API_VERSION}/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Verifica el estado del servicio y si el modelo está cargado."""
    model_loaded = hasattr(app.state, "model") and app.state.model is not None
    return HealthResponse(
        status="ok" if model_loaded else "error",
        modelo_cargado=model_loaded,
        version=f"{API_VERSION}.0.0",
        timestamp=datetime.now(TZ_COLOMBIA),
    )


@app.get(f"/{API_VERSION}/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Retorna metadata del modelo desplegado, features y métricas."""
    return ModelInfoResponse(
        modelo="XGBoost",
        version=MODEL_VERSION,
        fecha_entrenamiento=MODEL_DATE,
        metricas=MODEL_METRICS,
        n_features=len(app.state.feature_names),
        features=app.state.feature_names,
        clases={str(k): v for k, v in CLASS_NAMES.items()},
    )


@app.get(f"/{API_VERSION}/llm/status", response_model=LLMStatusResponse)
async def llm_status() -> LLMStatusResponse:
    """Retorna el estado del sistema LLM local."""
    if not app.state.llm_available:
        return LLMStatusResponse(
            ollama_available=False,
            active_model=None,
            embedding_model=EMBEDDING_MODEL_NAME,
            latency_p50_ms=None,
            latency_p95_ms=None,
            drift_monitoring_active=False,
            last_drift_check=None,
        )
    
    try:
        # Obtener estado del explanation generator
        explanation_status = app.state.explanation_generator.get_status()
        
        # Medir latencia con test de generación
        test_result = app.state.explanation_generator.test_generation()
        
        return LLMStatusResponse(
            ollama_available=explanation_status["ollama_available"],
            active_model=explanation_status["llm_model"],
            embedding_model=explanation_status.get("embedding_model", EMBEDDING_MODEL_NAME),
            latency_p50_ms=test_result.get("elapsed_time_s", 0) * 1000 if test_result["success"] else None,
            latency_p95_ms=None,  # Requeriría múltiples mediciones
            drift_monitoring_active=app.state.drift_detector is not None,
            last_drift_check=None,  # Se actualizaría con sistema de monitoreo
        )
    except Exception as e:
        logger.error("Error obteniendo estado LLM: %s", e)
        return LLMStatusResponse(
            ollama_available=False,
            active_model=None,
            embedding_model=EMBEDDING_MODEL_NAME,
            latency_p50_ms=None,
            latency_p95_ms=None,
            drift_monitoring_active=False,
            last_drift_check=None,
        )


@app.post(f"/{API_VERSION}/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest) -> PredictionResponse:
    """Predicción individual con explicación SHAP y enriquecimiento LLM opcional.

    Recibe las 48 features pre-computadas de una factura y opcionalmente
    campos textuales para enriquecimiento semántico. Retorna la clasificación
    predicha con factores de riesgo en lenguaje natural.
    """
    try:
        features_dict = request.features.model_dump()
        
        # Verificar si hay campos textuales para enriquecimiento
        has_text_fields = any([
            request.diagnostico_texto,
            request.cups_texto,
            request.observaciones
        ])
        
        # Preparar features base
        features_df = prepare_input(features_dict, app.state.feature_names)
        
        # Enriquecimiento con embeddings si está disponible y hay campos textuales
        if app.state.llm_available and has_text_fields:
            try:
                logger.info("Enriqueciendo features con embeddings semánticos")
                
                # Construir campos textuales
                text_fields = {
                    "diagnostico_texto": request.diagnostico_texto,
                    "cups_texto": request.cups_texto,
                    "observaciones": request.observaciones
                }
                
                # Enriquecer features
                numeric_features = features_df.iloc[0].values
                enriched_features = enrich_features(
                    numeric_features,
                    text_fields,
                    app.state.embedding_service,
                    app.state.nlp_pipeline
                )
                
                # Crear DataFrame enriquecido (nota: esto requeriría un modelo entrenado con features enriquecidas)
                # Por ahora, usamos el modelo base pero registramos que se hizo enriquecimiento
                logger.info("Features enriquecidas generadas (dimensión: %d)", len(enriched_features))
                
            except Exception as e:
                logger.warning("Error en enriquecimiento, usando features base: %s", e)

        # Predicción con modelo base
        result = explain_single_prediction(
            features_df=features_df,
            model=app.state.model,
            explainer=app.state.explainer,
            feature_names=app.state.feature_names,
            top_n=5,
        )

        # Generar explicación enriquecida si LLM está disponible
        explicacion_enriquecida = None
        if app.state.llm_available:
            try:
                # Extraer valores SHAP y features para el generador
                shap_values = {
                    factor.feature: factor.shap_value 
                    for factor in result["factores_detalle"]
                }
                feature_values = {
                    factor.feature: factor.feature_value 
                    for factor in result["factores_detalle"]
                }
                
                explicacion_enriquecida = app.state.explanation_generator.generate(
                    prediction=result["clase"],
                    confidence=result["confianza"],
                    shap_values=shap_values,
                    feature_values=feature_values
                )
                
                logger.info("Explicación enriquecida generada exitosamente")
                
            except Exception as e:
                logger.warning("Error generando explicación enriquecida: %s", e)

        return PredictionResponse(
            numero_factura=request.numero_factura,
            estado_predicho=result["estado_predicho"],
            clase=result["clase"],
            confianza=result["confianza"],
            probabilidades=Probabilities(**result["probabilidades"]),
            factores_riesgo=result["factores_riesgo"],
            factores_detalle=result["factores_detalle"],
            causal_probable=result["causal_probable"],
            accion_sugerida=result["accion_sugerida"],
            timestamp_prediccion=datetime.now(TZ_COLOMBIA),
            explicacion_enriquecida=explicacion_enriquecida,
            drift_warning=None,  # Se implementaría con monitoreo continuo
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("Error en predicción individual: %s", e)
        raise HTTPException(status_code=500, detail="Error interno del servidor") from e


@app.post(f"/{API_VERSION}/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Predicciones batch con explicaciones SHAP y enriquecimiento LLM opcional.

    Procesa hasta 1000 facturas en una sola llamada. Retorna predicciones
    individuales con tiempo total de procesamiento.
    """
    if len(request.facturas) > 1000:
        raise HTTPException(
            status_code=413,
            detail=f"Máximo 1000 facturas por batch, recibidas: {len(request.facturas)}",
        )

    try:
        t0 = time.time()

        features_list = [f.features.model_dump() for f in request.facturas]
        features_df = prepare_batch_input(features_list, app.state.feature_names)

        results = explain_batch_predictions(
            features_df=features_df,
            model=app.state.model,
            explainer=app.state.explainer,
            feature_names=app.state.feature_names,
            top_n=5,
        )

        now = datetime.now(TZ_COLOMBIA)
        predictions = []
        
        for i, result in enumerate(results):
            # Generar explicación enriquecida si LLM está disponible
            explicacion_enriquecida = None
            if app.state.llm_available:
                try:
                    shap_values = {
                        factor.feature: factor.shap_value 
                        for factor in result["factores_detalle"]
                    }
                    feature_values = {
                        factor.feature: factor.feature_value 
                        for factor in result["factores_detalle"]
                    }
                    
                    explicacion_enriquecida = app.state.explanation_generator.generate(
                        prediction=result["clase"],
                        confidence=result["confianza"],
                        shap_values=shap_values,
                        feature_values=feature_values
                    )
                except Exception as e:
                    logger.warning("Error generando explicación enriquecida para factura %d: %s", i, e)
            
            predictions.append(
                PredictionResponse(
                    numero_factura=request.facturas[i].numero_factura,
                    estado_predicho=result["estado_predicho"],
                    clase=result["clase"],
                    confianza=result["confianza"],
                    probabilidades=Probabilities(**result["probabilidades"]),
                    factores_riesgo=result["factores_riesgo"],
                    factores_detalle=result["factores_detalle"],
                    causal_probable=result["causal_probable"],
                    accion_sugerida=result["accion_sugerida"],
                    timestamp_prediccion=now,
                    explicacion_enriquecida=explicacion_enriquecida,
                    drift_warning=None,
                )
            )

        elapsed_ms = (time.time() - t0) * 1000

        return BatchPredictionResponse(
            predicciones=predictions,
            total=len(predictions),
            tiempo_procesamiento_ms=round(elapsed_ms, 1),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("Error en predicción batch: %s", e)
        raise HTTPException(status_code=500, detail="Error interno del servidor") from e


# ---------------------------------------------------------------------------
# Endpoints administrativos (reentrenamiento y monitoreo)
# ---------------------------------------------------------------------------

_ADMIN_PREFIX = f"/{API_VERSION}/admin"


@app.post(f"{_ADMIN_PREFIX}/retrain", response_model=AdminRetrainResponse)
async def admin_retrain() -> AdminRetrainResponse:
    """Activa reentrenamiento manual del modelo XGBoost.

    Requiere que el sistema de reentrenamiento esté disponible.
    Retorna 503 si no está inicializado.
    """
    if not RETRAINING_AVAILABLE or app.state.retraining_scheduler is None:
        raise HTTPException(
            status_code=503,
            detail="Sistema de reentrenamiento no disponible",
        )

    try:
        result = app.state.retraining_scheduler.retrain_with_recent_data()
        return AdminRetrainResponse(
            triggered=True,
            success=result.get("success", False),
            new_version=result.get("new_version"),
            metrics=result.get("metrics"),
            reason=result.get("reason"),
            timestamp=datetime.now(TZ_COLOMBIA),
        )
    except Exception as e:
        logger.exception("Error en reentrenamiento manual: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get(f"{_ADMIN_PREFIX}/model/metrics", response_model=AdminMetricsResponse)
async def admin_model_metrics() -> AdminMetricsResponse:
    """Retorna métricas actuales del modelo en producción."""
    from src.config import MODEL_METRICS

    recent_metrics = None
    performance_drop = False
    active_version = None

    if RETRAINING_AVAILABLE:
        try:
            if app.state.model_monitor:
                history = app.state.model_monitor.get_model_metrics_history(last_n=7)
                if history:
                    import numpy as np
                    recent_metrics = {
                        "f1_macro": float(np.mean([r["f1_macro"] for r in history])),
                        "recall_glosada": float(np.mean([r["recall_glosada"] for r in history])),
                        "recall_auditada": float(np.mean([r["recall_auditada"] for r in history])),
                        "recall_devuelta": float(np.mean([r["recall_devuelta"] for r in history])),
                    }
                    drop_check = app.state.model_monitor.detect_performance_drop()
                    performance_drop = drop_check.get("drop_detected", False)

            active = get_active_version()
            if active:
                active_version = active.get("version")
        except Exception as e:
            logger.warning("Error obteniendo métricas administrativas: %s", e)

    return AdminMetricsResponse(
        baseline_metrics=MODEL_METRICS,
        recent_metrics=recent_metrics,
        performance_drop_detected=performance_drop,
        active_version=active_version,
        timestamp=datetime.now(TZ_COLOMBIA),
    )


@app.get(f"{_ADMIN_PREFIX}/drift/status", response_model=AdminDriftResponse)
async def admin_drift_status() -> AdminDriftResponse:
    """Retorna el estado actual de concept drift del modelo."""
    if not RETRAINING_AVAILABLE or app.state.retraining_scheduler is None:
        return AdminDriftResponse(
            drift_status="sin_datos",
            mean_psi=None,
            retrain_recommended=False,
            timestamp=datetime.now(TZ_COLOMBIA),
        )

    try:
        ref_data, cur_data = app.state.retraining_scheduler._load_reference_and_current()
        if ref_data is None:
            return AdminDriftResponse(
                drift_status="sin_datos",
                mean_psi=None,
                retrain_recommended=False,
                timestamp=datetime.now(TZ_COLOMBIA),
            )

        import numpy as np
        from src.monitoring.drift_detector import compute_psi, check_drift

        psi_values = []
        for col in ref_data.columns:
            try:
                psi = compute_psi(ref_data[col].values, cur_data[col].values)
                psi_values.append(psi)
            except Exception:
                pass

        mean_psi = float(np.mean(psi_values)) if psi_values else 0.0
        drift_status = check_drift(mean_psi)

        return AdminDriftResponse(
            drift_status=drift_status,
            mean_psi=mean_psi,
            retrain_recommended=drift_status in ("warning", "alert"),
            timestamp=datetime.now(TZ_COLOMBIA),
        )
    except Exception as e:
        logger.exception("Error evaluando drift: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post(f"{_ADMIN_PREFIX}/model/rollback", response_model=AdminRollbackResponse)
async def admin_model_rollback(target_version: str | None = None) -> AdminRollbackResponse:
    """Revierte el modelo activo a una versión anterior.

    Args:
        target_version: Versión específica a activar. Si None, activa la
            penúltima versión disponible.
    """
    if not RETRAINING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Sistema de versionado no disponible",
        )

    try:
        versions = list_model_versions()
        if len(versions) < 2:
            return AdminRollbackResponse(
                success=False,
                reason="No hay versiones anteriores disponibles para rollback",
                timestamp=datetime.now(TZ_COLOMBIA),
            )

        if target_version is None:
            # Activar la penúltima versión (la segunda más reciente)
            target_version = versions[1]["version"]

        set_active_version(target_version)
        logger.info("Rollback ejecutado a versión: %s", target_version)

        return AdminRollbackResponse(
            success=True,
            rolled_back_to=target_version,
            timestamp=datetime.now(TZ_COLOMBIA),
        )
    except KeyError as e:
        return AdminRollbackResponse(
            success=False,
            reason=str(e),
            timestamp=datetime.now(TZ_COLOMBIA),
        )
    except Exception as e:
        logger.exception("Error en rollback: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
