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
    ModelInfoResponse,
    Probabilities,
    PredictionRequest,
    PredictionResponse,
)
from src.config import (
    API_VERSION,
    CLASS_NAMES,
    MODEL_DATE,
    MODEL_METRICS,
    MODEL_VERSION,
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

    app.state.feature_names = load_feature_names()
    app.state.model = load_model()
    app.state.explainer = create_shap_explainer(app.state.model)

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


@app.post(f"/{API_VERSION}/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest) -> PredictionResponse:
    """Predicción individual con explicación SHAP.

    Recibe las 48 features pre-computadas de una factura y retorna la
    clasificación predicha con factores de riesgo en lenguaje natural.
    """
    try:
        features_dict = request.features.model_dump()
        features_df = prepare_input(features_dict, app.state.feature_names)

        result = explain_single_prediction(
            features_df=features_df,
            model=app.state.model,
            explainer=app.state.explainer,
            feature_names=app.state.feature_names,
            top_n=5,
        )

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
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("Error en predicción individual: %s", e)
        raise HTTPException(status_code=500, detail="Error interno del servidor") from e


@app.post(f"/{API_VERSION}/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Predicciones batch con explicaciones SHAP.

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
