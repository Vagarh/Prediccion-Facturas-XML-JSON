# -*- coding: utf-8 -*-
"""Schemas Pydantic v2 para la API REST de predicción de auditoría.

Define los modelos de entrada y salida para todos los endpoints,
incluyendo validación estricta de las 48 features del modelo.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Entrada: Features de la factura
# ---------------------------------------------------------------------------


class FeatureInput(BaseModel):
    """Las 48 features pre-computadas de una factura médica.

    Cada campo corresponde a una feature del modelo XGBoost v1.
    Los valores deben ser numéricos (float). No se aceptan campos adicionales.
    """

    model_config = ConfigDict(extra="forbid")

    anio_radicacion: float = Field(description="Año de radicación de la factura")
    avg_valor_item: float = Field(description="Valor promedio por ítem facturado")
    desviacion_dias_nit: float = Field(
        description="Desviación de días de radicación respecto al promedio del prestador"
    )
    dias_hasta_radicacion: float = Field(
        description="Días entre prestación y radicación"
    )
    dias_radicacion_promedio_nit: float = Field(
        description="Días promedio de radicación del prestador"
    )
    dias_x_log_valor: float = Field(
        description="Interacción: días de radicación × log(valor)"
    )
    discrepancia_valor: float = Field(
        description="Diferencia absoluta entre valor factura y suma de detalles"
    )
    is_extemporanea: float = Field(
        description="Flag: radicación extemporánea (>22 días hábiles)"
    )
    log_avg_valor_item: float = Field(description="Log1p del valor promedio por ítem")
    log_discrepancia_valor: float = Field(
        description="Log1p de la discrepancia de valor"
    )
    log_max_valor_item: float = Field(
        description="Log1p del valor del ítem más costoso"
    )
    log_num_cups_distintos: float = Field(
        description="Log1p del número de CUPS distintos"
    )
    log_num_dx_distintos: float = Field(
        description="Log1p del número de diagnósticos distintos"
    )
    log_num_items: float = Field(description="Log1p del número de ítems")
    log_num_items_con_autorizacion: float = Field(
        description="Log1p de ítems con autorización previa"
    )
    log_sum_valor_facturado: float = Field(
        description="Log1p de la suma total facturada"
    )
    log_total_facturas_nit: float = Field(
        description="Log1p del total de facturas históricas del prestador"
    )
    log_valor_copago: float = Field(description="Log1p del valor de copago")
    log_valor_cuota_moderadora: float = Field(
        description="Log1p de la cuota moderadora"
    )
    log_valor_factura: float = Field(description="Log1p del valor de la factura")
    log_valor_promedio_nit: float = Field(
        description="Log1p del valor promedio histórico del prestador"
    )
    log_valor_x_tasa_glosa: float = Field(
        description="Riesgo compuesto: log(valor) × tasa de glosa del prestador"
    )
    max_valor_item: float = Field(description="Valor del ítem más costoso (COP)")
    mes_radicacion: float = Field(description="Mes de radicación (1-12)")
    num_cups_distintos: float = Field(
        description="Número de códigos CUPS distintos"
    )
    num_dx_distintos: float = Field(
        description="Número de diagnósticos CIE-10 distintos"
    )
    num_items: float = Field(description="Número total de ítems facturados")
    num_items_con_autorizacion: float = Field(
        description="Número de ítems con autorización previa"
    )
    num_tipos_servicio: float = Field(
        description="Número de tipos de servicio distintos"
    )
    porcentaje_mayor_item: float = Field(
        description="Porcentaje del ítem de mayor valor sobre el total"
    )
    ratio_autorizacion: float = Field(
        description="Ratio de ítems con autorización previa / total ítems"
    )
    ratio_cups_items: float = Field(
        description="Ratio CUPS distintos / total ítems (diversidad)"
    )
    ratio_valor_vs_nit: float = Field(
        description="Ratio valor factura / valor promedio del prestador"
    )
    regimen_encoded: float = Field(
        description="Régimen codificado (0=Subsidiado, 1=Contributivo)"
    )
    sin_detalles: float = Field(description="Flag: factura sin detalles de ítems")
    sum_valor_facturado: float = Field(
        description="Suma total de valores en detalle (COP)"
    )
    tasa_devolucion_nit: float = Field(
        description="Tasa histórica de devoluciones del prestador"
    )
    tasa_glosa_nit: float = Field(
        description="Tasa histórica de glosas del prestador"
    )
    tiene_copago: float = Field(description="Flag: la factura tiene copago")
    tiene_cuota_mod: float = Field(
        description="Flag: la factura tiene cuota moderadora"
    )
    tiene_fev: float = Field(
        description="Flag: tiene factura electrónica de venta (FEV)"
    )
    tipo_contrato_encoded: float = Field(
        description="Tipo de contrato codificado (0=CAPITA, 1=Evento, 2=Paquete)"
    )
    total_facturas_nit: float = Field(
        description="Total de facturas históricas del prestador (24 meses)"
    )
    trimestre_radicacion: float = Field(
        description="Trimestre de radicación (1-4)"
    )
    valor_copago: float = Field(description="Valor del copago (COP)")
    valor_cuota_moderadora: float = Field(
        description="Valor de la cuota moderadora (COP)"
    )
    valor_factura: float = Field(description="Valor total de la factura (COP)")
    valor_promedio_nit: float = Field(
        description="Valor promedio histórico de facturas del prestador (COP)"
    )


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class PredictionRequest(BaseModel):
    """Solicitud de predicción para una factura individual."""

    numero_factura: str = Field(
        description="Número identificador de la factura (e.g., FEV-2024-001234)"
    )
    features: FeatureInput = Field(
        description="Las 48 features pre-computadas de la factura"
    )
    # Campos textuales opcionales para enriquecimiento semántico
    diagnostico_texto: str | None = Field(
        default=None,
        description="Texto del diagnóstico principal CIE-10 (opcional para enriquecimiento)"
    )
    cups_texto: str | None = Field(
        default=None,
        description="Descripción del código CUPS principal (opcional para enriquecimiento)"
    )
    observaciones: str | None = Field(
        default=None,
        description="Observaciones clínicas del auditor (opcional para enriquecimiento)"
    )


class BatchPredictionRequest(BaseModel):
    """Solicitud de predicción para múltiples facturas."""

    facturas: list[PredictionRequest] = Field(
        max_length=1000,
        description="Lista de facturas a clasificar (máximo 1000)",
    )


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class RiskFactor(BaseModel):
    """Detalle de un factor de riesgo con valor SHAP."""

    feature: str = Field(description="Nombre de la feature")
    shap_value: float = Field(description="Valor SHAP (contribución a la predicción)")
    feature_value: float = Field(description="Valor numérico de la feature")
    descripcion: str = Field(
        description="Descripción en lenguaje natural del factor de riesgo"
    )


class Probabilities(BaseModel):
    """Probabilidades por clase."""

    auditada: float = Field(description="Probabilidad de Auditada (clase 0)")
    glosada: float = Field(description="Probabilidad de Glosada (clase 1)")
    devuelta: float = Field(description="Probabilidad de Devuelta (clase 2)")


class PredictionResponse(BaseModel):
    """Respuesta de predicción para una factura individual."""

    numero_factura: str
    estado_predicho: str = Field(
        description="Estado predicho: auditada, glosada o devuelta"
    )
    clase: int = Field(description="Código de clase: 0=Auditada, 1=Glosada, 2=Devuelta")
    confianza: float = Field(description="Probabilidad de la clase predicha")
    probabilidades: Probabilities
    factores_riesgo: list[str] = Field(
        description="Top factores de riesgo en lenguaje natural"
    )
    factores_detalle: list[RiskFactor] = Field(
        default=[],
        description="Detalle técnico de factores con valores SHAP",
    )
    causal_probable: str | None = Field(
        default=None,
        description="Código causal probable según Res. 2284/2023 (e.g., SO0101, DE16)",
    )
    accion_sugerida: str = Field(
        description="Acción recomendada para el auditor"
    )
    timestamp_prediccion: datetime = Field(
        description="Fecha y hora de la predicción (zona horaria Colombia)"
    )
    # Campos opcionales para integración LLM
    explicacion_enriquecida: str | None = Field(
        default=None,
        description="Explicación en lenguaje natural generada por LLM (si disponible)"
    )
    drift_warning: str | None = Field(
        default=None,
        description="Advertencia de concept drift si PSI > umbral (si aplica)"
    )


class BatchPredictionResponse(BaseModel):
    """Respuesta de predicción batch."""

    predicciones: list[PredictionResponse]
    total: int = Field(description="Número total de facturas procesadas")
    tiempo_procesamiento_ms: float = Field(
        description="Tiempo total de procesamiento en milisegundos"
    )


class ModelInfoResponse(BaseModel):
    """Información del modelo desplegado."""

    modelo: str = Field(description="Nombre del algoritmo")
    version: str = Field(description="Versión del modelo")
    fecha_entrenamiento: str = Field(description="Fecha de entrenamiento")
    metricas: dict[str, float] = Field(description="Métricas de evaluación en test set")
    n_features: int = Field(description="Número de features de entrada")
    features: list[str] = Field(description="Lista de nombres de features")
    clases: dict[str, str] = Field(
        description="Mapeo código → nombre de clase"
    )


class HealthResponse(BaseModel):
    """Respuesta del health check."""

    status: str = Field(description="Estado del servicio: ok / error")
    modelo_cargado: bool = Field(description="Si el modelo está cargado en memoria")
    version: str = Field(description="Versión de la API")
    timestamp: datetime = Field(description="Fecha y hora del health check")


class LLMStatusResponse(BaseModel):
    """Estado del sistema LLM local."""

    ollama_available: bool = Field(description="Si Ollama está disponible y responde")
    active_model: str | None = Field(
        default=None,
        description="Modelo LLM activo en Ollama (si disponible)"
    )
    embedding_model: str = Field(description="Modelo de embeddings en uso")
    latency_p50_ms: float | None = Field(
        default=None,
        description="Latencia percentil 50 de generación LLM (ms)"
    )
    latency_p95_ms: float | None = Field(
        default=None,
        description="Latencia percentil 95 de generación LLM (ms)"
    )
    drift_monitoring_active: bool = Field(
        description="Si el monitoreo de drift está activo"
    )
    last_drift_check: datetime | None = Field(
        default=None,
        description="Fecha del último chequeo de drift"
    )


# ---------------------------------------------------------------------------
# Schemas administrativos (reentrenamiento)
# ---------------------------------------------------------------------------


class AdminRetrainResponse(BaseModel):
    """Respuesta del endpoint de reentrenamiento manual."""

    triggered: bool = Field(description="Si el reentrenamiento fue activado")
    success: bool | None = Field(default=None, description="Si el reentrenamiento fue exitoso")
    new_version: str | None = Field(default=None, description="Nueva versión del modelo")
    metrics: dict[str, float] | None = Field(default=None, description="Métricas del nuevo modelo")
    reason: str | None = Field(default=None, description="Razón de fallo si no fue exitoso")
    timestamp: datetime = Field(description="Timestamp de la operación")


class AdminMetricsResponse(BaseModel):
    """Métricas actuales del modelo en producción."""

    baseline_metrics: dict[str, float] = Field(description="Métricas del modelo base")
    recent_metrics: dict[str, float] | None = Field(
        default=None, description="Métricas recientes (últimos 7 registros)"
    )
    performance_drop_detected: bool = Field(description="Si se detectó degradación")
    active_version: str | None = Field(default=None, description="Versión activa del modelo")
    timestamp: datetime = Field(description="Timestamp de la consulta")


class AdminDriftResponse(BaseModel):
    """Estado de concept drift del modelo."""

    drift_status: str = Field(description="Estado: stable, warning, alert, sin_datos")
    mean_psi: float | None = Field(default=None, description="PSI medio calculado")
    retrain_recommended: bool = Field(description="Si se recomienda reentrenamiento")
    timestamp: datetime = Field(description="Timestamp del chequeo")


class AdminRollbackResponse(BaseModel):
    """Respuesta del endpoint de rollback de modelo."""

    success: bool = Field(description="Si el rollback fue exitoso")
    rolled_back_to: str | None = Field(default=None, description="Versión a la que se revirtió")
    reason: str | None = Field(default=None, description="Razón de fallo si no fue exitoso")
    timestamp: datetime = Field(description="Timestamp de la operación")
