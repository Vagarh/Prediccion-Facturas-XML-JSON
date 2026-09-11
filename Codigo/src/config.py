# -*- coding: utf-8 -*-
"""Configuración central del proyecto de analítica predictiva para auditoría prestacional.

Contiene rutas, constantes del modelo, descripciones de riesgo en lenguaje natural,
mapeo de features a causales de glosa (Res. 2284/2023) y configuración de logging.
"""

import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------

BASE_DIR = Path(os.environ.get("SAVIA_BASE_DIR", Path(__file__).resolve().parent.parent))
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_PATH = MODELS_DIR / "artifacts" / "20260910_lgbm_tuned_champion.joblib"
FEATURE_LIST_PATH = DATA_DIR / "processed" / "20260910_feature_list.txt"

# ---------------------------------------------------------------------------
# Constantes del modelo
# ---------------------------------------------------------------------------

RANDOM_STATE: int = 42
DATE_PREFIX: str = "20260910"
API_VERSION: str = "v1"
MODEL_VERSION: str = "v3"
MODEL_DATE: str = "2026-09-10"
MODEL_ALGORITHM: str = "LightGBM"

CLASS_NAMES: dict[int, str] = {
    0: "Auditada",
    1: "Glosada",
    2: "Devuelta",
}

CLASS_ACTIONS: dict[int, str] = {
    0: "Fast-track: pago programado sin auditoría manual",
    1: "Enviar a auditor médico especializado para revisión detallada",
    2: "Notificar al prestador para corrección y re-radicación",
}

# Modelo v3 (2026-09): LightGBM afinado (Optuna) sobre datos frescos mar-2024/may-2026,
# reentrenado tras detectar que el v2 (2026-04) degradaba fuerte en producción por
# concept drift (ver .claude/09_historial_sesiones.md, sesión 2026-09-10/11).
MODEL_METRICS: dict[str, float] = {
    "f1_macro": 0.7577,
    "auc_roc_ovr": 0.9060,
    "recall_auditada": 0.5883,
    "recall_glosada": 0.8635,
    "recall_devuelta": 0.8568,
    "accuracy": 0.8058,
}

# ---------------------------------------------------------------------------
# Etiquetas de features en lenguaje natural (para reportes y explicaciones)
# ---------------------------------------------------------------------------

FEATURE_LABELS: dict[str, str] = {
    "valor_factura": "Valor de la factura (COP)",
    "num_items": "Número de ítems facturados",
    "num_cups_distintos": "Número de CUPS distintos",
    "num_dx_distintos": "Número de diagnósticos distintos",
    "sum_valor_facturado": "Suma total facturada en detalles",
    "max_valor_item": "Valor del ítem más costoso",
    "avg_valor_item": "Valor promedio por ítem",
    "num_items_con_autorizacion": "Ítems con autorización previa",
    "num_tipos_servicio": "Tipos de servicio distintos",
    "dias_hasta_radicacion": "Días hasta radicación",
    "valor_copago": "Valor copago",
    "valor_cuota_moderadora": "Valor cuota moderadora",
    "tasa_glosa_nit": "Tasa histórica de glosas del prestador",
    "tasa_devolucion_nit": "Tasa histórica de devoluciones del prestador",
    "total_facturas_nit": "Total facturas históricas del prestador",
    "valor_promedio_nit": "Valor promedio histórico del prestador",
    "dias_radicacion_promedio_nit": "Días promedio de radicación del prestador",
    "porcentaje_mayor_item": "% del ítem de mayor valor sobre total",
    "discrepancia_valor": "Discrepancia valor factura vs detalles",
    "tiene_fev": "Tiene factura electrónica (FEV)",
    "is_extemporanea": "Radicación extemporánea (>22 días)",
    "ratio_valor_vs_nit": "Ratio valor factura vs promedio prestador",
    "log_valor_x_tasa_glosa": "Riesgo compuesto valor × tasa glosa",
    "ratio_autorizacion": "Ratio de ítems con autorización previa",
    "ratio_cups_items": "Diversidad CUPS/ítems",
}

# ---------------------------------------------------------------------------
# Descripciones de riesgo en lenguaje natural para auditores
# Formato: (is_high_risky, texto_alto, texto_bajo)
# ---------------------------------------------------------------------------

RISK_DESCRIPTIONS: dict[str, tuple[bool, str, str]] = {
    "tasa_glosa_nit": (
        True,
        "Tasa histórica de glosas del prestador es alta ({val:.1%})",
        "Tasa histórica de glosas del prestador es baja ({val:.1%})",
    ),
    "tasa_devolucion_nit": (
        True,
        "Tasa histórica de devoluciones del prestador es alta ({val:.1%})",
        "Tasa histórica de devoluciones del prestador es baja ({val:.1%})",
    ),
    "num_dx_distintos": (
        True,
        "Alto número de diagnósticos distintos ({val:.0f}) — posible facturación compleja",
        "Bajo número de diagnósticos ({val:.0f})",
    ),
    "num_tipos_servicio": (
        True,
        "Alta diversidad de tipos de servicio ({val:.0f})",
        "Baja diversidad de tipos de servicio ({val:.0f})",
    ),
    "num_items_con_autorizacion": (
        True,
        "Muchos ítems con autorización previa ({val:.0f})",
        "Pocos ítems con autorización previa ({val:.0f})",
    ),
    "valor_factura": (
        True,
        "Valor de factura elevado (${val:,.0f} COP)",
        "Valor de factura bajo (${val:,.0f} COP)",
    ),
    "dias_hasta_radicacion": (
        True,
        "Radicación tardía ({val:.0f} días después de la prestación)",
        "Radicación oportuna ({val:.0f} días)",
    ),
    "is_extemporanea": (
        True,
        "Radicación extemporánea (>22 días hábiles) — causal DE16",
        "Radicación dentro de plazo legal",
    ),
    "ratio_valor_vs_nit": (
        True,
        "Valor {val:.1f}x por encima del promedio del prestador",
        "Valor {val:.1f}x respecto al promedio del prestador (dentro de rango)",
    ),
    "log_valor_x_tasa_glosa": (
        True,
        "Combinación de alto valor y alta tasa de glosa (riesgo compuesto)",
        "Bajo riesgo compuesto valor-glosa",
    ),
    "discrepancia_valor": (
        True,
        "Discrepancia de ${val:,.0f} entre valor facturado y suma de detalles",
        "Consistencia entre valor facturado y detalles",
    ),
    "ratio_autorizacion": (
        False,
        "Bajo ratio de autorización previa ({val:.1%})",
        "Alto ratio de autorización previa ({val:.1%})",
    ),
    "porcentaje_mayor_item": (
        True,
        "Un solo ítem concentra {val:.0%} del valor total de la factura",
        "Distribución equilibrada de valores entre ítems",
    ),
    "ratio_cups_items": (
        True,
        "Alta diversidad CUPS/ítems ({val:.2f}) — posible duplicidad FA0301",
        "Baja diversidad CUPS/ítems",
    ),
    "tiene_fev": (
        False,
        "Sin factura electrónica de venta (FEV)",
        "Tiene FEV electrónica",
    ),
    "desviacion_dias_nit": (
        True,
        "Días de radicación {val:.0f}x por encima del promedio del prestador",
        "Días de radicación dentro del rango habitual del prestador",
    ),
}

# ---------------------------------------------------------------------------
# Mapeo de features a causales de glosa (Res. 2284/2023)
# ---------------------------------------------------------------------------

FEATURE_TO_CAUSAL: dict[str, str] = {
    "is_extemporanea": "DE16",
    "dias_hasta_radicacion": "DE16",
    "ratio_cups_items": "FA0301",
    "discrepancia_valor": "FA0201",
    "tasa_glosa_nit": "SO",
    "num_items_con_autorizacion": "SO0201",
    "ratio_autorizacion": "SO0201",
    "porcentaje_mayor_item": "TA0101",
    "valor_factura": "TA0101",
}

# ---------------------------------------------------------------------------
# Configuración LLM Local (Fase 7 — Integración LLM)
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
"""URL base del servidor Ollama. Debe apuntar a localhost para cumplir Ley 1581."""

EMBEDDING_MODEL_NAME: str = os.environ.get(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
"""Nombre del modelo Sentence Transformers para generación de embeddings."""

EMBEDDING_DIM: int = 384
"""Dimensión fija del vector de embeddings producido por el modelo de embeddings."""

OLLAMA_TIMEOUT_S: float = float(os.environ.get("OLLAMA_TIMEOUT_S", "30.0"))
"""Timeout en segundos para llamadas al servidor Ollama."""

PSI_DRIFT_WARN_THRESHOLD: float = 0.20
"""Umbral PSI para advertencia de concept drift (PSI > 0.20 → advertencia)."""

PSI_DRIFT_ALERT_THRESHOLD: float = 0.15
"""Umbral PSI para alerta de concept drift (PSI > 0.15 → alerta)."""

LLM_CANDIDATES: list[str] = ["llama3.2", "mistral", "qwen2.5", "phi3"]
"""Modelos LLM candidatos a evaluar mediante Ollama."""

EXPLANATION_MAX_SECONDS: float = 10.0
"""Tiempo máximo (segundos, p95) para generar una explicación en lenguaje natural."""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(level: int = logging.INFO) -> None:
    """Configura el logging del proyecto con formato estándar.

    Args:
        level: Nivel de logging (por defecto INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
