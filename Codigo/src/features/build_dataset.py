# -*- coding: utf-8 -*-
"""Pipeline de feature engineering — de dataset crudo a splits temporales.

Port reproducible de ``notebooks/04_Feature_Engineering.ipynb`` (celdas 2-11).
Antes esta lógica vivía sólo en el notebook; ``src/features/engineering.py`` sólo
valida el vector de 48 features en inferencia, no lo construye.

Contrato: la salida de :func:`build_feature_frame` contiene **exactamente las 48
features** listadas en ``data/processed/20260309_feature_list.txt`` (28 base +
13 log1p + 4 flags + 6 interacciones). Romper ese contrato rompe la API v1.

Todas las funciones son puras (devuelven un DataFrame nuevo) y deterministas.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Columnas que no son features -------------------------------------------------
DROP_NON_FEATURES = [
    "nit_prestador", "tipo_contrato_codigo", "tipo_contrato",
    "regimen_codigo", "regimen", "pbs", "valor_factura_original",
]
DROP_ZERO_VARIANCE = ["num_items_pbs", "num_ambitos_distintos", "origen_factura"]
DATE_COLS = ["fecha_prestacion", "fecha_radicacion"]
ID_COL = "factura_id"
TARGET_COL = "target"

# --- Imputación ----------------------------------------------------------------
DETAIL_COLS = [
    "num_items", "num_cups_distintos", "num_dx_distintos", "sum_valor_facturado",
    "max_valor_item", "avg_valor_item", "num_items_con_autorizacion", "num_tipos_servicio",
]
NIT_COLS = [
    "total_facturas_nit", "tasa_devolucion_nit", "tasa_glosa_nit",
    "valor_promedio_nit", "dias_radicacion_promedio_nit",
]

# --- log1p (genera log_<col>) -------------------------------------------------
LOG_COLS = [
    "valor_factura", "valor_copago", "valor_cuota_moderadora", "num_items",
    "num_cups_distintos", "num_dx_distintos", "sum_valor_facturado", "max_valor_item",
    "avg_valor_item", "num_items_con_autorizacion", "total_facturas_nit",
    "valor_promedio_nit", "discrepancia_valor",
]

EXTEMPORANEA_DIAS = 22  # Res. 2284/2023, causal DE16


def drop_non_features(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina identificadores string y columnas de varianza cero (celda 3).

    Conserva ``factura_id``, ``target`` y las fechas (necesarias para el split).
    """
    cols = [c for c in DROP_NON_FEATURES + DROP_ZERO_VARIANCE if c in df.columns]
    logger.info("drop_non_features: eliminando %d columnas", len(cols))
    return df.drop(columns=cols).copy()


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa NaN: detalles → 0, históricos de NIT → mediana (celda 4)."""
    out = df.copy()
    for col in DETAIL_COLS:
        if col in out.columns:
            out[col] = out[col].fillna(0)
    for col in NIT_COLS:
        if col in out.columns and out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
    return out


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade ``log_<col> = log1p(col)`` para las 13 variables sesgadas (celda 5)."""
    out = df.copy()
    for col in LOG_COLS:
        if col in out.columns:
            out[f"log_{col}"] = np.log1p(out[col].clip(lower=0))
    return out


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Añade los 4 flags binarios (celda 5B-5D)."""
    out = df.copy()
    out["is_extemporanea"] = (out["dias_hasta_radicacion"] > EXTEMPORANEA_DIAS).astype("int8")
    out["tiene_copago"] = (out["valor_copago"] > 0).astype("int8")
    out["tiene_cuota_mod"] = (out["valor_cuota_moderadora"] > 0).astype("int8")
    out["sin_detalles"] = (out["num_items"] == 0).astype("int8")
    return out


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Añade las 6 features de interacción (celda 6)."""
    out = df.copy()
    out["ratio_valor_vs_nit"] = np.where(
        out["valor_promedio_nit"] > 0, out["valor_factura"] / out["valor_promedio_nit"], 0.0
    ).clip(0, 100)
    out["log_valor_x_tasa_glosa"] = out["log_valor_factura"] * out["tasa_glosa_nit"]
    out["dias_x_log_valor"] = out["dias_hasta_radicacion"] * out["log_valor_factura"]
    out["ratio_autorizacion"] = np.where(
        out["num_items"] > 0, out["num_items_con_autorizacion"] / out["num_items"], 0.0
    ).clip(0, 1)
    out["desviacion_dias_nit"] = np.where(
        out["dias_radicacion_promedio_nit"] > 0,
        out["dias_hasta_radicacion"] / out["dias_radicacion_promedio_nit"], 1.0,
    ).clip(0, 50)
    out["ratio_cups_items"] = np.where(
        out["num_items"] > 0, out["num_cups_distintos"] / out["num_items"], 0.0
    ).clip(0, 1)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Lista ordenada de features: todo excepto id, target y fechas (celda 8)."""
    excluded = {ID_COL, TARGET_COL, *DATE_COLS}
    return sorted(c for c in df.columns if c not in excluded)


def build_feature_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aplica el pipeline completo de FE sobre el dataset crudo.

    Args:
        df_raw: Salida de ``extract_fresh_dataset.py`` (39 columnas).

    Returns:
        DataFrame con ``factura_id``, ``target``, las 2 fechas y las 48 features.
    """
    df = drop_non_features(df_raw)
    df = impute(df)
    df = add_log_features(df)
    df = add_flags(df)
    df = add_interactions(df)
    feats = feature_columns(df)
    logger.info("build_feature_frame: %d features generadas", len(feats))
    return df[[ID_COL, TARGET_COL, *DATE_COLS, *feats]].copy()


def temporal_split(
    df: pd.DataFrame,
    val_start: str,
    test_start: str,
    test_end: str,
    oot_end: str | None = None,
    date_col: str = "fecha_radicacion",
) -> dict[str, pd.DataFrame]:
    """Divide por ``fecha_radicacion`` (nunca aleatorio — CLAUDE.md §2).

    Args:
        df: Salida de :func:`build_feature_frame`.
        val_start: Inicio de validación (train = ``[min, val_start)``).
        test_start: Inicio de test (val = ``[val_start, test_start)``).
        test_end: Fin de test = corte del horizonte confiable.
        oot_end: Fin de la ventana OOT (``[test_end, oot_end)``). Si es None, sin OOT.
        date_col: Columna de fecha para el corte.

    Returns:
        Dict con claves ``train``/``val``/``test`` (y ``oot`` si aplica).
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    ts_val, ts_test, ts_tend = map(pd.Timestamp, (val_start, test_start, test_end))

    splits = {
        "train": d[d[date_col] < ts_val],
        "val": d[(d[date_col] >= ts_val) & (d[date_col] < ts_test)],
        "test": d[(d[date_col] >= ts_test) & (d[date_col] < ts_tend)],
    }
    if oot_end is not None:
        ts_oend = pd.Timestamp(oot_end)
        splits["oot"] = d[(d[date_col] >= ts_tend) & (d[date_col] < ts_oend)]

    for name, part in splits.items():
        dist = part[TARGET_COL].value_counts().sort_index().to_dict()
        logger.info("  %-6s: %8d filas | clases %s", name, len(part), dist)
    return {k: v.reset_index(drop=True) for k, v in splits.items()}


LEAKAGE_COLS = frozenset(
    {"ejecucion_glosa_aut_ia", "estado_glosa_aut", "valor_inicial_glosa",
     "valor_pagado_factura", "estado_factura", "valor_factura_original"}
)


def assert_no_leakage(feature_cols: list[str]) -> None:
    """Lanza ``AssertionError`` si alguna feature es una columna de leakage conocida."""
    found = LEAKAGE_COLS.intersection(feature_cols)
    assert not found, f"Columnas de leakage en el vector de features: {sorted(found)}"
