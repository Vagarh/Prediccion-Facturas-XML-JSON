# -*- coding: utf-8 -*-
"""Extracción del dataset de entrenamiento desde la BD de Savia — versión 2026.

Port reproducible de ``notebooks/02_construccion_dataset.ipynb`` (celdas 2-8 +
correcciones de calidad de celdas 10-11), parametrizado por ventana temporal
explícita y con credenciales desde ``Codigo/.env`` (nunca hardcodeadas).

Estrategia (idéntica a la v1/v2 para mantener comparabilidad del feature set):
  1. IDs con TARGET: 3 queries por clase, prioridad Devuelta > Glosada > Auditada.
  2. Features de ``cm_facturas``: chunks de ``WHERE id IN (...)`` (índice primario).
  3. Agregados de ``cm_detalles``: **una sola** query por rango de ``fecha_radicacion``
     (usa índice; 8× más rápido que chunks de IN — ver .claude/02_base_datos.md §Optimización).
  4. Estadísticas históricas por NIT (tasa de glosa / devolución, valor promedio).
  5. Ensamble + features derivadas + correcciones de calidad conocidas.

Regla CLAUDE.md §2: **solo SELECT**. No se toca ``data/raw/``. Salida a
``data/processed/`` con prefijo de fecha.

Uso:
    python scripts/extract_fresh_dataset.py --dry-run
    python scripts/extract_fresh_dataset.py --start 2024-03-01
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.db import check_connection, get_engine  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
LOG_DIR = BASE_DIR / "reports"

CLASS_LABELS = {0: "Auditada", 1: "Glosada", 2: "Devuelta"}

LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_file = LOG_DIR / f"extract_fresh_dataset_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(_log_file, encoding="utf-8")],
)
logger = logging.getLogger("extract_fresh_dataset")


# ---------------------------------------------------------------------------
# Paso 1 — IDs con TARGET
# ---------------------------------------------------------------------------

def extract_target_ids(
    engine: Engine, start: str, end: str, glosada_limit: int, auditada_limit: int
) -> pd.DataFrame:
    """Extrae ``factura_id`` + ``target`` para el rango de radicación dado.

    Args:
        engine: Engine SQLAlchemy.
        start: Fecha de radicación mínima inclusive (``YYYY-MM-DD``).
        end: Fecha de radicación máxima exclusiva (``YYYY-MM-DD``).
        glosada_limit: Tope de facturas Glosadas a extraer.
        auditada_limit: Tope de facturas Auditadas (muestreo ``MOD(id,20)=0``).

    Returns:
        DataFrame ``[factura_id, target]`` deduplicado (prioridad Devuelta>Glosada>Auditada).
    """
    win = {"start": start, "end": end}

    logger.info("PASO 1 — IDs con TARGET (%s -> %s)", start, end)

    q_dev = text(
        """
        SELECT DISTINCT f.id AS factura_id, 2 AS target
        FROM cm_facturas f
        INNER JOIN cm_auditoria_devoluciones cad ON cad.cm_facturas_id = f.id
        WHERE f.fecha_radicacion >= :start AND f.fecha_radicacion < :end
        """
    )
    df_dev = pd.read_sql(q_dev, engine, params=win)
    logger.info("  Devuelta (2): %d", len(df_dev))

    q_glo = text(
        f"""
        SELECT DISTINCT f.id AS factura_id, 1 AS target
        FROM cm_facturas f
        INNER JOIN cm_detalles cd ON cd.cm_facturas_id = f.id
        INNER JOIN cm_auditoria_motivos_glosas camg ON camg.cm_detalles_id = cd.id
        WHERE f.fecha_radicacion >= :start AND f.fecha_radicacion < :end
          AND f.id NOT IN (
              SELECT cm_facturas_id FROM cm_auditoria_devoluciones WHERE cm_facturas_id IS NOT NULL
          )
        LIMIT {int(glosada_limit)}
        """
    )
    df_glo = pd.read_sql(q_glo, engine, params=win)
    logger.info("  Glosada (1):  %d", len(df_glo))

    q_aud = text(
        f"""
        SELECT f.id AS factura_id, 0 AS target
        FROM cm_facturas f
        WHERE f.fecha_radicacion >= :start AND f.fecha_radicacion < :end
          AND NOT EXISTS (SELECT 1 FROM cm_auditoria_devoluciones cad
                          WHERE cad.cm_facturas_id = f.id)
          AND NOT EXISTS (SELECT 1 FROM cm_detalles cd
                          INNER JOIN cm_auditoria_motivos_glosas camg ON camg.cm_detalles_id = cd.id
                          WHERE cd.cm_facturas_id = f.id)
          AND MOD(f.id, 20) = 0
        LIMIT {int(auditada_limit)}
        """
    )
    df_aud = pd.read_sql(q_aud, engine, params=win)
    logger.info("  Auditada (0): %d", len(df_aud))

    df_ids = pd.concat([df_dev, df_glo, df_aud], ignore_index=True)
    df_ids = df_ids.drop_duplicates(subset=["factura_id"], keep="first")

    dist = df_ids["target"].value_counts().sort_index()
    logger.info("  TOTAL IDs: %d", len(df_ids))
    for cls, cnt in dist.items():
        logger.info("    %d (%s): %d (%.1f%%)", cls, CLASS_LABELS[cls], cnt, 100 * cnt / len(df_ids))
    return df_ids


# ---------------------------------------------------------------------------
# Paso 2 — Features de cm_facturas (chunks de IN sobre índice primario)
# ---------------------------------------------------------------------------

def extract_factura_features(engine: Engine, factura_ids: list[int], chunk_size: int = 50_000) -> pd.DataFrame:
    """Extrae columnas de ``cm_facturas`` disponibles al momento de radicar."""
    logger.info("PASO 2 — Features de cm_facturas (%d IDs, chunks de %d)", len(factura_ids), chunk_size)
    frames: list[pd.DataFrame] = []
    for i in range(0, len(factura_ids), chunk_size):
        chunk = factura_ids[i : i + chunk_size]
        ids_str = ",".join(str(int(x)) for x in chunk)
        q = f"""
        SELECT
            f.id                       AS factura_id,
            f.nit                      AS nit_prestador,
            f.mae_tipo_contrato_codigo AS tipo_contrato_codigo,
            f.mae_tipo_contrato_valor  AS tipo_contrato,
            f.mae_regimen_codigo       AS regimen_codigo,
            f.mae_regimen_valor        AS regimen,
            f.valor_factura,
            f.valor_copago,
            f.valor_cuota_moderadora,
            f.fecha_prestacion,
            f.fecha_radicacion,
            DATEDIFF(DATE(f.fecha_radicacion), DATE(f.fecha_prestacion)) AS dias_hasta_radicacion,
            f.numero_contrato,
            f.origen_factura,
            f.pbs,
            f.cm_fe_rips_cargas_id,
            MONTH(f.fecha_radicacion)   AS mes_radicacion,
            QUARTER(f.fecha_radicacion) AS trimestre_radicacion,
            YEAR(f.fecha_radicacion)    AS anio_radicacion
        FROM cm_facturas f
        WHERE f.id IN ({ids_str})
        """
        frames.append(pd.read_sql(text(q), engine))
        logger.info("  chunk %d: %d filas", i // chunk_size + 1, len(frames[-1]))
    out = pd.concat(frames, ignore_index=True)
    logger.info("  cm_facturas: %d filas x %d cols", len(out), out.shape[1])
    return out


# ---------------------------------------------------------------------------
# Paso 3 — Agregados de cm_detalles (una query por rango de fecha)
# ---------------------------------------------------------------------------

def extract_detalle_features(engine: Engine, start: str, end: str, id_set: set[int]) -> pd.DataFrame:
    """Agrega ``cm_detalles`` por factura en una sola query filtrada por fecha.

    ``num_dx_distintos`` se deriva del prefijo CIE-10 de ``nombre_dx``
    (``codigo_dx`` tiene NULLs masivos — ver .claude/02_base_datos.md).
    """
    logger.info("PASO 3 — Agregados de cm_detalles (query única por rango de fecha)")
    q = text(
        """
        SELECT
            cd.cm_facturas_id AS factura_id,
            COUNT(*) AS num_items,
            COUNT(DISTINCT cd.ma_servicio_codigo) AS num_cups_distintos,
            COUNT(DISTINCT NULLIF(SUBSTRING_INDEX(cd.nombre_dx, '-', 1), '')) AS num_dx_distintos,
            SUM(cd.valor_facturado) AS sum_valor_facturado,
            MAX(cd.valor_facturado) AS max_valor_item,
            AVG(cd.valor_facturado) AS avg_valor_item,
            SUM(CASE WHEN cd.aplica_pbs = 1 THEN 1 ELSE 0 END) AS num_items_pbs,
            SUM(CASE WHEN cd.aplica_autorizacion = 1 THEN 1 ELSE 0 END) AS num_items_con_autorizacion,
            COUNT(DISTINCT cd.tipo_servicio) AS num_tipos_servicio,
            COUNT(DISTINCT cd.mae_ambito_codigo) AS num_ambitos_distintos
        FROM cm_detalles cd
        INNER JOIN cm_facturas f ON f.id = cd.cm_facturas_id
        WHERE f.fecha_radicacion >= :start AND f.fecha_radicacion < :end
        GROUP BY cd.cm_facturas_id
        """
    )
    df = pd.read_sql(q, engine, params={"start": start, "end": end})
    logger.info("  cm_detalles agregado: %d facturas (periodo completo)", len(df))
    df = df[df["factura_id"].isin(id_set)].reset_index(drop=True)
    logger.info("  tras filtrar a nuestros IDs: %d", len(df))
    logger.info(
        "  num_dx_distintos -> max=%s, filas>0=%d",
        df["num_dx_distintos"].max(), int((df["num_dx_distintos"] > 0).sum()),
    )
    return df


# ---------------------------------------------------------------------------
# Paso 4 — Estadísticas históricas por NIT
# ---------------------------------------------------------------------------

def extract_provider_stats(engine: Engine, nits: list[str], months: int = 24) -> pd.DataFrame:
    """Tasa histórica de glosa/devolución y valores promedio por NIT prestador.

    Nota: replica la aproximación *global* de la v1 (ventana de 24 meses, sin
    corte por fecha de cada factura). Es una fuente conocida de leakage leve,
    documentada en .claude/03_dataset_features.md; se mantiene por comparabilidad.
    """
    logger.info("PASO 4 — Estadísticas por NIT (%d NITs)", len(nits))
    nits_str = "'" + "','".join(str(n).replace("'", "''") for n in nits) + "'"

    q_stats = text(
        f"""
        SELECT
            f.nit AS nit_prestador,
            COUNT(DISTINCT f.id) AS total_facturas_nit,
            COUNT(DISTINCT cad.cm_facturas_id) AS total_devueltas_nit,
            ROUND(COUNT(DISTINCT cad.cm_facturas_id) * 100.0
                  / NULLIF(COUNT(DISTINCT f.id), 0), 2) AS tasa_devolucion_nit,
            ROUND(AVG(f.valor_factura), 0) AS valor_promedio_nit,
            ROUND(AVG(DATEDIFF(DATE(f.fecha_radicacion), DATE(f.fecha_prestacion))), 1)
                AS dias_radicacion_promedio_nit
        FROM cm_facturas f
        LEFT JOIN cm_auditoria_devoluciones cad ON cad.cm_facturas_id = f.id
        WHERE f.fecha_radicacion >= DATE_SUB(NOW(), INTERVAL {int(months)} MONTH)
          AND f.nit IN ({nits_str})
        GROUP BY f.nit
        """
    )
    df_stats = pd.read_sql(q_stats, engine)
    logger.info("  provider stats base: %d NITs", len(df_stats))

    q_glo = text(
        f"""
        SELECT f.nit AS nit_prestador,
               COUNT(DISTINCT f.id) AS facturas_con_glosa_nit
        FROM cm_facturas f
        INNER JOIN (
            SELECT cd.cm_facturas_id
            FROM cm_detalles cd
            INNER JOIN cm_auditoria_motivos_glosas camg ON camg.cm_detalles_id = cd.id
            GROUP BY cd.cm_facturas_id
        ) g ON g.cm_facturas_id = f.id
        WHERE f.fecha_radicacion >= DATE_SUB(NOW(), INTERVAL {int(months)} MONTH)
          AND f.nit IN ({nits_str})
        GROUP BY f.nit
        """
    )
    df_glo = pd.read_sql(q_glo, engine)
    logger.info("  provider glosa counts: %d NITs", len(df_glo))

    df = df_stats.merge(df_glo, on="nit_prestador", how="left")
    df["tasa_glosa_nit"] = (
        df["facturas_con_glosa_nit"].fillna(0) * 100 / df["total_facturas_nit"].replace(0, np.nan)
    ).round(2)
    return df


# ---------------------------------------------------------------------------
# Paso 5 — Ensamble + derivadas + correcciones
# ---------------------------------------------------------------------------

def assemble_dataset(
    df_ids: pd.DataFrame,
    df_fac: pd.DataFrame,
    df_det: pd.DataFrame,
    df_prov: pd.DataFrame,
) -> pd.DataFrame:
    """Une las piezas y aplica derivadas + correcciones de calidad conocidas."""
    logger.info("PASO 5 — Ensamble + derivadas + correcciones")
    df = df_ids.merge(df_fac, on="factura_id", how="left")
    df = df.merge(df_det, on="factura_id", how="left")
    df = df.merge(
        df_prov[
            [
                "nit_prestador", "total_facturas_nit", "tasa_devolucion_nit",
                "tasa_glosa_nit", "valor_promedio_nit", "dias_radicacion_promedio_nit",
            ]
        ],
        on="nit_prestador", how="left",
    )

    # Derivadas
    df["porcentaje_mayor_item"] = (
        df["max_valor_item"] / df["valor_factura"].replace(0, np.nan)
    ).clip(upper=1.0).round(4)
    df["discrepancia_valor"] = (df["sum_valor_facturado"] - df["valor_factura"]).abs()
    df["tiene_fev"] = df["cm_fe_rips_cargas_id"].notna().astype(int)

    # Correcciones de calidad (celda 10 del notebook 02)
    df["dias_hasta_radicacion"] = df["dias_hasta_radicacion"].clip(lower=0, upper=365)
    p999 = df["valor_factura"].quantile(0.999)
    n_wins = int((df["valor_factura"] > p999).sum())
    df["valor_factura_original"] = df["valor_factura"]
    df["valor_factura"] = df["valor_factura"].clip(upper=p999)
    logger.info("  winsorize valor_factura: p99.9=%.0f, filas afectadas=%d", p999, n_wins)

    # Encodings (celda 8)
    df["tipo_contrato_encoded"] = (
        df["tipo_contrato"].map({"CAPITA": 0, "Evento": 1, "EVENTO": 1, "Paquete": 2, "PAQUETE": 2})
        .fillna(-1).astype(int)
    )
    df["regimen_encoded"] = (
        df["regimen"].map({"SUBSIDIADO": 0, "CONTRIBUTIVO": 1}).fillna(-1).astype(int)
    )

    # Quitar columnas auxiliares no-feature
    df = df.drop(columns=[c for c in ("cm_fe_rips_cargas_id", "numero_contrato") if c in df.columns])

    # Imputar numéricas restantes con 0 (facturas sin detalles = señal)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0)

    logger.info("  dataset ensamblado: %d filas x %d cols", len(df), df.shape[1])
    dist = df["target"].value_counts().sort_index()
    for cls, cnt in dist.items():
        logger.info("    %d (%s): %d (%.1f%%)", cls, CLASS_LABELS[cls], cnt, 100 * cnt / len(df))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2024-03-01", help="Fecha de radicación mínima (YYYY-MM-DD).")
    p.add_argument("--end", default=None, help="Fecha de radicación máxima exclusiva (default: hoy).")
    p.add_argument("--glosada-limit", type=int, default=800_000)
    p.add_argument("--auditada-limit", type=int, default=400_000)
    p.add_argument("--chunk-size", type=int, default=50_000)
    p.add_argument("--out", default=None, help="Ruta del parquet de salida.")
    p.add_argument("--dry-run", action="store_true", help="Solo prueba conexión + IDs con TARGET.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    end = args.end or (date.today().isoformat())
    out_path = Path(args.out) if args.out else DATA_DIR / f"{date.today():%Y%m%d}_dataset_entrenamiento.parquet"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("EXTRACCIÓN DATASET FRESCO — ventana %s -> %s", args.start, end)
    logger.info("salida: %s | dry_run=%s | log=%s", out_path, args.dry_run, _log_file)
    logger.info("=" * 70)

    engine = get_engine(connect_timeout=60)
    if not check_connection(engine):
        logger.error("Sin conexión a la BD. Abortando.")
        sys.exit(1)

    df_ids = extract_target_ids(engine, args.start, end, args.glosada_limit, args.auditada_limit)

    if args.dry_run:
        logger.info("[DRY-RUN] Fin tras Paso 1.")
        return

    factura_ids = df_ids["factura_id"].tolist()
    df_fac = extract_factura_features(engine, factura_ids, args.chunk_size)
    df_det = extract_detalle_features(engine, args.start, end, set(factura_ids))
    nits = df_fac["nit_prestador"].dropna().unique().tolist()
    df_prov = extract_provider_stats(engine, nits)

    df_final = assemble_dataset(df_ids, df_fac, df_det, df_prov)

    df_final.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    size_mb = out_path.stat().st_size / 1e6
    logger.info("GUARDADO: %s (%.1f MB, %d filas, %d cols)", out_path, size_mb, len(df_final), df_final.shape[1])
    logger.info("Columnas: %s", list(df_final.columns))


if __name__ == "__main__":
    main()
