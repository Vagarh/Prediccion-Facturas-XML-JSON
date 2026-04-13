# -*- coding: utf-8 -*-
"""Extracción de campos textuales desde la BD para enriquecimiento semántico.

Lee los factura_id de los parquets train/val/test YA EXISTENTES y recupera
de la BD los campos textuales clínicos asociados a cada factura:
  - diagnostico_texto : diagnósticos CIE-10 concatenados (nombre_dx)
  - cups_texto        : nombres de servicios CUPS concatenados (ma_servicio_valor)
  - observaciones     : observaciones del auditor en glosas (cm_auditoria_motivos_glosas.observacion)

Regla crítica: SOLO SELECT. Nunca INSERT/UPDATE/DELETE/DROP.
Los parquets originales NO se sobreescriben; se genera un parquet nuevo
con sufijo '_enriched' en data/processed/.

Uso:
    py -3 scripts/extract_text_fields.py [--dry-run] [--chunk-size 5000]

Requisitos:
    pip install sqlalchemy pymysql pandas pyarrow
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
LOG_DIR = BASE_DIR / "reports"

# Credenciales BD (igual que notebooks)
DB_CONFIG: dict[str, Any] = {
    "user": "jcardonr",
    "password": "6nxx2yVWaKXyiF1cZbfS*",
    "host": "10.250.5.36",
    "port": "3306",
    "db_name": "system_savia",
}

# Parquets de entrada
SPLITS = {
    "train": DATA_DIR / "20260309_train.parquet",
    "val":   DATA_DIR / "20260309_val.parquet",
    "test":  DATA_DIR / "20260309_test.parquet",
}

# Parquets de salida (NO sobreescriben el original)
OUTPUT_SPLITS = {
    split: DATA_DIR / f"20260309_{split}_enriched.parquet"
    for split in SPLITS
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"extract_text_fields_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query de extracción de campos textuales
# ---------------------------------------------------------------------------

# Estrategia:
#   - diagnostico_texto : todos los nombre_dx del detalle, separados por " | "
#     (nombre_dx tiene formato "K083-Descripción" — se usa el campo completo
#      ya que incluye código + descripción, útil para embeddings clínicos)
#   - cups_texto        : todos los ma_servicio_valor del detalle, separados por " | "
#   - observaciones     : texto libre del auditor en motivos de glosa (si los hay),
#     separados por " | ". Para facturas Auditadas/Devueltas será NULL → cadena vacía.
#
# Uso de GROUP_CONCAT con ORDER BY para reproducibilidad.
# DISTINCT para evitar repetir el mismo dx/cups si aparece en múltiples ítems.
# Se filtran NULLs y cadenas vacías antes de concatenar.

TEXT_FIELDS_QUERY = text("""
SELECT
    f.id AS factura_id,

    /* Diagnósticos CIE-10: concatenación de nombre_dx de los detalles */
    GROUP_CONCAT(
        DISTINCT NULLIF(TRIM(cd.nombre_dx), '')
        ORDER BY cd.nombre_dx
        SEPARATOR ' | '
    ) AS diagnostico_texto,

    /* Servicios CUPS: concatenación de ma_servicio_valor de los detalles */
    GROUP_CONCAT(
        DISTINCT NULLIF(TRIM(cd.ma_servicio_valor), '')
        ORDER BY cd.ma_servicio_valor
        SEPARATOR ' | '
    ) AS cups_texto,

    /* Observaciones del auditor en motivos de glosa */
    GROUP_CONCAT(
        DISTINCT NULLIF(TRIM(camg.observacion), '')
        ORDER BY camg.observacion
        SEPARATOR ' | '
    ) AS observaciones

FROM cm_facturas f

/* JOIN a detalles para diagnósticos y CUPS */
LEFT JOIN cm_detalles cd
    ON cd.cm_facturas_id = f.id
   AND cd.nombre_dx     IS NOT NULL
   AND cd.nombre_dx      != ''

/* JOIN a motivos de glosa para observaciones del auditor */
LEFT JOIN cm_auditoria_motivos_glosas camg
    ON camg.cm_detalles_id = cd.id
   AND camg.observacion   IS NOT NULL
   AND camg.observacion    != ''

WHERE f.id IN :factura_ids

GROUP BY f.id
""")


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------


def build_engine():
    """Crea el motor SQLAlchemy con las credenciales configuradas.

    Returns:
        Engine de SQLAlchemy listo para usar.
    """
    cfg = DB_CONFIG
    url = f"mysql+pymysql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['db_name']}"
    engine = create_engine(
        url,
        connect_args={"connect_timeout": 60},
        pool_pre_ping=True,
    )
    logger.info("Motor SQLAlchemy creado: %s:%s/%s", cfg["host"], cfg["port"], cfg["db_name"])
    return engine


def test_connection(engine) -> bool:
    """Verifica que la conexión a la BD está activa.

    Args:
        engine: Motor SQLAlchemy.

    Returns:
        True si la conexión es exitosa.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Conexion a BD verificada correctamente.")
        return True
    except Exception as exc:
        logger.error("Fallo en la conexion a BD: %s", exc)
        return False


def extract_text_fields_chunk(
    engine,
    factura_ids: list[int],
) -> pd.DataFrame:
    """Extrae campos textuales para un chunk de factura_ids.

    Args:
        engine: Motor SQLAlchemy.
        factura_ids: Lista de IDs de facturas a consultar.

    Returns:
        DataFrame con columnas [factura_id, diagnostico_texto, cups_texto, observaciones].
    """
    # SQLAlchemy / MySQL requiere tuple para IN con bindparam
    ids_tuple = tuple(int(i) for i in factura_ids)

    with engine.connect() as conn:
        result = conn.execute(TEXT_FIELDS_QUERY, {"factura_ids": ids_tuple})
        rows = result.fetchall()

    df = pd.DataFrame(rows, columns=["factura_id", "diagnostico_texto", "cups_texto", "observaciones"])
    # Rellenar NULLs con cadena vacía (el EmbeddingService maneja campos vacíos → zero_vector)
    df[["diagnostico_texto", "cups_texto", "observaciones"]] = (
        df[["diagnostico_texto", "cups_texto", "observaciones"]].fillna("")
    )
    return df


def extract_all_text_fields(
    engine,
    factura_ids: list[int],
    chunk_size: int = 5000,
) -> pd.DataFrame:
    """Extrae campos textuales para todos los factura_ids en chunks.

    Divide los IDs en lotes de `chunk_size` para evitar timeouts y el límite
    de IN clauses en MySQL. Usa JOIN por `f.id IN (...)` que es eficiente
    con índice primario.

    Args:
        engine: Motor SQLAlchemy.
        factura_ids: Lista completa de IDs a procesar.
        chunk_size: Tamaño del lote por query (recomendado ≤5000).

    Returns:
        DataFrame consolidado con todos los campos textuales.
    """
    total = len(factura_ids)
    logger.info("Extrayendo campos textuales para %d facturas en chunks de %d...", total, chunk_size)

    frames: list[pd.DataFrame] = []
    n_chunks = (total + chunk_size - 1) // chunk_size

    for i in range(n_chunks):
        chunk = factura_ids[i * chunk_size : (i + 1) * chunk_size]
        logger.info("  Chunk %d/%d — %d IDs...", i + 1, n_chunks, len(chunk))

        try:
            df_chunk = extract_text_fields_chunk(engine, chunk)
            frames.append(df_chunk)
            logger.info(
                "  Chunk %d/%d completado: %d filas recuperadas.",
                i + 1, n_chunks, len(df_chunk),
            )
        except Exception as exc:
            logger.error("  Error en chunk %d/%d: %s — reintentando...", i + 1, n_chunks, exc)
            # Reintento único
            try:
                df_chunk = extract_text_fields_chunk(engine, chunk)
                frames.append(df_chunk)
                logger.info("  Chunk %d/%d recuperado en reintento.", i + 1, n_chunks)
            except Exception as exc2:
                logger.error("  Fallo definitivo en chunk %d/%d: %s", i + 1, n_chunks, exc2)
                # Insertar chunk de ceros para no perder el split
                empty = pd.DataFrame({
                    "factura_id": chunk,
                    "diagnostico_texto": [""] * len(chunk),
                    "cups_texto": [""] * len(chunk),
                    "observaciones": [""] * len(chunk),
                })
                frames.append(empty)

    all_text = pd.concat(frames, ignore_index=True)
    logger.info("Extracción completa: %d filas totales.", len(all_text))
    return all_text


def enrich_parquet(
    split_name: str,
    input_path: Path,
    output_path: Path,
    engine,
    chunk_size: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Enriquece un parquet con campos textuales y guarda el resultado.

    Args:
        split_name: Nombre del split ('train', 'val', 'test').
        input_path: Ruta al parquet original.
        output_path: Ruta al parquet de salida enriquecido.
        engine: Motor SQLAlchemy.
        chunk_size: Tamaño del chunk para extracción.
        dry_run: Si True, solo valida sin guardar.

    Returns:
        Diccionario con estadísticas del proceso.
    """
    logger.info("=" * 60)
    logger.info("Procesando split: %s", split_name.upper())
    logger.info("  Entrada:  %s", input_path)
    logger.info("  Salida:   %s", output_path)

    # Leer parquet original
    df_orig = pd.read_parquet(input_path)
    logger.info("  Filas leídas: %d | Columnas: %d", len(df_orig), len(df_orig.columns))

    # Verificar que no tenga ya los campos textuales
    existing_text_cols = [c for c in df_orig.columns if c in ["diagnostico_texto", "cups_texto", "observaciones"]]
    if existing_text_cols:
        logger.warning(
            "  El parquet ya tiene columnas textuales: %s — se sobreescriben con datos frescos de BD.",
            existing_text_cols,
        )

    # Extraer factura_ids
    if "factura_id" not in df_orig.columns:
        raise KeyError(f"El parquet {input_path} no tiene columna 'factura_id'.")

    factura_ids = df_orig["factura_id"].tolist()
    logger.info("  factura_ids a consultar: %d únicos", len(set(factura_ids)))

    if dry_run:
        logger.info("  [DRY-RUN] Extracción omitida. Saliendo.")
        return {"split": split_name, "n_rows": len(df_orig), "dry_run": True}

    # Extraer campos textuales desde BD
    df_text = extract_all_text_fields(engine, factura_ids, chunk_size=chunk_size)

    # Estadísticas de cobertura textual
    n_with_dx    = (df_text["diagnostico_texto"] != "").sum()
    n_with_cups  = (df_text["cups_texto"] != "").sum()
    n_with_obs   = (df_text["observaciones"] != "").sum()
    logger.info(
        "  Cobertura textual — diagnósticos: %d/%d (%.1f%%) | CUPS: %d/%d (%.1f%%) | observaciones: %d/%d (%.1f%%)",
        n_with_dx,    len(df_text), 100 * n_with_dx    / max(len(df_text), 1),
        n_with_cups,  len(df_text), 100 * n_with_cups  / max(len(df_text), 1),
        n_with_obs,   len(df_text), 100 * n_with_obs   / max(len(df_text), 1),
    )

    # Hacer JOIN sobre factura_id para añadir columnas textuales al parquet original
    # LEFT JOIN: facturas sin texto en BD quedan con cadena vacía
    if existing_text_cols:
        df_orig = df_orig.drop(columns=existing_text_cols)

    df_enriched = df_orig.merge(df_text, on="factura_id", how="left")
    df_enriched[["diagnostico_texto", "cups_texto", "observaciones"]] = (
        df_enriched[["diagnostico_texto", "cups_texto", "observaciones"]].fillna("")
    )

    logger.info(
        "  Parquet enriquecido: %d filas × %d columnas",
        len(df_enriched), len(df_enriched.columns),
    )

    # Guardar parquet enriquecido (NO sobreescribe el original)
    df_enriched.to_parquet(output_path, index=False, compression="snappy")
    logger.info("  Guardado: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)

    return {
        "split": split_name,
        "n_rows": len(df_enriched),
        "n_with_diagnostico": int(n_with_dx),
        "n_with_cups": int(n_with_cups),
        "n_with_observaciones": int(n_with_obs),
        "pct_diagnostico": round(100 * n_with_dx / max(len(df_text), 1), 2),
        "pct_cups": round(100 * n_with_cups / max(len(df_text), 1), 2),
        "pct_observaciones": round(100 * n_with_obs / max(len(df_text), 1), 2),
        "output_path": str(output_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Extrae campos textuales de la BD y enriquece los parquets train/val/test."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo verifica conexión y cuenta IDs, sin ejecutar queries ni guardar archivos.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Número de factura_ids por query a la BD (default: 5000).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="Splits a procesar (default: todos).",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada principal."""
    args = parse_args()

    logger.info("=" * 70)
    logger.info("EXTRACCIÓN DE CAMPOS TEXTUALES — Savia Salud EPS")
    logger.info("Inicio: %s", datetime.now().isoformat())
    logger.info("dry_run=%s | chunk_size=%d | splits=%s", args.dry_run, args.chunk_size, args.splits)
    logger.info("Log: %s", log_file)
    logger.info("=" * 70)

    # Construir engine y verificar conexión
    engine = build_engine()
    if not test_connection(engine):
        logger.error("No se pudo conectar a la BD. Abortando.")
        sys.exit(1)

    # Verificar que los parquets de entrada existen
    for split in args.splits:
        path = SPLITS[split]
        if not path.exists():
            logger.error("Parquet de entrada no encontrado: %s", path)
            sys.exit(1)
        logger.info("Parquet verificado: %s (%.1f MB)", path.name, path.stat().st_size / 1e6)

    # Procesar cada split
    results: list[dict] = []
    for split in args.splits:
        try:
            stats = enrich_parquet(
                split_name=split,
                input_path=SPLITS[split],
                output_path=OUTPUT_SPLITS[split],
                engine=engine,
                chunk_size=args.chunk_size,
                dry_run=args.dry_run,
            )
            results.append(stats)
        except Exception as exc:
            logger.exception("Error procesando split '%s': %s", split, exc)
            sys.exit(1)

    # Resumen final
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESUMEN FINAL")
    logger.info("=" * 70)
    for r in results:
        if r.get("dry_run"):
            logger.info("  [%s] DRY-RUN — %d filas (sin extracción)", r["split"].upper(), r["n_rows"])
        else:
            logger.info(
                "  [%s] %d filas | dx=%.1f%% | cups=%.1f%% | obs=%.1f%% -> %s",
                r["split"].upper(),
                r["n_rows"],
                r.get("pct_diagnostico", 0),
                r.get("pct_cups", 0),
                r.get("pct_observaciones", 0),
                Path(r.get("output_path", "")).name,
            )

    if not args.dry_run:
        logger.info("")
        logger.info("Parquets enriquecidos guardados. Próximo paso:")
        logger.info("  py -3 src/evaluation/retrain_xgboost.py")

    logger.info("Fin: %s", datetime.now().isoformat())


if __name__ == "__main__":
    main()
