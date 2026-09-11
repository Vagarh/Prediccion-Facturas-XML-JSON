# -*- coding: utf-8 -*-
"""Análisis del horizonte de etiquetas confiable (right-censoring).

Las facturas radicadas recientemente **no han completado el ciclo de auditoría de
glosas** (toma meses), por lo que su etiqueta "Auditada" (clase 0) es en realidad
"pendiente de auditar" — no confiable. Este script cuenta, por mes de
``fecha_radicacion``, el volumen de facturas Glosadas y Devueltas directamente en
la BD, e identifica el último mes en que el volumen de Glosada se mantiene ≥ 50%
de la mediana mensual del periodo de referencia (2024-2025).

Todo lo posterior a ese corte se descarta de train/val/test (queda como zona
censurada). Ver .claude/03_dataset_features.md §Right-Censoring.

Uso:
    python scripts/check_label_horizon.py
    python scripts/check_label_horizon.py --ref-end 2025-12-01 --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.db import check_connection, get_engine  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("check_label_horizon")


def monthly_counts(engine, start: str, end: str) -> pd.DataFrame:
    """Cuenta facturas por (mes de radicación, clase) en la BD.

    Returns:
        DataFrame con columnas ``[mes, glosada, devuelta, total_facturas]``.
    """
    logger.info("Contando Glosadas por mes (%s -> %s)...", start, end)
    q_glo = text(
        """
        SELECT DATE_FORMAT(f.fecha_radicacion, '%Y-%m-01') AS mes, COUNT(DISTINCT f.id) AS glosada
        FROM cm_facturas f
        INNER JOIN cm_detalles cd ON cd.cm_facturas_id = f.id
        INNER JOIN cm_auditoria_motivos_glosas camg ON camg.cm_detalles_id = cd.id
        WHERE f.fecha_radicacion >= :start AND f.fecha_radicacion < :end
          AND f.id NOT IN (SELECT cm_facturas_id FROM cm_auditoria_devoluciones
                           WHERE cm_facturas_id IS NOT NULL)
        GROUP BY mes ORDER BY mes
        """
    )
    df_glo = pd.read_sql(q_glo, engine, params={"start": start, "end": end})

    logger.info("Contando Devueltas por mes...")
    q_dev = text(
        """
        SELECT DATE_FORMAT(f.fecha_radicacion, '%Y-%m-01') AS mes, COUNT(DISTINCT f.id) AS devuelta
        FROM cm_facturas f
        INNER JOIN cm_auditoria_devoluciones cad ON cad.cm_facturas_id = f.id
        WHERE f.fecha_radicacion >= :start AND f.fecha_radicacion < :end
        GROUP BY mes ORDER BY mes
        """
    )
    df_dev = pd.read_sql(q_dev, engine, params={"start": start, "end": end})

    logger.info("Contando total de facturas por mes...")
    q_tot = text(
        """
        SELECT DATE_FORMAT(f.fecha_radicacion, '%Y-%m-01') AS mes, COUNT(*) AS total_facturas
        FROM cm_facturas f
        WHERE f.fecha_radicacion >= :start AND f.fecha_radicacion < :end
        GROUP BY mes ORDER BY mes
        """
    )
    df_tot = pd.read_sql(q_tot, engine, params={"start": start, "end": end})

    df = df_tot.merge(df_glo, on="mes", how="left").merge(df_dev, on="mes", how="left").fillna(0)
    for c in ("glosada", "devuelta", "total_facturas"):
        df[c] = df[c].astype(int)
    df["glosada_rate"] = (df["glosada"] / df["total_facturas"]).round(4)
    return df


def find_horizon(df: pd.DataFrame, ref_end: str, threshold: float, rate_threshold: float = 0.85) -> str:
    """Devuelve el corte (``YYYY-MM-01``, exclusivo) del horizonte de etiquetas confiable.

    Un mes es "maduro" (ciclo de auditoría completo) si cumple **ambas** señales
    respecto de la mediana del periodo de referencia (todo lo anterior a ``ref_end``):

      * ``glosada_rate`` ≥ ``rate_threshold`` * mediana(rate)  — las glosas ya se aplicaron
      * ``devuelta`` ≥ ``threshold`` * mediana(devuelta)        — las devoluciones ya se registraron

    El corte es el primer mes inmaduro tras la última racha madura (se ignora el
    mes en curso, parcial). Se devuelve ese mes como límite **exclusivo**.
    """
    ref = df[df["mes"] < ref_end]
    if ref.empty:
        raise ValueError("Periodo de referencia vacío; ajusta --ref-end.")
    med_rate = ref["glosada_rate"].median()
    med_dev = ref["devuelta"].median()
    rate_floor = rate_threshold * med_rate
    dev_floor = threshold * med_dev
    logger.info(
        "Referencia (< %s): mediana rate=%.4f (piso %.4f) | mediana devueltas=%.0f (piso %.0f)",
        ref_end, med_rate, rate_floor, med_dev, dev_floor,
    )

    d = df.sort_values("mes").reset_index(drop=True)
    d = d.iloc[:-1] if len(d) > 1 else d  # descartar mes en curso (parcial)
    d["mature"] = (d["glosada_rate"] >= rate_floor) & (d["devuelta"] >= dev_floor)
    for _, row in d.iterrows():
        flag = "OK " if row["mature"] else "CENSURADO"
        logger.info("  %s  rate=%.4f  devueltas=%-6d  %s", row["mes"], row["glosada_rate"], row["devuelta"], flag)

    # El corte es el inicio de la racha final e ininterrumpida de meses censurados
    # (los meses inmaduros aislados en el interior son ruido normal, no censura).
    mature = d["mature"].tolist()
    meses = d["mes"].tolist()
    cut_idx = len(meses)
    for i in range(len(mature) - 1, -1, -1):
        if mature[i]:
            break
        cut_idx = i
    horizon = meses[cut_idx] if cut_idx < len(meses) else meses[-1]
    logger.info("Corte de horizonte confiable (exclusivo): %s", horizon)
    return horizon


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default=None, help="default: hoy")
    p.add_argument("--ref-end", default="2025-08-01", help="Fin del periodo de referencia (pre-censura).")
    p.add_argument("--threshold", type=float, default=0.5, help="Piso de devueltas vs mediana de referencia.")
    p.add_argument("--rate-threshold", type=float, default=0.85, help="Piso de glosada_rate vs mediana.")
    p.add_argument("--from-json", default=None, help="Recalcula el horizonte desde un reporte previo (sin BD).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    end = args.end or date.today().isoformat()

    if args.from_json:
        prev = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        df = pd.DataFrame(prev["monthly"])
        logger.info("Recalculando horizonte desde %s (%d meses)", args.from_json, len(df))
    else:
        engine = get_engine()
        if not check_connection(engine):
            sys.exit(1)
        df = monthly_counts(engine, args.start, end)

    logger.info("\n%s", df.to_string(index=False))
    horizon = find_horizon(df, args.ref_end, args.threshold, args.rate_threshold)

    out = REPORTS_DIR / f"{date.today():%Y%m%d}_label_horizon.json"
    payload = {
        "generated_at": datetime.now().isoformat(),
        "window": {"start": args.start, "end": end},
        "ref_end": args.ref_end,
        "threshold": args.threshold,
        "reliable_horizon": horizon,
        "monthly": df.to_dict(orient="records"),
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Reporte guardado: %s", out)


if __name__ == "__main__":
    main()
