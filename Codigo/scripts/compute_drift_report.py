# -*- coding: utf-8 -*-
"""Reporte de concept drift entre el dataset viejo (v2) y el fresco (v3).

Compara la distribución de las 48 features entre los splits de entrenamiento
``20260309_train.parquet`` (referencia) y ``<prefix>_train.parquet`` (actual)
usando ``src.monitoring.drift_detector.DriftDetector`` (PSI).

El resultado alimenta el panel "Concept Drift" del dashboard (``web-next``) con
fechas y valores PSI reales, y sirve de evidencia empírica del drift 2025 para
la tesis.

Uso:
    python scripts/compute_drift_report.py --prefix 20260910
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
DATA_DIR = BASE / "data" / "processed"
REPORTS_DIR = BASE / "reports"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("compute_drift_report")

REF_PREFIX = "20260309"


def _load(prefix: str, split: str, feats: list[str]) -> np.ndarray:
    df = pd.read_parquet(DATA_DIR / f"{prefix}_{split}.parquet")
    return df[feats].to_numpy(np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", required=True)
    p.add_argument("--ref-prefix", default=REF_PREFIX)
    args = p.parse_args()

    from src.monitoring.drift_detector import DriftDetector

    feats = [
        l.strip()
        for l in (DATA_DIR / f"{args.ref_prefix}_feature_list.txt").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    ref = _load(args.ref_prefix, "train", feats)
    cur = _load(args.prefix, "train", feats)
    logger.info("ref=%s (%d filas) | cur=%s (%d filas) | %d features",
                args.ref_prefix, len(ref), args.prefix, len(cur), len(feats))

    detector = DriftDetector()
    result = detector.detect_feature_drift(ref, cur, feature_names=feats)

    out = {
        "generated_at": datetime.now().isoformat(),
        "reference_prefix": args.ref_prefix,
        "current_prefix": args.prefix,
        "n_features": len(feats),
        "overall_status": result["overall_status"],
        "mean_psi": result["mean_psi"],
        "max_psi": result["max_psi"],
        "drift_counts": result["drift_counts"],
        "top_drift_features": result["top_drift_features"],
        "psi_per_feature": result["psi_per_feature"],
    }
    path = REPORTS_DIR / f"{date.today():%Y%m%d}_drift_report.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("overall=%s | mean_psi=%.4f | max_psi=%.4f | alerts=%d",
                out["overall_status"], out["mean_psi"], out["max_psi"], out["drift_counts"]["alert"])
    for f in out["top_drift_features"]:
        logger.info("  %-28s PSI=%.4f  %s", f["name"], f["psi"], f["status"])
    logger.info("Reporte: %s", path)


if __name__ == "__main__":
    main()
