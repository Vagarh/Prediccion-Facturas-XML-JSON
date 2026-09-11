# -*- coding: utf-8 -*-
"""Genera los splits temporales de entrenamiento a partir del dataset crudo.

Aplica ``src.features.build_dataset`` al parquet producido por
``extract_fresh_dataset.py`` y escribe:

    data/processed/<PREFIX>_train.parquet
    data/processed/<PREFIX>_val.parquet
    data/processed/<PREFIX>_test.parquet
    data/processed/<PREFIX>_oot.parquet      (ventana fresca reservada)
    data/processed/<PREFIX>_feature_list.txt

Los cortes temporales por defecto asumen un horizonte de etiquetas confiable
alrededor de feb-2026 (ver ``check_label_horizon.py``); ajústalos con flags.

Uso:
    python scripts/build_features.py --raw data/processed/20260910_dataset_entrenamiento.parquet \\
        --val-start 2025-10-01 --test-start 2025-12-01 --test-end 2026-02-01 --oot-end 2026-04-01
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.build_dataset import (  # noqa: E402
    assert_no_leakage, build_feature_frame, feature_columns, temporal_split,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
FROZEN_FEATURE_LIST = DATA_DIR / "20260309_feature_list.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("build_features")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True, help="Parquet crudo de extract_fresh_dataset.py")
    p.add_argument("--prefix", default=f"{date.today():%Y%m%d}")
    p.add_argument("--val-start", required=True)
    p.add_argument("--test-start", required=True)
    p.add_argument("--test-end", required=True, help="Corte del horizonte de etiquetas confiable.")
    p.add_argument("--oot-end", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw)
    if not raw_path.exists():
        logger.error("No existe: %s", raw_path)
        sys.exit(1)

    df_raw = pd.read_parquet(raw_path)
    logger.info("Crudo: %d filas x %d cols", len(df_raw), df_raw.shape[1])

    df_fe = build_feature_frame(df_raw)
    feats = feature_columns(df_fe)
    assert_no_leakage(feats)
    logger.info("Features generadas: %d", len(feats))

    # Comparar contra el contrato congelado de 48 features
    if FROZEN_FEATURE_LIST.exists():
        frozen = [l.strip() for l in FROZEN_FEATURE_LIST.read_text(encoding="utf-8").splitlines() if l.strip()]
        missing, extra = set(frozen) - set(feats), set(feats) - set(frozen)
        if missing or extra:
            logger.error("DESAJUSTE con el contrato de 48 features:")
            logger.error("  faltan: %s", sorted(missing))
            logger.error("  sobran: %s", sorted(extra))
            sys.exit(2)
        logger.info("Contrato de 48 features: OK (coincide con %s)", FROZEN_FEATURE_LIST.name)

    splits = temporal_split(
        df_fe,
        val_start=args.val_start,
        test_start=args.test_start,
        test_end=args.test_end,
        oot_end=args.oot_end,
    )

    save_cols = ["factura_id", "target", *feats]
    for name, part in splits.items():
        out = DATA_DIR / f"{args.prefix}_{name}.parquet"
        part[save_cols].to_parquet(out, index=False, compression="snappy")
        logger.info("  guardado %s (%.1f MB, %d filas)", out.name, out.stat().st_size / 1e6, len(part))

    flist = DATA_DIR / f"{args.prefix}_feature_list.txt"
    flist.write_text("\n".join(feats), encoding="utf-8")
    logger.info("  guardado %s (%d features)", flist.name, len(feats))


if __name__ == "__main__":
    main()
