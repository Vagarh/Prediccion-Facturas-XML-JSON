# -*- coding: utf-8 -*-
"""Runner aislado de AutoGluon Tabular (Fase 3).

AutoGluon degrada pandas/pyarrow y arrastra ray/boto, así que corre en su propio
entorno ``.venv-ag`` y escribe un JSON compatible con el esquema de
``src.training.model_suite`` para que la suite lo fusione.

Uso (con el intérprete de .venv-ag):
    .venv-ag/Scripts/python.exe -m src.training.run_autogluon --prefix 20260910 \\
        --preset best_quality --time-limit 21600
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))
DATA_DIR = BASE / "data" / "processed"
EVAL_DIR = BASE / "reports" / "evaluations"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("run_autogluon")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", required=True)
    p.add_argument("--preset", default="best_quality")
    p.add_argument("--time-limit", type=int, default=21_600)
    p.add_argument("--max-train-rows", type=int, default=300_000,
                   help="Submuestreo estratificado del train para acotar el costo de AutoGluon.")
    args = p.parse_args()

    from autogluon.tabular import TabularPredictor

    from src.training.evaluation import evaluate_predictions  # sklearn-only, seguro en .venv-ag

    feats = [
        l.strip()
        for l in (DATA_DIR / f"{args.prefix}_feature_list.txt").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    train = pd.read_parquet(DATA_DIR / f"{args.prefix}_train.parquet")[feats + ["target"]]
    if args.max_train_rows and len(train) > args.max_train_rows:
        frac = args.max_train_rows / len(train)
        train = (
            train.groupby("target", group_keys=False)
            .apply(lambda g: g.sample(frac=frac, random_state=42))
            .reset_index(drop=True)
        )
        logger.info("train submuestreado a %d filas (estratificado)", len(train))
    val = pd.read_parquet(DATA_DIR / f"{args.prefix}_val.parquet")[feats + ["target"]]
    test = pd.read_parquet(DATA_DIR / f"{args.prefix}_test.parquet")[feats + ["target"]]
    oot_path = DATA_DIR / f"{args.prefix}_oot.parquet"
    oot = pd.read_parquet(oot_path)[feats + ["target"]] if oot_path.exists() else None

    out_dir = BASE / "models" / "artifacts" / f"autogluon_{args.prefix}"
    t0 = time.time()
    predictor = TabularPredictor(
        label="target", eval_metric="f1_macro", path=str(out_dir), verbosity=2,
    ).fit(
        train_data=train, tuning_data=val, use_bag_holdout=True,
        presets=args.preset, time_limit=args.time_limit,
    )
    train_seconds = round(time.time() - t0, 1)

    result: dict = {"model_name": "autogluon", "train_seconds": train_seconds,
                    "preset": args.preset, "leaderboard": predictor.leaderboard(silent=True).to_dict("records")}
    for name, df in (("test", test), ("oot", oot)):
        if df is None:
            continue
        proba = predictor.predict_proba(df[feats])
        proba = proba.reindex(columns=[0, 1, 2], fill_value=0.0).to_numpy()
        pred = proba.argmax(axis=1)
        result[name] = evaluate_predictions(df["target"].to_numpy(int), pred, proba,
                                            model_name="autogluon", dataset=name)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / f"{args.prefix}_autogluon.json"
    out.write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).isoformat(), **result}, indent=2, ensure_ascii=False,
    ), encoding="utf-8")
    logger.info("Guardado: %s", out)


if __name__ == "__main__":
    main()
