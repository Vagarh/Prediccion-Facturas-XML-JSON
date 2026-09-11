# -*- coding: utf-8 -*-
"""Refit final del modelo campeón a partir de sus ``best_params`` ya conocidos.

Evita repetir la búsqueda Optuna completa cuando el objeto de modelo entrenado
se perdió (p. ej. el proceso que lo entrenó murió antes de persistirlo, pero
sus ``best_params`` ya quedaron guardados en ``<prefix>_model_suite.json``).
Reentrena una sola vez sobre el train completo con esos hiperparámetros y
persiste el artefacto + metadata con el mismo esquema que
``src.training.model_suite.save_champion``.

Uso:
    python scripts/finalize_champion.py --prefix 20260910 --model lgbm_tuned
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import warnings  # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("finalize_champion")

from src.training.model_suite import (  # noqa: E402
    _glosada_weights, load_splits, save_champion, xy,
)
from src.training.evaluation import evaluate_predictions  # noqa: E402

EVAL_DIR = BASE / "reports" / "evaluations"


def refit_xgb(Xtr, ytr, Xva, yva, params: dict):
    import xgboost as xgb

    p = dict(params)
    wg = p.pop("w_glosada", 1.0)
    final = dict(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        random_state=42, n_jobs=-1, verbosity=0, n_estimators=2000,
        early_stopping_rounds=60, **p,
    )
    m = xgb.XGBClassifier(**final)
    m.fit(Xtr, ytr, sample_weight=_glosada_weights(ytr, wg), eval_set=[(Xva, yva)], verbose=False)
    return m


def refit_lgbm(Xtr, ytr, Xva, yva, params: dict):
    import lightgbm as lgb

    p = dict(params)
    wg = p.pop("w_glosada", 1.0)
    m = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, metric="multi_logloss", verbosity=-1,
        random_state=42, n_jobs=-1, n_estimators=2500, **p,
    )
    m.fit(Xtr, ytr, sample_weight=_glosada_weights(ytr, wg), eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(60, verbose=False)])
    return m


def refit_catboost(Xtr, ytr, Xva, yva, params: dict):
    from catboost import CatBoostClassifier, Pool

    p = dict(params)
    wg = p.pop("w_glosada", 1.0)
    m = CatBoostClassifier(
        loss_function="MultiClass", classes_count=3, eval_metric="TotalF1:average=Macro",
        random_seed=42, verbose=False, iterations=3000,
        class_weights=[1.0, wg, 1.0], early_stopping_rounds=60, **p,
    )
    m.fit(Xtr, ytr, eval_set=Pool(Xva, yva))
    return m


REFITTERS = {"xgb_tuned": refit_xgb, "xgb_enriched": refit_xgb,
             "lgbm_tuned": refit_lgbm, "catboost_tuned": refit_catboost}


def predict_fn(model, X):
    import numpy as np

    proba = model.predict_proba(X)
    pred = model.predict(X)
    pred = np.asarray(pred).ravel().astype(int)
    return pred, proba


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", required=True)
    p.add_argument("--model", required=True, choices=list(REFITTERS))
    p.add_argument("--suite-json", default=None)
    args = p.parse_args()

    suite_path = Path(args.suite_json) if args.suite_json else EVAL_DIR / f"{args.prefix}_model_suite.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    entry = next((r for r in suite["results"] if r["model_name"] == args.model), None)
    if entry is None or "best_params" not in entry:
        logger.error("No hay best_params para '%s' en %s", args.model, suite_path)
        sys.exit(1)
    params = entry["best_params"]
    logger.info("Reajustando %s con params: %s", args.model, params)

    enriched = args.model == "xgb_enriched"
    splits, feats = load_splits(args.prefix, enriched=enriched)
    if enriched:
        from src.training.enriched import build_enriched_xy

        data = {name: build_enriched_xy(df, feats, args.prefix, name) for name, df in splits.items()}
    else:
        data = {name: xy(df, feats) for name, df in splits.items()}

    Xtr, ytr = data["train"]
    Xva, yva = data["val"]
    model = REFITTERS[args.model](Xtr, ytr, Xva, yva, params)

    metrics = {"model_name": args.model}
    for ds in ("test", "oot"):
        if ds not in data:
            continue
        X, y = data[ds]
        y_pred, y_proba = predict_fn(model, X)
        m = evaluate_predictions(y, y_pred, y_proba, model_name=args.model, dataset=ds)
        metrics[ds] = m
        prev = entry.get(ds, {})
        logger.info(
            "  [%s] F1=%.4f (antes %.4f) RG=%.4f (antes %.4f) — %s",
            ds, m["f1_macro"], prev.get("f1_macro", float("nan")),
            m["recall_glosada"], prev.get("recall_glosada", float("nan")),
            "OK, coincide" if abs(m["f1_macro"] - prev.get("f1_macro", -1)) < 0.01 else "difiere del reporte previo",
        )

    path = save_champion(args.model, model, metrics, args.prefix, feats)
    logger.info("Campeón persistido: %s", path)


if __name__ == "__main__":
    main()
