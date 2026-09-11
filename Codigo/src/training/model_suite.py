# -*- coding: utf-8 -*-
"""Suite comparativa de modelos multiclase — Fase 3 (modelo v3, datos frescos 2026).

Entrena y evalúa, sobre los mismos splits temporales, una batería de modelos:

    xgb_v2_repro     Params del v2 base actual (referencia, sin tuning)
    xgb_tuned        XGBoost + Optuna (búsqueda en submuestra, refit en full)
    lgbm_tuned       LightGBM + Optuna
    catboost_tuned   CatBoost + Optuna
    xgb_enriched     xgb_tuned + embeddings semánticos (3 x 384) de campos clínicos
    tabpfn           TabPFN v2 (foundation model tabular; train submuestreado)
    tabm             TabM (deep tabular, pytabkit)
    autogluon        (opcional) leído de un JSON externo generado con .venv-ag

Cada modelo se evalúa en ``test`` y en ``oot`` con
:func:`src.training.evaluation.evaluate_predictions`. El campeón se elige con
:func:`src.training.evaluation.select_champion` (Recall Glosada ≥ 0.80 en test y
≥ 0.75 en OOT; luego máximo F1-macro).

Reglas: multiclase siempre, ``RANDOM_STATE = 42``, split temporal ya fijado.
Best-effort: si una dependencia opcional falla, ese modelo se omite y se registra.

Uso:
    python -m src.training.model_suite --prefix 20260910 --optuna-trials 40
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from src.config import MODELS_DIR, REPORTS_DIR
from src.training.evaluation import CLASS_NAMES, evaluate_predictions, select_champion

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("model_suite")

RANDOM_STATE = 42
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
EVAL_DIR = Path(REPORTS_DIR) / "evaluations"

PredictFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_splits(prefix: str, enriched: bool = False) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Carga train/val/test/oot y la lista de features del prefijo dado."""
    suffix = "_enriched" if enriched else ""
    feats = [
        l.strip()
        for l in (DATA_DIR / f"{prefix}_feature_list.txt").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    splits: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test", "oot"):
        path = DATA_DIR / f"{prefix}_{name}{suffix}.parquet"
        if path.exists():
            splits[name] = pd.read_parquet(path)
            logger.info("  %-6s%-9s %8d filas", name, suffix, len(splits[name]))
        elif name != "oot":
            raise FileNotFoundError(f"Falta split obligatorio: {path}")
    return splits, feats


def xy(df: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Extrae ``(X, y)`` como ``float32`` / ``int``."""
    return df[feats].to_numpy(np.float32), df["target"].to_numpy(int)


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, n: int, seed: int = RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray]:
    """Submuestra manteniendo la proporción de clases (para acelerar Optuna/TabPFN)."""
    if len(y) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    idx: list[np.ndarray] = []
    for cls in (0, 1, 2):
        cls_idx = np.where(y == cls)[0]
        take = max(1, round(n * len(cls_idx) / len(y)))
        idx.append(rng.choice(cls_idx, size=min(take, len(cls_idx)), replace=False))
    sel = np.concatenate(idx)
    rng.shuffle(sel)
    return X[sel], y[sel]


# ---------------------------------------------------------------------------
# Trainers — cada uno devuelve un PredictFn (X -> (y_pred, y_proba))
# ---------------------------------------------------------------------------

XGB_V2_PARAMS = dict(
    objective="multi:softprob", num_class=3, max_depth=6, learning_rate=0.1,
    n_estimators=100, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
    n_jobs=-1, verbosity=0, eval_metric="mlogloss", early_stopping_rounds=10,
)


def _xgb_predict_fn(model) -> PredictFn:
    return lambda X: (model.predict(X), model.predict_proba(X))


def train_xgb_v2_repro(Xtr, ytr, Xva, yva, **_) -> PredictFn:
    import xgboost as xgb

    model = xgb.XGBClassifier(**XGB_V2_PARAMS)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    train_xgb_v2_repro.model = model  # type: ignore[attr-defined]
    return _xgb_predict_fn(model)


def _optuna_search(objective, n_trials: int, name: str) -> dict[str, Any]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("  [%s] mejor score=%.4f params=%s", name, study.best_value, study.best_params)
    return study.best_params


# Piso de Recall Glosada durante la búsqueda: por encima del objetivo 0.80 para
# dar margen a la brecha val→test/oot.
RECALL_GLOSADA_FLOOR = 0.82


def _score_val(y_true, y_pred, floor: float = RECALL_GLOSADA_FLOOR) -> float:
    """Objetivo Optuna: maximizar F1-macro sujeto a Recall Glosada ≥ ``floor``.

    Nota histórica: en v1/v2 la clase Glosada era minoritaria (13%) y se usaba
    ``sample_weight='balanced'``. En el dataset v3 Glosada es mayoritaria (66%),
    así que el balanceo hunde su recall; se sustituye por un piso duro + peso de
    clase *afinable*.
    """
    from sklearn.metrics import f1_score, recall_score

    rg = recall_score(y_true, y_pred, labels=[1], average=None, zero_division=0)[0]
    if rg < floor:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro"))


def _glosada_weights(y: np.ndarray, w_glosada: float) -> np.ndarray:
    """Vector de ``sample_weight``: clase 1 → ``w_glosada``, clases 0/2 → 1.0."""
    w = np.ones(len(y), dtype=np.float64)
    w[y == 1] = w_glosada
    return w


def train_xgb_tuned(Xtr, ytr, Xva, yva, *, n_trials: int = 40, **_) -> PredictFn:
    import xgboost as xgb

    Xs, ys = stratified_subsample(Xtr, ytr, 150_000)

    def objective(trial) -> float:
        wg = trial.suggest_float("w_glosada", 0.4, 1.6)
        params = dict(
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, n_estimators=1500,
            early_stopping_rounds=40,
            max_depth=trial.suggest_int("max_depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 12),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-3, 5.0, log=True),
        )
        m = xgb.XGBClassifier(**params)
        m.fit(Xs, ys, sample_weight=_glosada_weights(ys, wg), eval_set=[(Xva, yva)], verbose=False)
        return _score_val(yva, m.predict(Xva))

    best = _optuna_search(objective, n_trials, "xgb_tuned")
    wg = best.pop("w_glosada", 1.0)
    final = dict(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, n_estimators=2000,
        early_stopping_rounds=60, **best,
    )
    model = xgb.XGBClassifier(**final)
    model.fit(Xtr, ytr, sample_weight=_glosada_weights(ytr, wg), eval_set=[(Xva, yva)], verbose=False)
    train_xgb_tuned.best_params = {**final, "w_glosada": wg}  # type: ignore[attr-defined]
    train_xgb_tuned.model = model  # type: ignore[attr-defined]
    return _xgb_predict_fn(model)


def train_lgbm_tuned(Xtr, ytr, Xva, yva, *, n_trials: int = 40, **_) -> PredictFn:
    import lightgbm as lgb

    Xs, ys = stratified_subsample(Xtr, ytr, 150_000)

    def objective(trial) -> float:
        wg = trial.suggest_float("w_glosada", 0.4, 1.6)
        params = dict(
            objective="multiclass", num_class=3, metric="multi_logloss", verbosity=-1,
            random_state=RANDOM_STATE, n_jobs=-1, n_estimators=1500,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 120),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        )
        m = lgb.LGBMClassifier(**params)
        m.fit(Xs, ys, sample_weight=_glosada_weights(ys, wg), eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(40, verbose=False)])
        return _score_val(yva, m.predict(Xva))

    best = _optuna_search(objective, n_trials, "lgbm_tuned")
    wg = best.pop("w_glosada", 1.0)
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, metric="multi_logloss", verbosity=-1,
        random_state=RANDOM_STATE, n_jobs=-1, n_estimators=2500, **best,
    )
    model.fit(Xtr, ytr, sample_weight=_glosada_weights(ytr, wg), eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(60, verbose=False)])
    train_lgbm_tuned.best_params = {**best, "w_glosada": wg}  # type: ignore[attr-defined]
    train_lgbm_tuned.model = model  # type: ignore[attr-defined]
    return lambda X: (model.predict(X), model.predict_proba(X))


def train_catboost_tuned(Xtr, ytr, Xva, yva, *, n_trials: int = 30, **_) -> PredictFn:
    from catboost import CatBoostClassifier, Pool

    Xs, ys = stratified_subsample(Xtr, ytr, 150_000)
    pool_va = Pool(Xva, yva)

    def objective(trial) -> float:
        wg = trial.suggest_float("w_glosada", 0.4, 1.6)
        params = dict(
            loss_function="MultiClass", classes_count=3, eval_metric="TotalF1:average=Macro",
            random_seed=RANDOM_STATE, verbose=False, iterations=1500,
            class_weights=[1.0, wg, 1.0], early_stopping_rounds=40,
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            random_strength=trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        )
        m = CatBoostClassifier(**params)
        m.fit(Xs, ys, eval_set=pool_va)
        return _score_val(yva, m.predict(Xva).ravel())

    best = _optuna_search(objective, n_trials, "catboost_tuned")
    wg = best.pop("w_glosada", 1.0)
    model = CatBoostClassifier(
        loss_function="MultiClass", classes_count=3, eval_metric="TotalF1:average=Macro",
        random_seed=RANDOM_STATE, verbose=False, iterations=3000,
        class_weights=[1.0, wg, 1.0], early_stopping_rounds=60, **best,
    )
    model.fit(Xtr, ytr, eval_set=pool_va)
    train_catboost_tuned.best_params = {**best, "w_glosada": wg}  # type: ignore[attr-defined]
    train_catboost_tuned.model = model  # type: ignore[attr-defined]
    return lambda X: (model.predict(X).ravel().astype(int), model.predict_proba(X))


def train_tabpfn(Xtr, ytr, Xva, yva, *, subsample: int = 20_000, **_) -> PredictFn:
    import os

    from tabpfn import TabPFNClassifier

    Xs, ys = stratified_subsample(Xtr, ytr, subsample)
    logger.info("  [tabpfn] entrenando con %d filas (submuestra)", len(ys))
    clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True, random_state=RANDOM_STATE)
    try:
        clf.fit(Xs, ys)
    except OSError as exc:
        # tabpfn v2 exige aceptar la licencia del modelo vía login de navegador/HF.
        # No es completable en un run automático y el cliente API violaría la Ley 1581.
        raise RuntimeError(
            "TabPFN v2 requiere aceptar la licencia del modelo de forma interactiva "
            "(HF login / navegador). Alternativa local no disponible en este entorno. "
            f"Detalle: {exc}. Para habilitarlo: aceptar en huggingface.co la licencia de "
            "'Prior-Labs/TabPFN-v2-clf' y exportar HF_TOKEN antes de correr la suite."
        ) from exc
    train_tabpfn.model = clf  # type: ignore[attr-defined]
    return lambda X: (clf.predict(X).astype(int), clf.predict_proba(X))


def train_tabm(Xtr, ytr, Xva, yva, *, subsample: int = 120_000, **_) -> PredictFn:
    from pytabkit import TabM_D_Classifier

    Xs, ys = stratified_subsample(Xtr, ytr, subsample)
    Xv, yv = stratified_subsample(Xva, yva, 30_000)
    logger.info("  [tabm] entrenando con %d filas (val %d)", len(ys), len(yv))
    clf = TabM_D_Classifier(
        n_epochs=30, batch_size=2048, random_state=RANDOM_STATE, device="cpu",
        val_metric_name="cross_entropy",
    )
    clf.fit(Xs, ys, X_val=Xv, y_val=yv)
    train_tabm.model = clf  # type: ignore[attr-defined]
    return lambda X: (clf.predict(X).astype(int).ravel(), clf.predict_proba(X))


TRAINERS: dict[str, Callable[..., PredictFn]] = {
    "xgb_v2_repro": train_xgb_v2_repro,
    "xgb_tuned": train_xgb_tuned,
    "lgbm_tuned": train_lgbm_tuned,
    "catboost_tuned": train_catboost_tuned,
    "tabm": train_tabm,
    # "tabpfn": TabPFN v2 exige aceptar licencia del modelo vía login interactivo/HF y el
    # cliente API violaría la Ley 1581 — excluido del run automático (best-effort acordado).
    "tabpfn": train_tabpfn,
}

# Modelos del run por defecto (tabpfn se pide explícitamente con --models ... tabpfn).
DEFAULT_MODELS = ["xgb_v2_repro", "xgb_tuned", "lgbm_tuned", "catboost_tuned", "tabm"]


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

MODELS: dict[str, Any] = {}  # name -> estimador entrenado (para persistir el campeón)


def run_model(
    name: str, trainer: Callable[..., PredictFn],
    data: dict[str, tuple[np.ndarray, np.ndarray]], *, n_trials: int,
) -> dict[str, Any] | None:
    """Entrena un modelo y lo evalúa en test y oot. Devuelve None si falla."""
    Xtr, ytr = data["train"]
    Xva, yva = data["val"]
    for attr in ("model", "best_params"):
        if hasattr(trainer, attr):
            delattr(trainer, attr)
    t0 = time.time()
    try:
        predict = trainer(Xtr, ytr, Xva, yva, n_trials=n_trials)
    except Exception as exc:  # noqa: BLE001 - best-effort
        logger.exception("  [%s] FALLÓ: %s", name, exc)
        return {"model_name": name, "error": f"{type(exc).__name__}: {exc}"}

    if getattr(trainer, "model", None) is not None:
        MODELS[name] = trainer.model

    out: dict[str, Any] = {"model_name": name, "train_seconds": round(time.time() - t0, 1)}
    for ds in ("test", "oot"):
        if ds not in data:
            continue
        X, y = data[ds]
        y_pred, y_proba = predict(X)
        out[ds] = evaluate_predictions(y, y_pred, y_proba, model_name=name, dataset=ds)
    bp = getattr(trainer, "best_params", None)
    if bp:
        out["best_params"] = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in bp.items()}
    return out


def save_champion(name: str, model: Any, metrics: dict[str, Any], prefix: str, feats: list[str]) -> Path:
    """Persiste el modelo campeón + metadata (esquema del metadata v2)."""
    import xgboost as xgb

    art_dir = Path(MODELS_DIR) / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    enriched = name == "xgb_enriched"
    is_xgb = isinstance(model, xgb.XGBClassifier)
    ext = "json" if is_xgb else "joblib"
    model_path = art_dir / f"{prefix}_{name}_champion.{ext}"
    if is_xgb:
        model.save_model(str(model_path))
    else:
        import joblib

        joblib.dump(model, model_path)

    meta = {
        "model_type": name,
        "framework": type(model).__module__.split(".")[0],
        "is_xgboost_json": is_xgb,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_features": len(feats) + (1152 if enriched else 0),
        "feature_list": feats if not enriched else feats + ["<+ 3x384 embeddings>"],
        "n_classes": 3,
        "class_names": CLASS_NAMES,
        "random_state": RANDOM_STATE,
        "metrics": {k: metrics[k] for k in ("test", "oot") if k in metrics},
    }
    meta_path = art_dir / f"{prefix}_{name}_champion_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Campeón guardado: %s (+ %s)", model_path.name, meta_path.name)
    return model_path


def build_comparison_md(results: list[dict[str, Any]], champion: dict[str, Any]) -> str:
    """Tabla markdown de la comparación."""
    lines = [
        "# Suite de modelos v3 — comparación", "",
        "| modelo | F1-macro test | AUC test | Recall Glosada test | Recall Glosada OOT | F1 OOT | Δ F1 | seg |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['model_name']} | — ERROR: {r['error']} | | | | | | |")
            continue
        t, o = r.get("test", {}), r.get("oot", {})
        d = (t.get("f1_macro", 0) - o.get("f1_macro", 0)) if o else 0
        lines.append(
            f"| {r['model_name']} | {t.get('f1_macro', float('nan')):.4f} | "
            f"{t.get('auc_roc_ovr', float('nan')):.4f} | {t.get('recall_glosada', float('nan')):.4f} | "
            f"{(o.get('recall_glosada', float('nan')) if o else float('nan')):.4f} | "
            f"{(o.get('f1_macro', float('nan')) if o else float('nan')):.4f} | {d:+.4f} | "
            f"{(r.get('train_seconds') or 0):.0f} |"
        )
    lines += ["", f"**Campeón:** `{champion['champion']}`  ", f"Elegibles: {champion['eligibles']}", ""]
    return "\n".join(lines)


def _merge(results: list[dict[str, Any]], new: dict[str, Any]) -> None:
    """Sustituye por nombre o añade el resultado ``new`` a ``results`` (in-place)."""
    for i, r in enumerate(results):
        if r.get("model_name") == new.get("model_name"):
            results[i] = new
            return
    results.append(new)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", required=True)
    p.add_argument("--optuna-trials", type=int, default=40)
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS, choices=list(TRAINERS))
    p.add_argument("--skip-enriched", action="store_true")
    p.add_argument("--only-enriched", action="store_true", help="Solo entrena xgb_enriched y refunde.")
    p.add_argument("--autogluon-json", default=None, help="Reporte externo de AutoGluon a fusionar.")
    p.add_argument(
        "--out-json", default=None,
        help="Ruta de salida alterna (evita condiciones de carrera si se corre en paralelo "
             "con otra invocación sobre el mismo --prefix; fusionar manualmente después).",
    )
    args = p.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    splits, feats = load_splits(args.prefix)
    data = {name: xy(df, feats) for name, df in splits.items()}
    out_json = Path(args.out_json) if args.out_json else EVAL_DIR / f"{args.prefix}_model_suite.json"

    # Acumular sobre lo ya calculado en corridas previas (runs incrementales).
    results: list[dict[str, Any]] = []
    if out_json.exists():
        prev = json.loads(out_json.read_text(encoding="utf-8"))
        results = prev.get("results", [])
        logger.info("Refundiendo sobre %d resultados previos.", len(results))

    to_run = [] if args.only_enriched else list(args.models)
    logger.info("Features: %d | modelos a correr: %s", len(feats), to_run)

    for name in to_run:
        logger.info("=== %s ===", name)
        r = run_model(name, TRAINERS[name], data, n_trials=args.optuna_trials)
        if r:
            _merge(results, r)
            _dump_partial(out_json, results, args.prefix)

    if args.only_enriched or not args.skip_enriched:
        r = _run_enriched(args.prefix, args.optuna_trials, data)
        if r:
            _merge(results, r)
            _dump_partial(out_json, results, args.prefix)

    if args.autogluon_json and Path(args.autogluon_json).exists():
        ag = json.loads(Path(args.autogluon_json).read_text(encoding="utf-8"))
        _merge(results, ag)

    champ = select_champion({r["model_name"]: {k: r[k] for k in ("test", "oot") if k in r}
                             for r in results if "error" not in r})
    _dump_partial(out_json, results, args.prefix, champion=champ)
    md = Path(REPORTS_DIR) / f"{args.prefix}_model_suite_comparison.md"
    md.write_text(build_comparison_md(results, champ), encoding="utf-8")
    logger.info("Reporte: %s", out_json)
    logger.info("Comparación: %s", md)
    logger.info("CAMPEÓN: %s | elegibles: %s", champ["champion"], champ["eligibles"])

    # Persistir el mejor modelo de 48 features (el que se cablea a la API v1).
    api_ranking = [r for r in champ["ranking"] if r["eligible"] and r["model"] != "xgb_enriched"]
    api_ranking = api_ranking or [r for r in champ["ranking"] if r["model"] != "xgb_enriched"]
    if api_ranking:
        best = api_ranking[0]["model"]
        model_obj = MODELS.get(best)
        metrics = next((r for r in results if r["model_name"] == best), {})
        if model_obj is not None:
            save_champion(best, model_obj, metrics, args.prefix, feats)
        else:
            logger.warning("Sin objeto de modelo para '%s' (¿AutoGluon externo?) — no se persiste.", best)


def _run_enriched(
    prefix: str, n_trials: int, base_data, *, max_train_rows: int = 250_000,
) -> dict[str, Any] | None:
    """Entrena xgb_enriched si existen los parquets *_enriched + embeddings.

    Los embeddings de texto (MiniLM en CPU) son el cuello de botella: a las
    velocidades observadas, embeber el train completo (~842K filas x 3 campos)
    toma más de 15h. Se submuestrea train de forma estratificada a
    ``max_train_rows`` (val/test/oot, mucho más chicos, van completos) — mismo
    criterio que la submuestra usada para la búsqueda Optuna de los GBDT.
    """
    try:
        splits_e, feats_e = load_splits(prefix, enriched=True)
    except FileNotFoundError:
        logger.warning("Sin parquets *_enriched — se omite xgb_enriched.")
        return None
    try:
        from src.training.enriched import build_enriched_xy
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo preparar features enriquecidas: %s", exc)
        return None

    if max_train_rows and len(splits_e["train"]) > max_train_rows:
        df_tr = splits_e["train"]
        frac = max_train_rows / len(df_tr)
        splits_e["train"] = (
            df_tr.groupby("target", group_keys=False)
            .apply(lambda g: g.sample(frac=frac, random_state=RANDOM_STATE))
            .reset_index(drop=True)
        )
        logger.info(
            "xgb_enriched: train submuestreado de %d a %d filas (embeddings CPU son el cuello de botella)",
            len(df_tr), len(splits_e["train"]),
        )

    data_e = {}
    for name, df in splits_e.items():
        data_e[name] = build_enriched_xy(df, feats_e, prefix, name)
    return run_model("xgb_enriched", train_xgb_tuned, data_e, n_trials=n_trials)


def _dump_partial(path: Path, results: list, prefix: str, champion: dict | None = None) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prefix": prefix,
        "class_names": CLASS_NAMES,
        "random_state": RANDOM_STATE,
        "results": results,
    }
    if champion:
        payload["champion_selection"] = champion
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    main()
