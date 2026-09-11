# -*- coding: utf-8 -*-
"""Regenera la explicabilidad SHAP para el modelo campeón v3 (Fase 4).

Port del notebook ``06_SHAP_Explainability.ipynb``:
  * ``shap.TreeExplainer`` exacto sobre una muestra estratificada (~2000 facturas)
    si el campeón es de árbol (XGBoost / LightGBM / CatBoost).
  * ``permutation_importance`` como fallback para modelos no-árbol.

Salidas:
  reports/evaluations/<prefix>_shap_importance.json   (importancia global + por clase)
  reports/figures/shap_*_2026.png                      (4 figuras a 300 dpi)

Uso:
    python scripts/refresh_shap.py --prefix 20260910 \\
        --model models/artifacts/20260910_xgb_tuned_champion.json --sample 1998
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
DATA_DIR = BASE / "data" / "processed"
EVAL_DIR = BASE / "reports" / "evaluations"
FIG_DIR = BASE / "reports" / "figures"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("refresh_shap")

CLASS_NAMES = ["Auditada", "Glosada", "Devuelta"]


def _load_model(path: Path):
    """Carga XGBoost-JSON o un artefacto joblib genérico."""
    if path.suffix == ".json":
        import xgboost as xgb

        m = xgb.XGBClassifier()
        m.load_model(str(path))
        return m, "tree"
    import joblib

    m = joblib.load(path)
    kind = "tree" if type(m).__name__.lower().startswith(("xgb", "lgbm", "catboost")) else "other"
    return m, kind


def _stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    per = max(1, n // 3)
    parts = [g.sample(min(per, len(g)), random_state=seed) for _, g in df.groupby("target")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", required=True)
    p.add_argument("--model", required=True, help="Ruta al artefacto del campeón.")
    p.add_argument("--sample", type=int, default=1998)
    args = p.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    feats = [
        l.strip()
        for l in (DATA_DIR / f"{args.prefix}_feature_list.txt").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    test = pd.read_parquet(DATA_DIR / f"{args.prefix}_test.parquet")
    sample = _stratified_sample(test, args.sample)
    X = sample[feats].to_numpy(np.float32)
    y = sample["target"].to_numpy(int)
    logger.info("Muestra SHAP: %d filas (%s)", len(sample), np.bincount(y).tolist())

    model, kind = _load_model(Path(args.model))
    logger.info("Modelo %s (%s)", type(model).__name__, kind)

    global_imp: dict[str, float] = {}
    per_class: dict[str, dict[str, float]] = {}

    if kind == "tree":
        import shap

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)  # lista de 3 (n, f) o (n, f, 3)
        sv_list = sv if isinstance(sv, list) else [sv[..., k] for k in range(sv.shape[-1])]
        abs_mean_total = np.mean([np.abs(s) for s in sv_list], axis=(0, 1))
        order = np.argsort(abs_mean_total)[::-1]
        global_imp = {feats[i]: float(abs_mean_total[i]) for i in order}
        for k, cname in enumerate(CLASS_NAMES):
            am = np.abs(sv_list[k]).mean(axis=0)
            per_class[cname] = {feats[i]: float(am[i]) for i in np.argsort(am)[::-1][:15]}

        # Figura 1 — importancia global top 20
        top = list(global_imp)[:20][::-1]
        plt.figure(figsize=(9, 8))
        plt.barh(top, [global_imp[f] for f in top], color="#0066FF")
        plt.xlabel("mean(|SHAP value|)")
        plt.title("SHAP · Importancia global (v3, 2026)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "shap_01_importancia_global_2026.png", dpi=300)
        plt.close()

        # Figura 2 — beeswarm por clase
        for k, cname in enumerate(CLASS_NAMES):
            plt.figure()
            shap.summary_plot(sv_list[k], X, feature_names=feats, show=False, max_display=15)
            plt.title(f"SHAP beeswarm · {cname} (v3)")
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"shap_02_beeswarm_{cname.lower()}_2026.png", dpi=300)
            plt.close()
    else:
        from sklearn.inspection import permutation_importance

        r = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring="f1_macro")
        order = np.argsort(r.importances_mean)[::-1]
        global_imp = {feats[i]: float(r.importances_mean[i]) for i in order}
        logger.warning("Campeón no-árbol: se usa permutation_importance (no SHAP exacto).")
        top = list(global_imp)[:20][::-1]
        plt.figure(figsize=(9, 8))
        plt.barh(top, [global_imp[f] for f in top], color="#0066FF")
        plt.xlabel("Permutation importance (Δ F1-macro)")
        plt.title("Importancia por permutación (v3, 2026)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "shap_01_importancia_global_2026.png", dpi=300)
        plt.close()

    out = {
        "generated_at": datetime.now().isoformat(),
        "prefix": args.prefix,
        "model": Path(args.model).name,
        "method": "TreeExplainer" if kind == "tree" else "permutation_importance",
        "n_sample": len(sample),
        "global_importance": global_imp,
        "per_class_top": per_class,
    }
    path = EVAL_DIR / f"{args.prefix}_shap_importance.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Top-5 global: %s", list(global_imp)[:5])
    logger.info("Guardado: %s", path)


if __name__ == "__main__":
    main()
