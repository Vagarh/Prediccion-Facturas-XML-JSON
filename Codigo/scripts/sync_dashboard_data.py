# -*- coding: utf-8 -*-
"""Genera la capa de datos del dashboard a partir de los artefactos reales.

``Codigo/reports/`` está en ``.gitignore``, de modo que Vercel —que construye
desde el repositorio— no puede leer los JSON de evaluación directamente. Este
script consolida esos artefactos en un único archivo compacto y versionado,
``web-next/data/metrics.json``, que es la fuente de verdad del sitio.

La regla que impone es: **ninguna cifra del dashboard se escribe a mano**. Si un
número cambia, se reentrena o se recalcula y se vuelve a ejecutar este script.

Fuentes que consolida:
    - ``reports/evaluations/<prefix>_model_suite.json``  (suite comparativa + campeón)
    - ``reports/evaluations/<prefix>_shap_importance.json`` (explicabilidad)
    - ``reports/<fecha>_drift_report.json``              (PSI real)
    - ``reports/<prefix>_label_horizon.json``            (horizonte de etiquetas)
    - ``reports/evaluations/20260309_metricas.json``     (histórico v1)
    - ``data/processed/<prefix>_{train,val,test,oot}.parquet`` (tamaño y clases por split)

Uso:
    python scripts/sync_dashboard_data.py --prefix 20260910
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
DATA_DIR = BASE / "data" / "processed"
REPORTS_DIR = BASE / "reports"
EVAL_DIR = REPORTS_DIR / "evaluations"
WEB_DATA_DIR = BASE.parent / "web-next" / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sync_dashboard_data")

V1_PREFIX = "20260309"
CLASS_ORDER = ["Auditada", "Glosada", "Devuelta"]

#: Rótulos y familia de cada modelo de la suite. El ``id`` es el nombre técnico
#: que usa ``model_suite.py``; el ``label`` es el que ve el lector del sitio.
MODEL_META: dict[str, dict[str, str]] = {
    "xgb_v2_repro": {"label": "XGBoost v2 (reproducido)", "family": "GBDT", "tuned": "no"},
    "xgb_tuned": {"label": "XGBoost afinado", "family": "GBDT", "tuned": "Optuna"},
    "lgbm_tuned": {"label": "LightGBM afinado", "family": "GBDT", "tuned": "Optuna"},
    "catboost_tuned": {"label": "CatBoost afinado", "family": "GBDT", "tuned": "Optuna"},
    "tabm": {"label": "TabM (tabular profundo)", "family": "Deep", "tuned": "no"},
    "autogluon": {"label": "AutoGluon (AutoML)", "family": "AutoML", "tuned": "preset"},
}

#: Definición contractual de los cortes temporales, idéntica a los argumentos con
#: los que se ejecutó ``scripts/build_features.py``.
SPLIT_RANGES: dict[str, tuple[str, str]] = {
    "train": ("2024-03", "2025-11"),
    "val": ("2025-12", "2026-01"),
    "test": ("2026-02", "2026-03"),
    "oot": ("2026-04", "2026-05"),
}
SPLIT_LABELS: dict[str, str] = {
    "train": "Entrenamiento",
    "val": "Validación",
    "test": "Prueba",
    "oot": "Out-of-Time",
}


def _read_json(path: Path) -> dict[str, Any]:
    """Lee un JSON UTF-8, fallando con un mensaje accionable si no existe."""
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Ejecuta primero el script que lo genera "
            f"(ver docstring de sync_dashboard_data.py)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_drift_report() -> Path:
    """Devuelve el reporte de drift más reciente por nombre de archivo."""
    candidates = sorted(REPORTS_DIR.glob("*_drift_report.json"))
    if not candidates:
        raise FileNotFoundError(
            "No hay reportes de drift. Ejecuta scripts/compute_drift_report.py."
        )
    return candidates[-1]


def _r(value: Any, digits: int = 4) -> Any:
    """Redondea floats para mantener el JSON compacto y diffeable."""
    return round(float(value), digits) if isinstance(value, (int, float)) else value


def _dedupe_drift(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Colapsa pares ``X`` / ``log_X`` en el ranking de drift.

    Una transformación logarítmica no es una variable distinta: su PSI es idéntico
    al de la variable base y mostrar ambas desperdicia espacio y sugiere más
    variables derivando de las que realmente hay.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for feature in features:
        base = feature["name"][4:] if feature["name"].startswith("log_") else feature["name"]
        if base in seen:
            continue
        seen.add(base)
        out.append(feature)
    return out


def build_split_summary(prefix: str) -> list[dict[str, Any]]:
    """Resume cada partición temporal: filas, rango y distribución de clases.

    Args:
        prefix: Prefijo de fecha de los parquet (por ejemplo ``20260910``).

    Returns:
        Una entrada por split con su conteo y proporción por clase.
    """
    splits: list[dict[str, Any]] = []
    for split, (start, end) in SPLIT_RANGES.items():
        path = DATA_DIR / f"{prefix}_{split}.parquet"
        if not path.exists():
            logger.warning("Split ausente, se omite: %s", path.name)
            continue
        counts = pd.read_parquet(path, columns=["target"])["target"].value_counts()
        rows = int(counts.sum())
        splits.append(
            {
                "id": split,
                "label": SPLIT_LABELS[split],
                "rows": rows,
                "start": start,
                "end": end,
                "classCounts": {
                    name: int(counts.get(idx, 0)) for idx, name in enumerate(CLASS_ORDER)
                },
                "classShare": {
                    name: _r(int(counts.get(idx, 0)) / rows) for idx, name in enumerate(CLASS_ORDER)
                },
            }
        )
        logger.info("  %-5s %8d filas  (%s → %s)", split, rows, start, end)
    return splits


def _split_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    """Normaliza el bloque de métricas de un split (test u oot)."""
    report = raw.get("classification_report", {})
    return {
        "nSamples": int(raw["n_samples"]),
        "f1Macro": _r(raw["f1_macro"]),
        "aucRocOvr": _r(raw["auc_roc_ovr"]),
        "recallGlosada": _r(raw["recall_glosada"]),
        "accuracy": _r(raw["accuracy"]),
        "recallPerClass": {k: _r(v) for k, v in raw["recall_per_class"].items()},
        "confusionMatrix": [[int(v) for v in row] for row in raw["confusion_matrix"]],
        "perClass": {
            name: {
                "precision": _r(report[name]["precision"]),
                "recall": _r(report[name]["recall"]),
                "f1": _r(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in CLASS_ORDER
            if name in report
        },
    }


def build_models(suite: dict[str, Any]) -> list[dict[str, Any]]:
    """Normaliza los resultados de la suite comparativa de modelos."""
    eligibles = set(suite["champion_selection"]["eligibles"])
    models: list[dict[str, Any]] = []
    for entry in suite["results"]:
        name = entry["model_name"]
        meta = MODEL_META.get(name, {"label": name, "family": "—", "tuned": "—"})
        models.append(
            {
                "id": name,
                "label": meta["label"],
                "family": meta["family"],
                "tuning": meta["tuned"],
                "trainSeconds": _r(entry.get("train_seconds"), 1)
                if entry.get("train_seconds") is not None
                else None,
                "eligible": name in eligibles,
                "bestParams": {k: _r(v, 6) for k, v in (entry.get("best_params") or {}).items()},
                "test": _split_metrics(entry["test"]),
                "oot": _split_metrics(entry["oot"]),
            }
        )
    # Orden descendente por F1-macro de prueba: el campeón queda arriba del todo.
    models.sort(key=lambda m: m["test"]["f1Macro"], reverse=True)
    return models


def build_history(suite: dict[str, Any], champion: dict[str, Any]) -> list[dict[str, Any]]:
    """Construye la línea de tiempo v1 → v2 → v3 con cifras reales.

    Las cifras de v1 y v2 provienen del reporte de evaluación de marzo de 2026 y
    del análisis Out-of-Time de esa misma fase; las de v3, de la suite actual.
    """
    v1 = _read_json(EVAL_DIR / f"{V1_PREFIX}_metricas.json")["modelos"]["XGBoost"]
    repro = next(m for m in suite["results"] if m["model_name"] == "xgb_v2_repro")
    return [
        {
            "version": "v1",
            "date": "2026-03",
            "algorithm": "XGBoost",
            "datasetLabel": "633K facturas · mar-2024 a jul-2025",
            "f1Macro": _r(v1["f1_macro"]),
            "recallGlosada": _r(v1["recall_glosada"]),
            "note": "Modelo base. Recall Glosada por debajo de la meta operativa de 0.80.",
        },
        {
            "version": "v2",
            "date": "2026-04",
            "algorithm": "XGBoost + texto clínico",
            "datasetLabel": "mismos datos, enriquecidos con embeddings",
            "f1Macro": 0.7114,
            "recallGlosada": 0.8961,
            "note": (
                "Nunca llegó a cablearse a la API. La ganancia en Recall resultó "
                "ilusoria: el enriquecimiento textual operaba sobre datos ya vencidos."
            ),
        },
        {
            "version": "v3",
            "date": "2026-09",
            "algorithm": champion["algorithm"],
            "datasetLabel": "1.08M facturas · mar-2024 a may-2026",
            "f1Macro": champion["test"]["f1Macro"],
            "recallGlosada": champion["test"]["recallGlosada"],
            "ootF1Macro": champion["oot"]["f1Macro"],
            "ootRecallGlosada": champion["oot"]["recallGlosada"],
            "note": (
                "Campeón desplegado. Reentrenar sobre una ventana vigente redujo la "
                "degradación Out-of-Time del Recall Glosada a menos de medio punto."
            ),
            "reproducedBaseline": {
                "label": "v2 reproducido sobre datos frescos",
                "f1Macro": _r(repro["test"]["f1_macro"]),
                "recallGlosada": _r(repro["test"]["recall_glosada"]),
            },
        },
    ]


def build_payload(prefix: str) -> dict[str, Any]:
    """Ensambla el objeto completo que consumirá el dashboard."""
    suite = _read_json(EVAL_DIR / f"{prefix}_model_suite.json")
    shap = _read_json(EVAL_DIR / f"{prefix}_shap_importance.json")
    horizon = _read_json(REPORTS_DIR / f"{prefix}_label_horizon.json")
    drift_path = _latest_drift_report()
    drift = _read_json(drift_path)
    logger.info("Reporte de drift: %s", drift_path.name)

    selection = suite["champion_selection"]
    models = build_models(suite)
    champion_id = selection["champion"]
    champion = next(m for m in models if m["id"] == champion_id)
    champion_algorithm = champion["label"].split(" ")[0]

    splits = build_split_summary(prefix)
    total_invoices = sum(s["rows"] for s in splits)

    global_shap = list(shap["global_importance"].items())

    payload: dict[str, Any] = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sourcePrefix": prefix,
        "randomState": suite.get("random_state", 42),
        "classNames": CLASS_ORDER,
        "champion": {
            "id": champion_id,
            "label": champion["label"],
            "algorithm": champion_algorithm,
            "version": "v3",
            "tieBreak": bool(selection.get("tie_break")),
            "test": champion["test"],
            "oot": champion["oot"],
        },
        "dataset": {
            "totalInvoices": total_invoices,
            "featureCount": len(global_shap),
            "windowStart": SPLIT_RANGES["train"][0],
            "windowEnd": SPLIT_RANGES["oot"][1],
            "labelHorizon": horizon["reliable_horizon"],
            "splits": splits,
            "monthly": [
                {
                    "month": m["mes"][:7],
                    "total": int(m["total_facturas"]),
                    "glosada": int(m["glosada"]),
                    "devuelta": int(m["devuelta"]),
                    "glosadaRate": _r(m["glosada_rate"]),
                    "censored": m["mes"] >= horizon["reliable_horizon"],
                }
                for m in horizon["monthly"]
            ],
        },
        "models": models,
        "championSelection": {
            "champion": champion_id,
            "tieBreak": bool(selection.get("tie_break")),
            "ranking": [
                {
                    "model": r["model"],
                    "label": MODEL_META.get(r["model"], {}).get("label", r["model"]),
                    "f1MacroTest": _r(r["f1_macro_test"]),
                    "recallGlosadaTest": _r(r["recall_glosada_test"]),
                    "recallGlosadaOot": _r(r["recall_glosada_oot"]),
                    "f1Degradation": _r(r["f1_degradation"]),
                    "eligible": bool(r["eligible"]),
                }
                for r in selection["ranking"]
            ],
        },
        "drift": {
            "referencePrefix": drift["reference_prefix"],
            "currentPrefix": drift["current_prefix"],
            "status": drift["overall_status"],
            "meanPsi": _r(drift["mean_psi"]),
            "maxPsi": _r(drift["max_psi"]),
            "nFeatures": int(drift["n_features"]),
            "counts": drift["drift_counts"],
            "top": [
                {"name": f["name"], "psi": _r(f["psi"]), "status": f["status"]}
                for f in _dedupe_drift(drift["top_drift_features"])
            ],
        },
        "shap": {
            "method": shap["method"],
            "nSample": int(shap["n_sample"]),
            "model": shap["model"],
            "global": [{"feature": k, "value": _r(v)} for k, v in global_shap[:20]],
            "perClass": {
                cls: [{"feature": k, "value": _r(v)} for k, v in feats.items()]
                for cls, feats in shap["per_class_top"].items()
            },
        },
    }
    payload["history"] = build_history(suite, payload["champion"])
    return payload


def main() -> None:
    """Punto de entrada del script."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prefix", default="20260910", help="Prefijo de fecha de los artefactos")
    parser.add_argument(
        "--out",
        type=Path,
        default=WEB_DATA_DIR / "metrics.json",
        help="Ruta del JSON de salida consumido por el dashboard",
    )
    args = parser.parse_args()

    payload = build_payload(args.prefix)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    champion = payload["champion"]
    logger.info(
        "Campeón: %s | F1 test %.4f | Recall Glosada test %.4f / OOT %.4f",
        champion["label"],
        champion["test"]["f1Macro"],
        champion["test"]["recallGlosada"],
        champion["oot"]["recallGlosada"],
    )
    logger.info(
        "%d modelos · %d facturas · drift %s (PSI medio %.3f)",
        len(payload["models"]),
        payload["dataset"]["totalInvoices"],
        payload["drift"]["status"],
        payload["drift"]["meanPsi"],
    )
    logger.info("Escrito: %s (%.1f KB)", args.out, args.out.stat().st_size / 1024)


if __name__ == "__main__":
    main()
