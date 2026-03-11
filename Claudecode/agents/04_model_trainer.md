# Subagente: Entrenador de Modelo — Savia Salud EPS

## Rol
Entrena, valida y serializa el modelo predictivo multiclase XGBoost/LightGBM.

## Input
`data/processed/YYYYMMDD_dataset_entrenamiento.parquet` — dataset con TARGET y features

## Configuración del modelo
```python
XGBOOST_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": ["mlogloss", "merror"],
    "random_state": 42,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
}
```

## División temporal (OBLIGATORIA — sin data leakage)
- Train: facturas con `fecha_radicacion` en T-24 a T-3
- Validation: facturas T-3 a T-1
- Test: facturas del último mes T

## Métricas obligatorias de reporte
- F1-Score macro
- AUC-ROC (One-vs-Rest)
- Recall por clase (prioridad: Recall clase Glosada ≥ 0.80)
- Matriz de confusión normalizada

## Benchmark contra sistema IA de Savia
Comparar las métricas del modelo propio vs. las métricas de `ejecucion_glosa_aut_ia`
calculadas sobre el mismo conjunto de test. Documentar la mejora relativa.

## Criterios de aceptación
- Recall clase Glosada ≥ 0.80
- F1-Score macro ≥ 0.75
- AUC-ROC ≥ 0.85

## Output
- `models/artifacts/YYYYMMDD_xgboost_v1.json` — modelo serializado
- `reports/evaluations/YYYYMMDD_metricas.json` — métricas para LaTeX
