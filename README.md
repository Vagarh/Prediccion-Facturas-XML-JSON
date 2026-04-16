Codigo/
│
├── notebooks/                     # Análisis y prototipado (9 notebooks)
│   ├── 00_analisis_bd.ipynb        # Mapeo de BD MySQL (554 tablas)
│   ├── 01_exploracion_tablas_bd.ipynb
│   ├── 02_construccion_dataset.ipynb  # Extracción 728K filas, JOINs validados
│   ├── 03_EDA.ipynb                # EDA: 12 análisis, 11 figuras a 300dpi
│   ├── 04_Feature_Engineering.ipynb   # 48 features, right-censoring corregido
│   ├── 05_Model_Training.ipynb     # XGBoost vs LightGBM vs RF
│   ├── 06_SHAP_Explainability.ipynb   # TreeExplainer, beeswarm, waterfall
│   ├── 07_Validacion_Datos_Reales.ipynb  # Validación OOT (concept drift)
│   └── 08_Analisis_Profundo_Comparacion.ipynb  # Ground truth validation
│
├── src/                           # Código de producción modularizado
│   ├── config.py                  # Configuración central, paths, constantes
│   ├── features/
│   │   └── engineering.py         # Validación 48 features, prepare_input/batch
│   ├── models/
│   │   └── predict.py             # Carga XGBoost JSON, predict_single/batch
│   ├── explainability/
│   │   └── shap_explanations.py   # TreeExplainer + explicaciones en español
│   └── api/
│       ├── main.py                # FastAPI: 4 endpoints con lifespan
│       └── schemas.py             # Pydantic v2: FeatureInput (48 campos)
│
├── tests/                         # Suite de testing — 147 tests, 96.85% cobertura
│   ├── conftest.py                # Fixtures session/module/function scoped
│   ├── test_config.py             # 10 tests — unit
│   ├── test_features.py           # 22 tests — unit, property (Hypothesis)
│   ├── test_predict.py            # 15 tests — unit, integration, property
│   ├── test_explainability.py     # 20 tests — unit, integration
│   ├── test_schemas.py            # 23 tests — unit, property
│   ├── test_api.py                # 22 tests — integration
│   ├── test_performance.py        # 7 tests  — performance (<500ms p95)
│   ├── test_security.py           # 12 tests — SQL injection, XSS, path traversal
│   ├── test_model_regression.py   # 10 tests — regression con dataset estático
│   ├── test_docker.py             # 6 tests  — Docker (requiere daemon)
│   └── fixtures/
│       ├── sample_data.py         # Factories: features válidas, inválidas, maliciosas
│       ├── mock_models.py         # Mocks XGBoost y SHAP para unit tests
│       ├── create_test_dataset.py # Script: 100 muestras estratificadas
│       └── test_dataset_100.parquet  # Dataset de regresión (33/33/34)
│
├── data/
│   ├── raw/                       # Datos ORIGINALES e INMUTABLES — NO MODIFICAR
│   ├── database_analysis/         # Muestras y análisis de BD (2 filas × 15 tablas)
│   ├── Info_data_base/            # Metadata de 554 tablas
│   ├── interim/                   # Datos en proceso de transformación
│   └── processed/                 # Dataset final en Parquet
│       ├── 20260309_train.parquet  # 515K facturas
│       ├── 20260309_val.parquet    # 78K facturas
│       ├── 20260309_test.parquet   # 40K facturas
│       └── 20260309_feature_list.txt  # 48 feature names
│
├── models/artifacts/              # Modelos serializados
│   ├── 20260309_xgboost_v1.json   # Mejor modelo — AUC 0.908, F1 0.749
│   ├── 20260309_lightgbm_v1.joblib
│   └── 20260309_randomforest_v1.joblib
│
├── reports/
│   ├── figures/                   # Figuras a 300dpi para tesis
│   │   ├── shap_01_importancia_global.png
│   │   ├── shap_02_beeswarm_{auditada,glosada,devuelta}_v2.png
│   │   ├── shap_03_waterfall_{auditada,glosada,devuelta}_v2.png
│   │   └── shap_04_dependence_glosada.png
│   └── evaluations/               # Métricas y artefactos JSON
│       ├── 20260309_shap_importance.json
│       └── 20260309_shap_values_sample.npz
│
├── .github/workflows/
│   └── tests.yml                  # CI/CD: 3 jobs (unit, integration, coverage)
│
├── Dockerfile                     # python:3.11-slim, healthcheck, uvicorn :8000
├── requirements.txt               # Dependencias: xgboost, shap, fastapi, pytest...
├── pytest.ini                     # Markers, coverage ≥80%, Hypothesis profiles
└── regenerar_figuras_shap.py      # Script para regenerar figuras SHAP corregidas
