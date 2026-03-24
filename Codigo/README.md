# Sistema Predictivo para Auditoría Prestacional — Savia Salud EPS
### Maestría en Ciencia de Datos · Universidad EAFIT · 2026

Sistema de Machine Learning para clasificación automática de facturas médicas en tres estados:
**Auditada**, **Glosada** y **Devuelta**. Incluye API REST productiva con explicabilidad SHAP integrada.

---

## Métricas del Modelo (XGBoost v1)

| Métrica | Resultado |
|---|---|
| AUC-ROC (OvR) | **0.9078** |
| F1-macro | **0.7490** |
| Recall Glosada | **72.6%** |
| Dataset entrenamiento | 633K facturas (mar-2024 – jul-2025) |
| Features | 48 variables |

---

## Estructura del Proyecto

```
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
```

---

## Inicio Rápido

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Levantar la API

```bash
uvicorn src.api.main:app --reload --port 8000
```

O con Docker:

```bash
docker build -t auditoria-api .
docker run -p 8000:8000 auditoria-api
```

### 3. Endpoints disponibles

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/v1/health` | Health check |
| `GET` | `/v1/model/info` | Versión, métricas, fecha |
| `POST` | `/v1/predict` | Predicción individual (48 features) |
| `POST` | `/v1/predict/batch` | Predicción batch |

### 4. Ejemplo de predicción

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"numero_factura": "FEV-2024-001234", "features": {...}}'
```

Respuesta:
```json
{
  "estado_predicho": "glosada",
  "confianza": 0.98,
  "probabilidades": {"auditada": 0.02, "glosada": 0.98, "devuelta": 0.00},
  "factores_riesgo": [
    "Log ítems con autorización alto — impacto +0.65",
    "Combinación de alto valor y alta tasa de glosa (riesgo compuesto)"
  ],
  "causal_probable": "SO0201",
  "accion_sugerida": "Enviar a auditor médico especializado"
}
```

---

## Ejecutar Tests

```bash
# Todos (excepto Docker)
pytest -m "not docker" -v

# Solo unit tests (rápido, sin modelo en disco)
pytest -m "unit" -v

# Con reporte de cobertura
pytest -m "not docker and not performance" --cov=src --cov-report=html

# Paralelo
pytest -m "not docker" -n auto
```

### Resultados actuales

| Métrica | Resultado | Meta |
|---|---|---|
| Tests ejecutados | 141 passed, 6 deselected (docker) | — |
| Cobertura global | **96.85%** | ≥ 80% ✅ |
| Tiempo ejecución | **21.76s** | < 5 min ✅ |

---

## Notebooks — Flujo de Trabajo

```
00 → 01 → 02          Mapeo y extracción de BD MySQL
           ↓
          03           EDA (12 análisis, 11 figuras)
           ↓
          04           Feature Engineering (48 features, right-censoring)
           ↓
          05           Entrenamiento XGBoost/LightGBM/RF
           ↓
          06           Explicabilidad SHAP (TreeExplainer exacto)
           ↓
       07 → 08         Validación OOT + análisis profundo ground truth
```

---

## Hallazgos Críticos

- **Right-censoring:** Los splits val/test originales no tenían clase Glosada (facturas ago-2025+ sin auditar). Corregido truncando al período mar-2024 – jul-2025.
- **Concept drift OOT:** F1 baja de 0.743 (test) a 0.594 (OOT ago-dic 2025). Causa: Glosada pasó de 40% a 67% del volumen. Mitigación: reentrenamiento trimestral automático.
- **Ground truth validado:** JOINs con `cm_auditoria_motivos_glosas` confirmados correctos. `estado_glosa_aut` y `ejecucion_glosa_aut_ia` son estados del pipeline interno, NO ground truth.

---

## Arquitectura AWS (Propuesta para Producción)

```
IPS → API Gateway → Lambda (predict) → Response
EventBridge (cron) → Step Functions → Lambda (batch) → S3 → RDS
CloudWatch (PSI drift >15%) → alerta → SageMaker reentrenamiento
```

Costo estimado: **~$38 USD/mes** para 150K facturas/mes.
ROI: 1 glosa detectada = 170× el costo mensual de infraestructura.

---

## Stack Tecnológico

| Capa | Tecnología |
|---|---|
| Lenguaje | Python 3.11 |
| Modelo | XGBoost 3.2, LightGBM, scikit-learn |
| Explicabilidad | SHAP 0.51 (TreeExplainer exacto) |
| API | FastAPI, Pydantic v2, Uvicorn |
| Testing | pytest, Hypothesis, psutil |
| CI/CD | GitHub Actions |
| Contenedor | Docker (python:3.11-slim) |
| Datos | MySQL (BD Savia Salud), Parquet |

---

*Datos provistos por Savia Salud EPS bajo acuerdo de confidencialidad para uso académico.*
*Proyecto de tesis — Maestría en Ciencia de Datos, Universidad EAFIT, 2026.*
