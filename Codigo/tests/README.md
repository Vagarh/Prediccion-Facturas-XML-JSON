# Suite de Testing — API Predicción Auditoría

## Arquitectura de 6 Capas

```
Unit Tests → Integration Tests → Performance Tests
Security Tests → Model Regression → Docker Tests
```

## Ejecutar Tests

```bash
# Todos los tests (excepto docker)
pytest -m "not docker" -v

# Solo unit tests (rápido, sin modelo)
pytest -m "unit" -v

# Integration tests (requiere modelo en disco)
pytest -m "integration" -v

# Property tests (Hypothesis)
pytest -m "property" -v

# Performance tests
pytest -m "performance" -v

# Security tests
pytest -m "security" -v

# Regression tests (requiere test_dataset_100.parquet)
pytest -m "regression" -v

# Docker tests (requiere Docker daemon)
pytest -m "docker" -v

# Paralelo (rápido)
pytest -m "not docker" -n auto

# Con reporte de cobertura
pytest -m "not docker and not performance" --cov=src --cov-report=html
```

## Markers

| Marker | Descripción | Requiere Modelo |
|---|---|---|
| `unit` | Tests unitarios aislados con mocks | No |
| `property` | Property-based tests (Hypothesis) | Algunos |
| `integration` | Tests end-to-end con API real | Sí |
| `performance` | Benchmarks de latencia y memoria | Sí |
| `security` | Validación contra ataques | Sí |
| `regression` | Métricas del modelo contra dataset | Sí |
| `docker` | Tests de contenedor Docker | Sí + Docker |

## Generar Dataset de Prueba

```bash
python tests/fixtures/create_test_dataset.py
```

Genera `tests/fixtures/test_dataset_100.parquet` (100 muestras estratificadas 33/33/34).

## Inventario de Tests

| Archivo | Tests | Tipo |
|---|---|---|
| `test_config.py` | 10 | unit |
| `test_features.py` | 22 | unit, property |
| `test_predict.py` | 15 | unit, integration, property |
| `test_explainability.py` | 20 | unit, integration |
| `test_schemas.py` | 23 | unit, property |
| `test_api.py` | 22 | integration |
| `test_performance.py` | 7 | performance |
| `test_security.py` | 12 | security |
| `test_model_regression.py` | 10 | regression |
| `test_docker.py` | 6 | docker |
| **Total** | **~147** | |

## Fixtures Compartidos (`conftest.py`)

- **Session:** `feature_names`, `model`, `explainer`
- **Module:** `test_client`
- **Function:** `sample_features`, `realistic_features`

## Factories (`tests/fixtures/`)

- `sample_data.py` — Features válidas, realistas, inválidas, payloads maliciosos
- `mock_models.py` — Mocks de XGBoost y SHAP TreeExplainer
- `create_test_dataset.py` — Script para generar dataset de regresión

## Cobertura Objetivo

- Global: ≥80%
- `src/features/`: ≥85%
- `src/api/`: ≥90%
- `src/models/`: ≥85%
