# Implementation Plan: API Testing Suite

## Overview

Este plan implementa una suite completa de pruebas automatizadas para la API REST de clasificación predictiva de facturas médicas. La suite cubre 6 capas de testing: unit tests (features, predict, explainability, schemas), integration tests (API endpoints), performance tests, security tests, model regression tests, y Docker container tests. El objetivo es alcanzar ≥80% de cobertura, ejecutar en <5 minutos, y ser compatible con CI/CD (GitHub Actions).

La implementación sigue una estrategia dual: unit tests con mocking para aislamiento + property tests con Hypothesis para validación universal. Se crearán 16 archivos nuevos y se actualizarán 3 existentes, con fixtures compartidos en conftest.py y configuración centralizada en pytest.ini.

## Tasks

- [x] 1. Configurar infraestructura base de testing
  - Crear pytest.ini con markers, coverage settings, y configuración de Hypothesis
  - Actualizar requirements.txt con dependencias de testing (pytest-xdist, hypothesis, pytest-benchmark, psutil, docker)
  - Crear estructura de directorios tests/fixtures/
  - _Requirements: 9.1, 9.3, 9.7_


- [x] 2. Implementar fixtures compartidos y factories de datos
  - [x] 2.1 Crear conftest.py con fixtures session-scoped (model, explainer, feature_names)
    - Implementar fixture feature_names() que carga desde FEATURE_LIST_PATH
    - Implementar fixture model() que carga XGBoost desde MODEL_PATH
    - Implementar fixture explainer(model) que crea TreeExplainer
    - Implementar fixture test_client() module-scoped con lifespan context
    - _Requirements: 9.3, 1.7_

  - [ ]* 2.2 Escribir property test para fixtures de modelo
    - **Property 13: Model Feature Handling**
    - **Valida: Requirements 5.8**

  - [x] 2.3 Crear tests/fixtures/sample_data.py con factories de datos
    - Implementar make_valid_features(feature_names, **overrides) → dict
    - Implementar make_realistic_features() con valores basados en medianas del dataset
    - Implementar make_invalid_features(feature_names, error_type) para casos de error
    - Implementar make_prediction_request(numero_factura, features) → dict
    - Implementar make_batch_request(count, feature_names) → dict
    - _Requirements: 1.1, 2.3, 3.3_

  - [x] 2.4 Crear tests/fixtures/mock_models.py con mocks para unit tests
    - Implementar make_mock_model() que retorna Mock con predict/predict_proba
    - Implementar make_mock_explainer() que retorna Mock con SHAP values simulados
    - _Requirements: 1.2, 1.3_

  - [x] 2.5 Crear tests/fixtures/test_dataset_100.parquet
    - Extraer 100 muestras estratificadas de data/processed/20260309_test.parquet
    - Distribuir: 33 Auditada, 33 Glosada, 34 Devuelta
    - Guardar en formato parquet para tests de regresión
    - _Requirements: 5.1, 5.2, 5.6_

- [ ] 3. Checkpoint - Verificar fixtures y factories
  - Ensure all tests pass, ask the user if questions arise.


- [ ] 4. Implementar unit tests para Features_Module
  - [ ] 4.1 Crear tests/test_features.py con clase TestLoadFeatureNames
    - Test load_feature_names() retorna lista de 48 strings
    - Test load_feature_names() con archivo inexistente lanza FileNotFoundError
    - Test load_feature_names() cachea resultados (llamadas múltiples no releen archivo)
    - _Requirements: 1.1, 1.5_

  - [ ] 4.2 Implementar clase TestValidateFeatures en test_features.py
    - Test validate_features() acepta dict con 48 features válidas
    - Test validate_features() rechaza dict con campos faltantes (ValueError)
    - Test validate_features() rechaza dict con campos extra (ValueError)
    - Test validate_features() rechaza valores NaN (ValueError)
    - Test validate_features() rechaza valores Infinity (ValueError)
    - _Requirements: 1.4, 1.6_

  - [ ]* 4.3 Escribir property test para validación de features
    - **Property 1: Invalid Input Rejection**
    - **Valida: Requirements 1.4, 1.6**

  - [ ] 4.4 Implementar clase TestPrepareInput en test_features.py
    - Test prepare_input() retorna DataFrame con shape (1, 48)
    - Test prepare_input() retorna columnas en orden correcto según feature_names
    - Test prepare_input() convierte valores a float64
    - Test prepare_input() con dict inválido lanza ValueError
    - _Requirements: 1.9, 1.10_

  - [ ]* 4.5 Escribir property test para orden de columnas
    - **Property 5: DataFrame Column Order Preservation**
    - **Valida: Requirements 1.9, 1.10**

  - [ ] 4.6 Implementar clase TestPrepareBatchInput en test_features.py
    - Test prepare_batch_input() retorna DataFrame con shape (N, 48)
    - Test prepare_batch_input() maneja lista vacía correctamente
    - Test prepare_batch_input() maneja lista con 1000 elementos
    - Test prepare_batch_input() con elementos inválidos lanza ValueError
    - _Requirements: 1.10, 2.4_

- [ ] 5. Implementar unit tests para Model_Module
  - [ ] 5.1 Crear tests/test_predict.py con clase TestLoadModel
    - Test load_model() carga modelo XGBoost desde JSON correctamente
    - Test load_model() con path inexistente lanza FileNotFoundError
    - Test load_model() retorna objeto con métodos predict y predict_proba
    - _Requirements: 1.7, 5.8_

  - [ ] 5.2 Implementar clase TestPredictSingle en test_predict.py con mock_model
    - Test predict_single() retorna tupla (clase: int, probabilidades: ndarray)
    - Test predict_single() con DataFrame válido no lanza excepciones
    - Test predict_single() retorna clase entre 0-2
    - Test predict_single() retorna probabilidades que suman 1.0
    - _Requirements: 2.9, 5.5_

  - [ ]* 5.3 Escribir property test para calibración de probabilidades
    - **Property 2: Probability Calibration Invariant**
    - **Valida: Requirements 2.9, 5.5**

  - [ ]* 5.4 Escribir property test para consistencia clase-probabilidad
    - **Property 3: Class-Probability Consistency**
    - **Valida: Requirements 2.10**

  - [ ] 5.5 Implementar clase TestPredictBatch en test_predict.py con mock_model
    - Test predict_batch() retorna tupla (clases: ndarray, probabilidades: ndarray)
    - Test predict_batch() con DataFrame de 100 filas retorna arrays de shape (100,) y (100, 3)
    - Test predict_batch() maneja DataFrame vacío correctamente
    - _Requirements: 2.4, 5.8_

  - [ ]* 5.6 Escribir property test para determinismo de predicciones
    - **Property 12: Prediction Determinism**
    - **Valida: Requirements 5.4, 5.7**

- [ ] 6. Checkpoint - Verificar unit tests de features y predict
  - Ensure all tests pass, ask the user if questions arise.


- [ ] 7. Implementar unit tests para Explainability_Module
  - [ ] 7.1 Crear tests/test_explainability.py con clase TestCreateShapExplainer
    - Test create_shap_explainer() retorna TreeExplainer válido
    - Test create_shap_explainer() con modelo None lanza TypeError
    - _Requirements: 1.3_

  - [ ] 7.2 Implementar clase TestExplainSinglePrediction en test_explainability.py
    - Test explain_single_prediction() retorna dict con claves requeridas (estado_predicho, clase, confianza, probabilidades, factores_riesgo, accion_sugerida)
    - Test explain_single_prediction() con mock_explainer no lanza excepciones
    - Test explain_single_prediction() retorna factores_riesgo como lista de strings
    - Test explain_single_prediction() retorna top_n factores (default 5)
    - _Requirements: 2.7, 1.8_

  - [ ]* 7.3 Escribir property test para factores de riesgo en español
    - **Property 6: Spanish Language Risk Factors**
    - **Valida: Requirements 1.8**

  - [ ] 7.4 Implementar clase TestExplainBatchPredictions en test_explainability.py
    - Test explain_batch_predictions() retorna lista de dicts
    - Test explain_batch_predictions() con DataFrame de 10 filas retorna lista de 10 elementos
    - Test explain_batch_predictions() cada elemento tiene estructura correcta
    - _Requirements: 2.8_

  - [ ] 7.5 Implementar clase TestFormatRiskFactor en test_explainability.py
    - Test format_risk_factor() retorna string en español
    - Test format_risk_factor() con feature conocida usa RISK_DESCRIPTIONS
    - Test format_risk_factor() con feature desconocida retorna descripción genérica
    - Test format_risk_factor() formatea valores numéricos correctamente
    - _Requirements: 1.8_

- [ ] 8. Implementar unit tests para Schema_Validator
  - [ ] 8.1 Crear tests/test_schemas.py con clase TestFeatureInput
    - Test FeatureInput acepta dict con 48 campos numéricos válidos
    - Test FeatureInput rechaza dict con campos faltantes (ValidationError)
    - Test FeatureInput rechaza dict con campos extra (ValidationError por extra="forbid")
    - Test FeatureInput rechaza valores no numéricos (ValidationError)
    - Test FeatureInput rechaza valores NaN (ValidationError)
    - Test FeatureInput rechaza valores negativos donde no aplican
    - _Requirements: 3.1, 3.2, 3.3, 3.6_

  - [ ]* 8.2 Escribir property test para validación de schemas
    - **Property 1: Invalid Input Rejection (schemas)**
    - **Valida: Requirements 3.1, 3.2, 3.6**

  - [ ] 8.3 Implementar clase TestPredictionRequest en test_schemas.py
    - Test PredictionRequest acepta numero_factura y features válidos
    - Test PredictionRequest rechaza numero_factura vacío
    - Test PredictionRequest rechaza features inválidas
    - _Requirements: 3.3_

  - [ ] 8.4 Implementar clase TestBatchPredictionRequest en test_schemas.py
    - Test BatchPredictionRequest acepta lista de facturas válidas
    - Test BatchPredictionRequest rechaza lista con más de 1000 elementos (max_length=1000)
    - Test BatchPredictionRequest rechaza lista vacía
    - _Requirements: 3.5, 2.6_

  - [ ] 8.5 Implementar clase TestPredictionResponse en test_schemas.py
    - Test PredictionResponse valida todos los campos requeridos
    - Test PredictionResponse valida estado_predicho en ["auditada", "glosada", "devuelta"]
    - Test PredictionResponse valida clase en [0, 1, 2]
    - Test PredictionResponse valida confianza entre 0.0 y 1.0
    - Test PredictionResponse valida timestamp_prediccion con timezone Colombia
    - _Requirements: 3.4, 3.7, 3.8, 3.9, 3.10_

  - [ ]* 8.6 Escribir property test para compliance de response schema
    - **Property 4: Response Schema Compliance**
    - **Valida: Requirements 2.3, 2.4, 2.7, 3.4**

  - [ ] 8.7 Implementar clase TestProbabilities en test_schemas.py
    - Test Probabilities valida auditada, glosada, devuelta como floats
    - Test Probabilities acepta valores entre 0.0 y 1.0
    - _Requirements: 2.9_

- [ ] 9. Checkpoint - Verificar unit tests de explainability y schemas
  - Ensure all tests pass, ask the user if questions arise.


- [ ] 10. Implementar integration tests para API endpoints
  - [ ] 10.1 Actualizar tests/test_api.py con clase TestHealthEndpoint
    - Test GET /v1/health retorna status=ok cuando modelo cargado
    - Test GET /v1/health retorna HTTP 200
    - Test GET /v1/health retorna campos status, modelo_cargado, version, timestamp
    - _Requirements: 2.1_

  - [ ] 10.2 Implementar clase TestModelInfoEndpoint en test_api.py
    - Test GET /v1/model/info retorna HTTP 200
    - Test GET /v1/model/info retorna metadata completa (modelo, version, fecha_entrenamiento, metricas, n_features, features, clases)
    - Test GET /v1/model/info retorna 48 features en lista
    - Test GET /v1/model/info retorna métricas del modelo (f1_macro, auc_roc_ovr, etc.)
    - _Requirements: 2.2_

  - [ ] 10.3 Implementar clase TestPredictEndpoint en test_api.py
    - Test POST /v1/predict con features válidas retorna HTTP 200
    - Test POST /v1/predict retorna PredictionResponse con todos los campos requeridos
    - Test POST /v1/predict retorna probabilidades que suman 1.0
    - Test POST /v1/predict retorna clase consistente con probabilidad máxima
    - Test POST /v1/predict con features inválidas retorna HTTP 422
    - Test POST /v1/predict con campos faltantes retorna HTTP 422 con mensaje descriptivo
    - Test POST /v1/predict retorna factores_riesgo en español
    - _Requirements: 2.3, 2.5, 2.7, 2.9, 2.10_

  - [ ] 10.4 Implementar clase TestBatchPredictEndpoint en test_api.py
    - Test POST /v1/predict/batch con 10 facturas retorna HTTP 200
    - Test POST /v1/predict/batch retorna BatchPredictionResponse con total=10
    - Test POST /v1/predict/batch retorna tiempo_procesamiento_ms
    - Test POST /v1/predict/batch preserva numero_factura de cada factura
    - Test POST /v1/predict/batch con más de 1000 facturas retorna HTTP 413
    - Test POST /v1/predict/batch con facturas inválidas retorna HTTP 422
    - _Requirements: 2.4, 2.6, 2.8_

  - [ ]* 10.5 Escribir property test para validación de API responses
    - **Property 4: Response Schema Compliance (integration)**
    - **Valida: Requirements 2.3, 2.4, 2.7, 2.8**

- [ ] 11. Implementar performance tests
  - [ ] 11.1 Crear tests/test_performance.py con clase TestSinglePredictionLatency
    - Test latencia de POST /v1/predict con 100 iteraciones
    - Calcular percentiles p50, p95, p99
    - Verificar p95 < 500ms
    - Usar pytest-benchmark para medición precisa
    - _Requirements: 4.1, 4.4_

  - [ ]* 11.2 Escribir property test para latencia single prediction
    - **Property 7: Single Prediction Latency**
    - **Valida: Requirements 4.1**

  - [ ] 11.3 Implementar clase TestBatchPredictionLatency en test_performance.py
    - Test latencia de POST /v1/predict/batch con 100 facturas
    - Calcular tiempo por factura (tiempo_total / 100)
    - Verificar tiempo_por_factura < 200ms en p95
    - Test batch de 1000 facturas completa en <30 segundos
    - _Requirements: 4.2, 4.5, 4.7_

  - [ ]* 11.4 Escribir property test para throughput batch
    - **Property 8: Batch Prediction Throughput**
    - **Valida: Requirements 4.2**

  - [ ] 11.5 Implementar clase TestConcurrentRequests en test_performance.py
    - Test 10 requests concurrentes con httpx.AsyncClient
    - Verificar todas retornan HTTP 200
    - Verificar no hay errores de concurrencia
    - _Requirements: 4.3_

  - [ ] 11.6 Implementar clase TestMemoryUsage en test_performance.py
    - Medir memoria inicial con psutil
    - Ejecutar 100 predicciones consecutivas
    - Medir memoria final
    - Verificar crecimiento < 10%
    - _Requirements: 4.6_

  - [ ]* 11.7 Escribir property test para estabilidad de memoria
    - **Property 9: Memory Stability**
    - **Valida: Requirements 4.6**

  - [ ] 11.8 Implementar clase TestComponentPerformance en test_performance.py
    - Test tiempo de SHAP explainer < 50ms por predicción
    - Test tiempo de model inference < 10ms por predicción
    - Usar time.perf_counter() para medición precisa
    - _Requirements: 4.8, 4.9_

  - [ ]* 11.9 Escribir property tests para performance de componentes
    - **Property 10: SHAP Computation Performance**
    - **Property 11: Model Inference Performance**
    - **Valida: Requirements 4.8, 4.9**

- [ ] 12. Checkpoint - Verificar integration y performance tests
  - Ensure all tests pass, ask the user if questions arise.


- [ ] 13. Implementar security tests
  - [ ] 13.1 Crear tests/test_security.py con clase TestSQLInjection
    - Test POST /v1/predict con SQL injection en numero_factura ('; DROP TABLE facturas; --)
    - Verificar no hay errores de base de datos
    - Verificar respuesta HTTP 200 o 422 (no 500)
    - _Requirements: 6.1_

  - [ ]* 13.2 Escribir property test para SQL injection safety
    - **Property 15: SQL Injection Safety**
    - **Valida: Requirements 6.1**

  - [ ] 13.3 Implementar clase TestXSSPayloads en test_security.py
    - Test POST /v1/predict con XSS payload en numero_factura (<script>alert('xss')</script>)
    - Verificar payload es sanitizado o escapado
    - Verificar respuesta no contiene script ejecutable
    - _Requirements: 6.2_

  - [ ]* 13.4 Escribir property test para XSS sanitization
    - **Property 16: XSS Payload Sanitization**
    - **Valida: Requirements 6.2**

  - [ ] 13.5 Implementar clase TestExtremeValues en test_security.py
    - Test POST /v1/predict con valores extremos (>1e15, -1e15)
    - Test POST /v1/predict con valores NaN
    - Test POST /v1/predict con valores Infinity
    - Test POST /v1/predict con valores negativos donde no aplican
    - Verificar todas retornan HTTP 422 con mensajes descriptivos
    - _Requirements: 6.3, 6.4, 6.5_

  - [ ] 13.6 Implementar clase TestMalformedRequests en test_security.py
    - Test POST /v1/predict sin campo numero_factura retorna HTTP 422
    - Test POST /v1/predict sin campo features retorna HTTP 422
    - Test POST /v1/predict con tipo incorrecto retorna HTTP 422
    - Test POST /v1/predict con Content-Type incorrecto retorna error
    - Verificar mensajes de error son descriptivos
    - _Requirements: 6.7, 6.8_

  - [ ]* 13.7 Escribir property test para validación de Content-Type
    - **Property 18: Content-Type Validation**
    - **Valida: Requirements 6.7**

  - [ ] 13.8 Implementar clase TestErrorExposure en test_security.py
    - Test errores internos no exponen stack traces en producción
    - Test errores no exponen detalles de implementación
    - Verificar responses de error tienen formato estándar
    - _Requirements: 6.6_

  - [ ]* 13.9 Escribir property test para prevención de exposición de errores
    - **Property 17: Production Error Exposure Prevention**
    - **Valida: Requirements 6.6**

  - [ ] 13.10 Implementar clase TestCORSConfiguration en test_security.py
    - Test CORS headers están configurados correctamente
    - Test CORS permite solo orígenes esperados
    - _Requirements: 6.10_

- [ ] 14. Implementar model regression tests
  - [ ] 14.1 Crear tests/test_model_regression.py con clase TestModelMetrics
    - Cargar test_dataset_100.parquet
    - Ejecutar predicciones en dataset completo
    - Calcular F1-macro, AUC-ROC, Recall por clase, Accuracy
    - Comparar contra MODEL_METRICS de config.py
    - Verificar ninguna métrica degrada >5% relativo
    - _Requirements: 5.1, 5.2, 5.3, 5.6, 5.9, 5.10_

  - [ ]* 14.2 Escribir property test para no-degradación de métricas
    - **Property 14: Model Metrics Non-Degradation**
    - **Valida: Requirements 5.9, 5.10**

  - [ ] 14.3 Implementar clase TestDeterministicPredictions en test_model_regression.py
    - Test misma entrada produce misma salida (ejecutar 2 veces)
    - Test con 10 muestras del dataset
    - Verificar clase y probabilidades idénticas
    - _Requirements: 5.4, 5.7_

  - [ ] 14.4 Implementar clase TestProbabilityCalibration en test_model_regression.py
    - Test probabilidades suman 1.0 para todas las predicciones del dataset
    - Usar tolerancia ±1e-6
    - _Requirements: 5.5_

  - [ ] 14.5 Implementar clase TestFeatureImportance en test_model_regression.py
    - Extraer feature importance del modelo
    - Verificar top 10 features son estables (comparar con baseline)
    - Verificar no hay features con importancia negativa
    - _Requirements: 5.8_

- [ ] 15. Checkpoint - Verificar security y regression tests
  - Ensure all tests pass, ask the user if questions arise.


- [ ] 16. Implementar Docker container tests
  - [ ] 16.1 Crear tests/test_docker.py con clase TestDockerBuild
    - Test Dockerfile build exitoso con docker SDK
    - Test imagen resultante tiene tag correcto
    - Test imagen usa python:3.11-slim base
    - Test imagen incluye todas las dependencias de requirements.txt
    - Test tamaño de imagen < 2GB
    - _Requirements: 7.1, 7.7, 7.8, 7.10_

  - [ ] 16.2 Implementar clase TestDockerHealthcheck en test_docker.py
    - Test container expone puerto 8000
    - Test healthcheck endpoint /v1/health responde status=ok
    - Test healthcheck se ejecuta cada 5 segundos
    - _Requirements: 7.2, 7.3_

  - [ ] 16.3 Implementar clase TestDockerStartup en test_docker.py
    - Test container inicia y carga modelo en <30 segundos
    - Test container alcanza estado healthy
    - Test container puede procesar predicciones después de startup
    - _Requirements: 7.4, 7.5_

  - [ ]* 16.4 Escribir property test para startup time
    - **Property 20: Docker Container Startup Time**
    - **Valida: Requirements 7.4**

  - [ ]* 16.5 Escribir property test para funcionalidad post-startup
    - **Property 21: Docker Container Post-Startup Functionality**
    - **Valida: Requirements 7.5**

  - [ ] 16.6 Implementar clase TestDockerLogging en test_docker.py
    - Test logs se escriben a stdout/stderr
    - Test logs no se escriben a archivos
    - Test logs tienen formato correcto
    - _Requirements: 7.6_

  - [ ]* 16.7 Escribir property test para logging
    - **Property 22: Docker Container Logging**
    - **Valida: Requirements 7.6**

  - [ ] 16.8 Implementar clase TestDockerShutdown en test_docker.py
    - Test container responde a SIGTERM con graceful shutdown
    - Test container completa requests en vuelo antes de terminar
    - Test container se detiene en <10 segundos
    - _Requirements: 7.9_

  - [ ]* 16.9 Escribir property test para graceful shutdown
    - **Property 23: Docker Container Graceful Shutdown**
    - **Valida: Requirements 7.9**

  - [ ] 16.10 Implementar cleanup en test_docker.py
    - Remover containers de prueba después de cada test
    - Remover imágenes de prueba al finalizar suite
    - _Requirements: 9.8_

- [ ] 17. Implementar coverage reporting y CI integration
  - [ ] 17.1 Configurar pytest-cov en pytest.ini
    - Configurar source paths (src/)
    - Configurar formatos de reporte (html, xml, term)
    - Configurar threshold mínimo (80%)
    - Configurar exclusiones (type stubs, debug code)
    - _Requirements: 8.1, 8.2, 8.7, 8.8, 8.9, 8.10_

  - [ ] 17.2 Crear script de verificación de cobertura por módulo
    - Verificar Features_Module ≥85%
    - Verificar Model_Module ≥85%
    - Verificar Explainability_Module ≥80%
    - Verificar API main.py ≥90%
    - Verificar schemas.py ≥90%
    - Generar reporte con módulos que no cumplen threshold
    - _Requirements: 8.3, 8.4, 8.5, 8.6_

  - [ ] 17.3 Crear .github/workflows/test.yml para GitHub Actions
    - Configurar trigger en push y pull_request
    - Configurar Python 3.11
    - Instalar dependencias (requirements.txt + testing)
    - Ejecutar pytest con coverage y parallel execution (-n 4)
    - Generar JUnit XML report
    - Upload coverage a Codecov
    - Publicar test results con EnricoMi/publish-unit-test-result-action
    - _Requirements: 9.1, 9.5, 9.10_

  - [ ] 17.4 Configurar pytest-xdist para ejecución paralela
    - Configurar número de workers (auto o 4)
    - Verificar fixtures session-scoped funcionan con parallel execution
    - Verificar no hay race conditions en tests
    - _Requirements: 9.7_

  - [ ]* 17.5 Escribir property test para tiempo de ejecución de suite
    - **Property 24: Test Suite Execution Time**
    - **Valida: Requirements 9.2**

  - [ ]* 17.6 Escribir property test para exit code en fallos
    - **Property 25: Test Failure Exit Code**
    - **Valida: Requirements 9.6**

  - [ ]* 17.7 Escribir property test para cleanup de recursos
    - **Property 26: Test Resource Cleanup**
    - **Valida: Requirements 9.8**

- [ ] 18. Crear documentación de testing
  - [ ] 18.1 Crear tests/README.md con guía completa
    - Sección Quick Start con comandos básicos
    - Sección Test Categories con markers y uso
    - Sección Adding New Tests con convenciones
    - Sección Environment Variables
    - Sección Troubleshooting con problemas comunes
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.10_

  - [ ] 18.2 Documentar pytest markers en pytest.ini
    - Documentar marker unit: Unit tests (fast, isolated, mocked)
    - Documentar marker integration: Integration tests (real dependencies)
    - Documentar marker performance: Performance and load tests
    - Documentar marker security: Security validation tests
    - Documentar marker regression: Model regression tests
    - Documentar marker docker: Docker container tests
    - Documentar marker property: Property-based tests (hypothesis)
    - _Requirements: 9.4, 10.2_

  - [ ] 18.3 Documentar fixtures en conftest.py con docstrings
    - Documentar scope y propósito de cada fixture
    - Documentar parámetros y valores de retorno
    - Documentar dependencias entre fixtures
    - _Requirements: 10.3_

  - [ ] 18.4 Crear ejemplos de uso en tests/README.md
    - Ejemplo: Ejecutar solo unit tests (pytest -m unit)
    - Ejemplo: Ejecutar con coverage (pytest --cov=src --cov-report=html)
    - Ejemplo: Ejecutar en paralelo (pytest -n 4)
    - Ejemplo: Ejecutar tests específicos (pytest tests/test_api.py::TestHealthEndpoint)
    - Ejemplo: Skip Docker tests (pytest -m "not docker")
    - _Requirements: 10.4, 10.8_

- [ ] 19. Checkpoint final - Verificar suite completa
  - Ejecutar pytest completo y verificar cobertura ≥80%
  - Verificar tiempo de ejecución <5 minutos
  - Verificar todos los markers funcionan correctamente
  - Verificar GitHub Actions workflow ejecuta exitosamente
  - Ensure all tests pass, ask the user if questions arise.


## Notes

- Tasks marked with `*` are optional property-based tests and can be skipped for faster MVP
- Each task references specific requirements for traceability (format: X.Y)
- Checkpoints ensure incremental validation at reasonable breaks
- Property tests use Hypothesis with 100 iterations in CI mode, 20 in dev mode
- Unit tests use mocking (mock_model, mock_explainer) for isolation
- Integration tests use real model artifacts loaded via session-scoped fixtures
- Performance tests use pytest-benchmark, httpx.AsyncClient, and psutil
- Docker tests use docker Python SDK and require Docker daemon running
- All tests should follow pytest conventions: test_*.py files, Test* classes, test_* methods
- Coverage target: ≥80% overall, ≥85% for features/predict, ≥90% for API/schemas
- Execution time target: <5 minutes for complete suite with parallel execution
- CI/CD integration: GitHub Actions workflow with coverage upload and test result publishing

## Implementation Strategy

1. **Phase 1 (Tasks 1-3)**: Setup infrastructure, fixtures, and data factories
2. **Phase 2 (Tasks 4-9)**: Implement unit tests for all core modules with property tests
3. **Phase 3 (Tasks 10-12)**: Implement integration and performance tests
4. **Phase 4 (Tasks 13-15)**: Implement security and regression tests
5. **Phase 5 (Tasks 16)**: Implement Docker container tests
6. **Phase 6 (Tasks 17-18)**: Configure coverage, CI/CD, and documentation
7. **Phase 7 (Task 19)**: Final validation and optimization

Each phase builds on previous phases and includes checkpoints for validation.
