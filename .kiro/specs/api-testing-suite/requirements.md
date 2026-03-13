# Requirements Document

## Introduction

Este documento define los requisitos para una suite completa de pruebas de validación del código de producción de la API REST de clasificación predictiva de facturas médicas (Fase 6). La suite debe verificar que el sistema es viable y operativo antes del despliegue, cubriendo tests unitarios, integración, validación de datos, performance, regresión del modelo y seguridad.

El sistema bajo prueba es una API REST FastAPI que expone un modelo XGBoost v1 (F1-macro=0.749, AUC=0.908) con 4 endpoints, 48 features de entrada validadas con Pydantic v2, y explicabilidad SHAP. La suite debe ejecutarse en CI/CD y garantizar cobertura mínima del 80% según estándares del proyecto (CLAUDE.MD §3.1).

## Glossary

- **Test_Suite**: Conjunto completo de pruebas automatizadas que validan el código de producción
- **API_System**: Sistema FastAPI con 4 endpoints (health, model/info, predict, predict/batch)
- **Model_Module**: Módulo `src/models/predict.py` que carga y ejecuta el modelo XGBoost
- **Features_Module**: Módulo `src/features/engineering.py` que valida y prepara las 48 features
- **Explainability_Module**: Módulo `src/explainability/shap_explanations.py` que genera explicaciones SHAP
- **Schema_Validator**: Pydantic v2 schemas en `src/api/schemas.py` que validan entrada/salida
- **Docker_Container**: Contenedor Docker construido desde `Dockerfile` con la aplicación
- **Coverage_Report**: Reporte de cobertura de código generado por pytest-cov
- **CI_Pipeline**: Pipeline de integración continua que ejecuta tests antes del despliegue
- **Feature_Input**: Schema Pydantic con 48 campos numéricos obligatorios del modelo
- **SHAP_Explainer**: TreeExplainer de SHAP que calcula valores de contribución de features
- **Latency_Target**: Tiempo máximo de respuesta (<200ms batch, <500ms tiempo real según CLAUDE.MD §3.4)
- **Model_Metrics**: Métricas de evaluación del modelo (F1-macro ≥0.75, Recall Glosada ≥0.73)

## Requirements

### Requirement 1: Unit Testing de Módulos Core

**User Story:** Como desarrollador, quiero tests unitarios de los módulos core (features, predict, explainability), para verificar que cada componente funciona correctamente de forma aislada.

#### Acceptance Criteria

1. THE Test_Suite SHALL test all public functions in Features_Module with valid and invalid inputs
2. THE Test_Suite SHALL test all public functions in Model_Module with mock model artifacts
3. THE Test_Suite SHALL test all public functions in Explainability_Module with mock SHAP explainer
4. WHEN a function receives invalid input, THE Test_Suite SHALL verify that appropriate exceptions are raised
5. THE Test_Suite SHALL verify that Features_Module validates all 48 required fields from Feature_Input
6. THE Test_Suite SHALL verify that Features_Module rejects inputs with missing or extra fields
7. THE Test_Suite SHALL verify that Model_Module loads the XGBoost model from JSON format correctly
8. THE Test_Suite SHALL verify that Explainability_Module generates risk factors in Spanish language
9. THE Test_Suite SHALL verify that prepare_input function returns DataFrame with correct column order
10. THE Test_Suite SHALL verify that prepare_batch_input function handles lists of feature dictionaries correctly

### Requirement 2: Integration Testing de API Endpoints

**User Story:** Como desarrollador, quiero tests de integración end-to-end de los 4 endpoints de la API, para verificar que el sistema completo funciona correctamente.

#### Acceptance Criteria

1. THE Test_Suite SHALL test GET /v1/health endpoint and verify status=ok response
2. THE Test_Suite SHALL test GET /v1/model/info endpoint and verify all metadata fields are present
3. THE Test_Suite SHALL test POST /v1/predict endpoint with valid Feature_Input and verify PredictionResponse schema
4. THE Test_Suite SHALL test POST /v1/predict/batch endpoint with multiple facturas and verify BatchPredictionResponse schema
5. WHEN API_System receives invalid Feature_Input, THE Test_Suite SHALL verify HTTP 422 status code is returned
6. WHEN API_System receives batch request with more than 1000 facturas, THE Test_Suite SHALL verify HTTP 413 status code is returned
7. THE Test_Suite SHALL verify that predict endpoint returns all required fields (estado_predicho, clase, confianza, probabilidades, factores_riesgo, accion_sugerida)
8. THE Test_Suite SHALL verify that batch endpoint returns tiempo_procesamiento_ms field
9. THE Test_Suite SHALL verify that probabilidades sum to 1.0 for each prediction
10. THE Test_Suite SHALL verify that clase matches the highest probability in probabilidades

### Requirement 3: Schema Validation Testing

**User Story:** Como desarrollador, quiero tests de validación de schemas Pydantic, para asegurar que los datos de entrada y salida cumplen con las especificaciones.

#### Acceptance Criteria

1. THE Test_Suite SHALL verify that Feature_Input schema rejects inputs with missing required fields
2. THE Test_Suite SHALL verify that Feature_Input schema rejects inputs with extra fields (extra="forbid")
3. THE Test_Suite SHALL verify that Feature_Input schema accepts all 48 valid numeric fields
4. THE Test_Suite SHALL verify that PredictionResponse schema validates all output fields correctly
5. THE Test_Suite SHALL verify that BatchPredictionRequest schema enforces max_length=1000 constraint
6. WHEN Schema_Validator receives non-numeric values for features, THE Test_Suite SHALL verify ValidationError is raised
7. THE Test_Suite SHALL verify that estado_predicho field contains only valid values (auditada, glosada, devuelta)
8. THE Test_Suite SHALL verify that clase field contains only valid codes (0, 1, 2)
9. THE Test_Suite SHALL verify that confianza field is between 0.0 and 1.0
10. THE Test_Suite SHALL verify that timestamp_prediccion field uses Colombia timezone (America/Bogota)

### Requirement 4: Performance and Load Testing

**User Story:** Como ingeniero de operaciones, quiero tests de performance y carga, para verificar que la API cumple con los objetivos de latencia y throughput.

#### Acceptance Criteria

1. WHEN API_System processes single prediction, THE Test_Suite SHALL verify response time is less than 500ms
2. WHEN API_System processes batch of 100 facturas, THE Test_Suite SHALL verify response time is less than 200ms per factura
3. THE Test_Suite SHALL verify that API_System can handle at least 10 concurrent requests without errors
4. THE Test_Suite SHALL measure and report average latency for predict endpoint
5. THE Test_Suite SHALL measure and report average latency for predict/batch endpoint
6. THE Test_Suite SHALL verify that memory usage remains stable during 100 consecutive predictions
7. WHEN API_System processes 1000 facturas in batch, THE Test_Suite SHALL verify total processing time is less than 30 seconds
8. THE Test_Suite SHALL verify that SHAP_Explainer computation does not exceed 50ms per prediction
9. THE Test_Suite SHALL verify that model inference time is less than 10ms per prediction
10. THE Test_Suite SHALL generate performance report with latency percentiles (p50, p95, p99)

### Requirement 5: Model Regression Testing

**User Story:** Como científico de datos, quiero tests de regresión del modelo, para asegurar que las métricas de evaluación se mantienen después de cambios en el código.

#### Acceptance Criteria

1. THE Test_Suite SHALL verify that Model_Module achieves F1-macro score greater than or equal to 0.75 on test set
2. THE Test_Suite SHALL verify that Model_Module achieves Recall for Glosada class greater than or equal to 0.73
3. THE Test_Suite SHALL verify that Model_Module achieves AUC-ROC greater than or equal to 0.90
4. THE Test_Suite SHALL verify that Model_Module predictions match expected outputs for known test cases
5. THE Test_Suite SHALL verify that model output probabilities are calibrated (sum to 1.0)
6. WHEN Model_Module processes test dataset, THE Test_Suite SHALL verify accuracy is greater than or equal to 0.74
7. THE Test_Suite SHALL verify that model predictions are deterministic for same input
8. THE Test_Suite SHALL verify that model handles all 48 features without errors
9. THE Test_Suite SHALL compare current model metrics against baseline metrics from MODEL_METRICS constant
10. IF model metrics degrade by more than 5%, THEN THE Test_Suite SHALL fail the regression test

### Requirement 6: Security Testing

**User Story:** Como ingeniero de seguridad, quiero tests de seguridad, para verificar que la API es resistente a inputs maliciosos y ataques comunes.

#### Acceptance Criteria

1. WHEN API_System receives SQL injection attempts in numero_factura field, THE Test_Suite SHALL verify no database errors occur
2. WHEN API_System receives XSS payloads in string fields, THE Test_Suite SHALL verify payloads are sanitized
3. WHEN API_System receives extremely large numeric values (>1e15), THE Test_Suite SHALL verify graceful error handling
4. WHEN API_System receives negative values for features that should be positive, THE Test_Suite SHALL verify validation errors
5. WHEN API_System receives NaN or Infinity values, THE Test_Suite SHALL verify appropriate error responses
6. THE Test_Suite SHALL verify that API_System does not expose internal error stack traces in production mode
7. THE Test_Suite SHALL verify that API_System validates Content-Type header for POST requests
8. WHEN API_System receives requests without required fields, THE Test_Suite SHALL verify HTTP 422 with descriptive error message
9. THE Test_Suite SHALL verify that API_System rate limiting works correctly (if implemented)
10. THE Test_Suite SHALL verify that API_System CORS configuration allows only expected origins

### Requirement 7: Docker Container Testing

**User Story:** Como ingeniero DevOps, quiero tests del contenedor Docker, para verificar que la aplicación se ejecuta correctamente en el entorno containerizado.

#### Acceptance Criteria

1. THE Test_Suite SHALL verify that Docker_Container builds successfully from Dockerfile
2. THE Test_Suite SHALL verify that Docker_Container exposes port 8000 correctly
3. THE Test_Suite SHALL verify that Docker_Container healthcheck endpoint responds with status=ok
4. WHEN Docker_Container starts, THE Test_Suite SHALL verify that model artifacts are loaded within 30 seconds
5. THE Test_Suite SHALL verify that Docker_Container can process predictions after startup
6. THE Test_Suite SHALL verify that Docker_Container logs are written to stdout/stderr
7. THE Test_Suite SHALL verify that Docker_Container uses python:3.11-slim base image
8. THE Test_Suite SHALL verify that Docker_Container includes all required dependencies from requirements.txt
9. WHEN Docker_Container receives SIGTERM signal, THE Test_Suite SHALL verify graceful shutdown
10. THE Test_Suite SHALL verify that Docker_Container size is less than 2GB

### Requirement 8: Coverage Reporting

**User Story:** Como líder técnico, quiero reportes de cobertura de código, para asegurar que al menos el 80% del código está cubierto por tests.

#### Acceptance Criteria

1. THE Test_Suite SHALL generate Coverage_Report using pytest-cov tool
2. THE Test_Suite SHALL verify that overall code coverage is greater than or equal to 80%
3. THE Test_Suite SHALL verify that Features_Module has coverage greater than or equal to 85%
4. THE Test_Suite SHALL verify that Model_Module has coverage greater than or equal to 85%
5. THE Test_Suite SHALL verify that Explainability_Module has coverage greater than or equal to 80%
6. THE Test_Suite SHALL verify that API endpoints (main.py) have coverage greater than or equal to 90%
7. THE Coverage_Report SHALL include line coverage, branch coverage, and function coverage metrics
8. THE Coverage_Report SHALL identify uncovered lines and branches
9. THE Coverage_Report SHALL be generated in HTML format for human review
10. THE Coverage_Report SHALL be generated in XML format for CI_Pipeline integration

### Requirement 9: Test Execution and CI Integration

**User Story:** Como ingeniero de CI/CD, quiero que la suite de tests se ejecute automáticamente en el pipeline, para prevenir regresiones antes del despliegue.

#### Acceptance Criteria

1. THE Test_Suite SHALL be executable with single command `pytest` from project root
2. THE Test_Suite SHALL complete execution in less than 5 minutes
3. THE Test_Suite SHALL use pytest fixtures for setup and teardown of test resources
4. THE Test_Suite SHALL use pytest markers to categorize tests (unit, integration, performance, security)
5. THE Test_Suite SHALL generate JUnit XML report for CI_Pipeline integration
6. WHEN any test fails, THE Test_Suite SHALL exit with non-zero status code
7. THE Test_Suite SHALL support parallel execution with pytest-xdist plugin
8. THE Test_Suite SHALL clean up temporary files and resources after execution
9. THE Test_Suite SHALL provide clear error messages when tests fail
10. THE Test_Suite SHALL be compatible with GitHub Actions, GitLab CI, and Jenkins

### Requirement 10: Test Documentation

**User Story:** Como nuevo desarrollador del equipo, quiero documentación clara de cómo ejecutar y mantener los tests, para poder contribuir efectivamente al proyecto.

#### Acceptance Criteria

1. THE Test_Suite SHALL include README.md file with instructions for running tests locally
2. THE Test_Suite SHALL document all pytest markers and their purposes
3. THE Test_Suite SHALL document all pytest fixtures and their scopes
4. THE Test_Suite SHALL document how to run specific test categories (e.g., only unit tests)
5. THE Test_Suite SHALL document how to generate and view coverage reports
6. THE Test_Suite SHALL document how to add new tests following project conventions
7. THE Test_Suite SHALL document required environment variables for testing
8. THE Test_Suite SHALL document how to run tests inside Docker_Container
9. THE Test_Suite SHALL document how to run performance tests with locust or similar tool
10. THE Test_Suite SHALL document troubleshooting steps for common test failures
