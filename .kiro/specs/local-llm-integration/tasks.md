# Plan de Implementación: Integración LLM Local

## Visión General

Extender el pipeline de auditoría prestacional de Savia Salud EPS con embeddings semánticos
(Sentence Transformers + spaCy) y generación de explicaciones en lenguaje natural (Ollama),
manteniendo el XGBoost v1 como modelo base irremplazable y cumplimiento estricto con la
Ley 1581 de Colombia. Toda la infraestructura opera en local.

## Tareas

- [x] 1. Infraestructura y dependencias LLM local
  - Agregar `sentence-transformers`, `spacy`, `ollama` (cliente Python), `numpy` (NPZ),
    `scipy` (PSI) a `Codigo/requirements.txt` o `pyproject.toml`
  - Descargar modelo spaCy en español: `python -m spacy download es_core_news_md`
  - Documentar en `Codigo/README.md` los pasos de instalación de Ollama y descarga de
    modelos candidatos (`ollama pull llama3.2`, `mistral`, `qwen2.5`, `phi3`)
  - Agregar constantes LLM a `Codigo/src/config.py`:
    `OLLAMA_BASE_URL`, `EMBEDDING_MODEL_NAME`, `EMBEDDING_DIM`, `OLLAMA_TIMEOUT_S`,
    `PSI_DRIFT_WARN_THRESHOLD = 0.20`, `PSI_DRIFT_ALERT_THRESHOLD = 0.15`,
    `LLM_CANDIDATES`, `EXPLANATION_MAX_SECONDS = 10.0`
  - _Requisitos: 1.1, 1.3, 8.1_

- [x] 2. Privacy Guard
  - [x] 2.1 Implementar `Codigo/src/privacy/guard.py`
    - Clase `PrivacyGuard` con método `verify_local_only()` que comprueba que
      `OLLAMA_BASE_URL` apunta a `localhost`/`127.0.0.1` y lanza `PrivacyViolationError`
      si detecta host externo
    - Método `check_ollama_available() -> bool` que hace GET a
      `{OLLAMA_BASE_URL}/api/tags` con timeout corto
    - Método `log_inference(model: str, input_hash: str) -> None` que escribe en el
      logger local sin almacenar contenido textual
    - Usar `logging` (no `print`), type hints y docstrings Google-style
    - _Requisitos: 1.1, 1.2, 1.5_
  - [ ] 2.2 Escribir tests unitarios para PrivacyGuard
    - Verificar que `verify_local_only()` lanza excepción con host externo
    - Verificar que `check_ollama_available()` retorna `False` cuando el servidor no responde
    - _Requisitos: 1.1, 1.2_

- [x] 3. NLP Pipeline con spaCy
  - [x] 3.1 Implementar `Codigo/src/nlp/pipeline.py`
    - Función `load_nlp_pipeline() -> spacy.Language` que carga `es_core_news_md`
      con caché singleton (carga una sola vez por proceso)
    - Función `preprocess_text(text: str | None) -> str` que aplica:
      tokenización, minúsculas, eliminación de stopwords médicos, lematización,
      y retorna cadena vacía `""` si el input es `None` o vacío
    - Lista `MEDICAL_STOPWORDS` con términos irrelevantes para embeddings clínicos
      (ej. "paciente", "presenta", "refiere")
    - Usar `logging`, type hints y docstrings Google-style
    - _Requisitos: 2.3_
  - [ ] 3.2 Escribir tests unitarios para NLP Pipeline
    - Verificar que `preprocess_text(None)` retorna `""`
    - Verificar que stopwords médicos son eliminados
    - Verificar que la función es idempotente (aplicar dos veces = mismo resultado)
    - _Requisitos: 2.3, 2.4_
  - [ ] 3.3 Escribir property test para NLP Pipeline
    - **Propiedad 1: Idempotencia del preprocesamiento**
    - Para cualquier texto de entrada, `preprocess_text(preprocess_text(t)) == preprocess_text(t)`
    - **Valida: Requisito 2.3**

- [x] 4. Embedding Service
  - [x] 4.1 Implementar `Codigo/src/embeddings/service.py`
    - Clase `EmbeddingService` con `__init__(model_name: str, random_state: int = RANDOM_STATE)`
    - Método `encode(text: str) -> np.ndarray` que retorna vector de dimensión `EMBEDDING_DIM`
    - Método `encode_batch(texts: list[str]) -> np.ndarray` para procesamiento vectorizado
    - Método `zero_vector() -> np.ndarray` que retorna vector de ceros de dimensión `EMBEDDING_DIM`
    - Lógica de null-handling: si `text` es `None` o `""` después de preprocesamiento,
      retornar `zero_vector()` sin invocar el modelo
    - Usar `logging`, type hints y docstrings Google-style; `RANDOM_STATE = 42`
    - _Requisitos: 2.1, 2.4, 8.1_
  - [ ] 4.2 Escribir property test para Embedding Service
    - **Propiedad 2: Dimensión fija de embeddings**
    - Para cualquier texto no nulo, `len(service.encode(text)) == EMBEDDING_DIM`
    - **Valida: Requisito 2.1**
  - [ ] 4.3 Escribir property test para null-handling
    - **Propiedad 3: Vector cero para inputs nulos**
    - `service.encode(None)` y `service.encode("")` producen `np.zeros(EMBEDDING_DIM)`
    - **Valida: Requisito 2.4**
  - [ ] 4.4 Escribir tests unitarios para EmbeddingService
    - Verificar que `encode_batch` produce la misma forma que N llamadas a `encode`
    - Verificar que el modelo se carga una sola vez (singleton)
    - _Requisitos: 2.1, 2.4_

- [x] 5. Checkpoint — Verificar que todos los tests pasan hasta aquí
  - Ejecutar `pytest Codigo/tests/ -x --tb=short` y confirmar que los 147 tests
    existentes siguen pasando junto con los nuevos. Consultar al usuario si hay dudas.

- [ ] 6. Extensión del Feature Pipeline
  - [x] 6.1 Implementar `Codigo/src/features/embedding_pipeline.py`
    - Función `build_text_fields(row: dict) -> dict[str, str]` que extrae
      `diagnostico_texto`, `cups_texto`, `observaciones` de un registro
    - Función `enrich_features(numeric_features: np.ndarray, text_fields: dict[str, str],
      embedding_service: EmbeddingService, nlp_pipeline: spacy.Language) -> np.ndarray`
      que concatena embeddings al vector numérico de 48 features
    - Función `generate_split_embeddings(df: pd.DataFrame, split_col: str,
      embedding_service: EmbeddingService) -> np.ndarray` que genera embeddings
      respetando la división temporal (sin shuffle, ordenado por `fecha_radicacion`)
    - Serialización de embeddings en formato NPZ con `np.savez_compressed`
    - `RANDOM_STATE = 42` en cualquier operación con aleatoriedad
    - _Requisitos: 2.2, 2.5, 2.6, 8.1, 8.3, 8.4_
  - [ ] 6.2 Escribir property test para Feature Pipeline
    - **Propiedad 4: Dimensión del vector enriquecido**
    - `len(enrich_features(...)) == 48 + EMBEDDING_DIM * n_text_fields`
    - **Valida: Requisito 2.2**
  - [ ] 6.3 Escribir property test para preservación de split temporal
    - **Propiedad 5: No data leakage temporal**
    - Los embeddings generados para el conjunto de test no contienen información
      de registros con `fecha_radicacion` anterior al corte de test
    - **Valida: Requisito 2.5, 8.3**
  - [ ] 6.4 Escribir tests unitarios para embedding_pipeline
    - Verificar que `enrich_features` con todos los campos nulos produce vector de
      longitud `48 + EMBEDDING_DIM * n_text_fields` (relleno con ceros)
    - Verificar que el orden temporal se preserva en `generate_split_embeddings`
    - _Requisitos: 2.4, 2.5_

- [ ] 7. Evaluador de candidatos LLM
  - [x] 7.1 Implementar `Codigo/src/evaluation/llm_evaluator.py`
    - Clase `LLMEvaluator` con método `evaluate_candidate(model_name: str,
      X_test: np.ndarray, y_test: np.ndarray) -> dict` que retorna métricas:
      `recall_glosada`, `f1_macro`, `auc_roc`, `recall_per_class`, `confusion_matrix`
    - Método `measure_embedding_separability(embeddings: np.ndarray,
      labels: np.ndarray) -> float` que calcula separabilidad semántica entre
      las 3 clases (Auditada/Glosada/Devuelta)
    - Método `measure_latency(model_name: str, sample_size: int = 1000) -> dict`
      que retorna `p50_ms` y `p95_ms`
    - Método `run_full_evaluation(candidates: list[str]) -> dict` que evalúa todos
      los candidatos y produce reporte JSON con recomendación justificada
    - Lógica de resultado negativo: si ningún candidato supera Recall Glosada 0.7261,
      documentar y recomendar continuar con pipeline base
    - Serializar reporte en `Codigo/reports/llm_evaluation_{DATE_PREFIX}.json`
    - _Requisitos: 3.1, 3.2, 3.4, 3.5, 3.6, 6.1–6.7_
  - [-] 7.2 Implementar reentrenamiento XGBoost con features enriquecidas
    - Script `Codigo/src/evaluation/retrain_xgboost.py` que:
      carga train/val/test parquet, genera embeddings con `EmbeddingService`,
      concatena al vector de 48 features, reentrena XGBoost con `RANDOM_STATE = 42`,
      evalúa en test y OOT, serializa modelo en `Codigo/models/artifacts/` con prefijo
      de fecha y sufijo `_v2_enriched.json`
    - Mantener tarea multiclase (3 clases) siempre
    - _Requisitos: 2.6, 2.7, 6.1–6.7, 8.1, 8.3, 8.4_
  - [ ] 7.3 Escribir tests unitarios para LLMEvaluator
    - Verificar que `run_full_evaluation` produce JSON con todas las claves requeridas
    - Verificar que el resultado negativo se documenta correctamente
    - _Requisitos: 3.5, 3.6_

- [ ] 8. Generador de Explicaciones
  - [ ] 8.1 Implementar `Codigo/src/explainability/explanation_generator.py`
    - Clase `ExplanationGenerator` con `__init__(privacy_guard: PrivacyGuard,
      llm_model: str, timeout_s: float = EXPLANATION_MAX_SECONDS)`
    - Método `generate(prediction: int, confidence: float,
      shap_values: dict, feature_values: dict) -> str` que produce narrativa en español
      con: (a) predicción + confianza, (b) top-3 factores de riesgo SHAP en lenguaje
      auditor (sin mencionar "SHAP", "embedding", "feature importance"),
      (c) causal Res. 2284/2023 más relevante de `FEATURE_TO_CAUSAL`
    - Degradación graciosa: si Ollama no disponible, retornar explicación SHAP técnica
      existente de `shap_explanations.py`
    - Verificar que la generación completa en < `EXPLANATION_MAX_SECONDS` (p95)
    - Usar `logging`, type hints y docstrings Google-style
    - _Requisitos: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_
  - [ ] 8.2 Escribir property test para Explanation Generator
    - **Propiedad 6: Degradación graciosa siempre retorna string no vacío**
    - Para cualquier combinación de predicción/SHAP values, `generate(...)` retorna
      `str` no vacío incluso cuando Ollama no está disponible
    - **Valida: Requisito 4.5**
  - [ ] 8.3 Escribir tests unitarios para ExplanationGenerator
    - Verificar que la narrativa para clase Glosada/Devuelta incluye al menos un
      causal de Res. 2284 (SO0101, SO0201, FA0101, FA0301, TA0101, DE16, DE01)
    - Verificar que la narrativa no contiene términos técnicos de ML
    - Verificar que el fallback SHAP se activa cuando Ollama no responde
    - _Requisitos: 4.3, 4.4, 4.5_

- [ ] 9. Monitoreo de Concept Drift con PSI
  - [ ] 9.1 Implementar `Codigo/src/monitoring/drift_detector.py`
    - Función `compute_psi(reference: np.ndarray, current: np.ndarray,
      n_bins: int = 10) -> float` que calcula Population Stability Index
    - Función `compute_embedding_psi(ref_embeddings: np.ndarray,
      cur_embeddings: np.ndarray) -> dict[str, float]` que calcula PSI por dimensión
      reducida (PCA a 10 componentes) y retorna `{"mean_psi": float, "max_psi": float}`
    - Función `check_drift(psi_value: float) -> str` que retorna `"stable"`,
      `"warning"` (PSI > 0.15) o `"alert"` (PSI > 0.20)
    - _Requisitos: 7.1, 7.3, 7.4_
  - [ ] 9.2 Escribir property test para PSI
    - **Propiedad 7: PSI es cero para distribuciones idénticas**
    - `compute_psi(arr, arr) == 0.0` para cualquier array de referencia
    - **Valida: Requisito 7.3**
  - [ ] 9.3 Escribir tests unitarios para drift_detector
    - Verificar que PSI > 0.20 activa nivel `"alert"`
    - Verificar que distribuciones idénticas producen PSI ≈ 0
    - _Requisitos: 7.3, 7.4_

- [ ] 10. Integración con la API FastAPI
  - [ ] 10.1 Extender schemas en `Codigo/src/api/schemas.py`
    - Agregar campos opcionales a `PredictionRequest`:
      `diagnostico_texto: str | None = None`,
      `cups_texto: str | None = None`,
      `observaciones: str | None = None`
    - Agregar campo opcional a `PredictionResponse`:
      `explicacion_enriquecida: str | None = None`,
      `drift_warning: str | None = None`
    - Crear schema `LLMStatusResponse` con campos:
      `ollama_available: bool`, `active_model: str | None`,
      `embedding_model: str`, `latency_p50_ms: float | None`,
      `latency_p95_ms: float | None`
    - No modificar campos existentes (compatibilidad hacia atrás)
    - _Requisitos: 5.1, 5.2, 5.5, 5.6, 7.4_
  - [ ] 10.2 Extender `Codigo/src/api/main.py`
    - Inicializar `PrivacyGuard`, `EmbeddingService`, `ExplanationGenerator` y
      `DriftDetector` en el lifespan; retornar HTTP 503 si Ollama no disponible
      sin degradar el endpoint XGBoost base
    - Modificar `POST /v1/predict` para: si campos textuales presentes → enriquecer
      features con `EmbeddingService`; si ausentes → pipeline base sin cambios
    - Agregar `GET /v1/llm/status` que retorna `LLMStatusResponse`
    - Incluir `drift_warning` en response cuando PSI de embeddings > 0.20
    - _Requisitos: 1.4, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 7.4_
  - [ ] 10.3 Escribir tests de integración para nuevos endpoints
    - Verificar que `POST /v1/predict` sin campos textuales retorna response idéntico
      al comportamiento actual (compatibilidad hacia atrás)
    - Verificar que `GET /v1/llm/status` retorna 200 con schema correcto
    - Verificar que HTTP 503 se retorna cuando Ollama no está disponible al inicio
    - _Requisitos: 5.1, 5.4, 5.5, 1.4_

- [ ] 11. Checkpoint — Verificar cobertura y compatibilidad
  - Ejecutar `pytest Codigo/tests/ --cov=Codigo/src --cov-report=term-missing`
    y verificar que la cobertura total es ≥ 80% y que los 147 tests existentes
    siguen pasando. Consultar al usuario si hay dudas.

- [ ] 12. Reproducibilidad y artefactos experimentales
  - [ ] 12.1 Implementar `Codigo/src/evaluation/experiment_logger.py`
    - Clase `ExperimentLogger` con método `log_run(model_name: str,
      hyperparams: dict, metrics: dict) -> None` que escribe en
      `Codigo/reports/experiment_log_{DATE_PREFIX}.json` en formato JSON Lines
    - Incluir: timestamp ISO-8601, versión del modelo LLM, hiperparámetros,
      métricas completas (F1-macro, Recall Glosada, AUC-ROC, Matriz de Confusión)
    - _Requisitos: 8.2, 8.5_
  - [ ] 12.2 Verificar determinismo del pipeline completo
    - Agregar test que ejecuta el pipeline de evaluación dos veces con los mismos
      datos y verifica que las métricas son idénticas (`RANDOM_STATE = 42`,
      división temporal fija)
    - _Requisitos: 8.5_
  - [ ] 12.3 Escribir property test para ExperimentLogger
    - **Propiedad 8: Idempotencia de serialización JSON**
    - Serializar y deserializar un `dict` de métricas produce el mismo `dict`
    - **Valida: Requisito 8.2**

- [ ] 13. Checkpoint final — Todos los tests pasan
  - Ejecutar `pytest Codigo/tests/ -v --tb=short` y confirmar suite completa verde.
    Verificar cobertura ≥ 80%. Consultar al usuario si hay dudas.

- [ ] 14. Demo MVP con Interfaz Gráfica Atractiva
  - [ ] 14.1 Implementar interfaz web con Streamlit
    - Crear `Codigo/demo/streamlit_app.py` con interfaz moderna y atractiva
    - Diseño responsive con sidebar para configuración y área principal para resultados
    - Tema oscuro/claro configurable, colores corporativos de Savia Salud EPS
    - Componentes: formulario de entrada, visualización de predicciones, explicaciones enriquecidas
    - Métricas en tiempo real: latencia, confianza, drift warnings
    - _Requisitos: 5.1, 5.2, 4.1, 4.2_
  - [ ] 14.2 Integración con pipeline completo
    - Conectar con API FastAPI existente o directamente con modelos
    - Carga de modelos XGBoost base y enriquecido (si disponible)
    - Inicialización de EmbeddingService, PrivacyGuard y ExplanationGenerator
    - Manejo de errores gracioso con mensajes informativos para el usuario
    - _Requisitos: 1.4, 2.1, 4.5_
  - [ ] 14.3 Funcionalidades de demostración
    - **Entrada de datos:** Formulario intuitivo para ingresar datos de factura
      (48 features numéricas + campos textuales opcionales)
    - **Predicción comparativa:** Mostrar lado a lado resultados del modelo base vs enriquecido
    - **Explicaciones enriquecidas:** Narrativas en lenguaje natural contextualizadas
    - **Visualizaciones:** Gráficos de confianza, importancia de features, métricas de separabilidad
    - **Estado del sistema:** Panel de monitoreo con estado de Ollama, latencias, drift warnings
    - _Requisitos: 3.1, 4.1, 4.2, 7.4_
  - [ ] 14.4 Elementos visuales atractivos
    - **Dashboard moderno:** Cards con métricas, progress bars, indicadores de estado
    - **Gráficos interactivos:** Plotly para visualizaciones dinámicas (SHAP, embeddings, métricas)
    - **Animaciones sutiles:** Transiciones suaves, loading spinners, efectos hover
    - **Iconografía médica:** Iconos relacionados con auditoría, salud, análisis
    - **Paleta de colores:** Verde/azul corporativo con acentos para alertas y éxitos
    - _Requisitos: UX/UI atractiva_
  - [ ] 14.5 Casos de uso demostrativos
    - **Caso 1:** Factura típica sin campos textuales (modelo base)
    - **Caso 2:** Factura con diagnóstico CIE-10 complejo (enriquecimiento semántico)
    - **Caso 3:** Factura con observaciones clínicas detalladas (explicación enriquecida)
    - **Caso 4:** Simulación de concept drift (alertas PSI)
    - Datos sintéticos realistas basados en patrones del dataset real
    - _Requisitos: 2.2, 4.3, 7.4_
  - [ ] 14.6 Documentación y deployment del demo
    - README específico para el demo en `Codigo/demo/README.md`
    - Instrucciones de instalación y ejecución paso a paso
    - Screenshots de la interfaz y casos de uso
    - Dockerfile específico para el demo con todas las dependencias
    - Script de inicialización que verifica Ollama y descarga modelos necesarios
    - _Requisitos: Documentación y deployment_

## Notas

- Las tareas marcadas con `*` son opcionales y pueden omitirse para un MVP más rápido
- `RANDOM_STATE = 42` en todos los componentes con aleatoriedad
- La división temporal nunca es aleatoria — siempre por `fecha_radicacion`
- La tarea es siempre multiclase (3 clases): Auditada / Glosada / Devuelta
- No usar `print()` en `src/` — usar `logging`
- Todas las funciones públicas requieren type hints y docstrings Google-style
- No realizar llamadas a APIs externas — toda inferencia es local
