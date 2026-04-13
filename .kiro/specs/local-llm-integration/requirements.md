# Documento de Requisitos: Integración LLM Local

## Introducción

Este documento define los requisitos funcionales y no funcionales para la integración de un modelo de lenguaje local (LLM) en el pipeline de auditoría prestacional de Savia Salud EPS. La integración complementa el modelo XGBoost v1 existente con dos capacidades: (1) enriquecimiento semántico del vector de features mediante embeddings de campos textuales (diagnósticos CIE-10, códigos CUPS, observaciones clínicas), y (2) generación de explicaciones en lenguaje natural contextualizadas con causales de la Resolución 2284/2023 para auditores médicos.

El objetivo primario es superar la meta de Recall Glosada ≥ 0.80 (actual: 0.7261) sin comprometer el F1-macro ni la trazabilidad regulatoria. Toda la infraestructura opera estrictamente en local para cumplir con la Ley 1581 de Colombia — los datos sensibles de salud nunca pueden salir de la infraestructura propia de Savia Salud EPS.

## Glosario

- **LLM_Local**: Modelo de lenguaje grande ejecutado localmente mediante Ollama (candidatos: Llama 3.2, Mistral 7B, Qwen2.5, Phi-3 Mini)
- **Embedding_Service**: Componente basado en Sentence Transformers que genera representaciones vectoriales densas de campos textuales
- **NLP_Pipeline**: Componente spaCy en español para preprocesamiento y normalización de texto clínico
- **XGBoost_Model**: Modelo de clasificación multiclase XGBoost v1 (irremplazable como modelo base)
- **Feature_Pipeline**: Pipeline de ingeniería de características que produce el vector de 48 features numéricas
- **Explanation_Generator**: Componente que produce narrativas en lenguaje natural para auditores médicos
- **Audit_API**: API FastAPI existente con 4 endpoints de predicción y explicabilidad
- **Evaluator**: Componente de evaluación experimental que compara candidatos LLM en métricas de negocio
- **Privacy_Guard**: Capa de validación que garantiza que ningún dato sensible salga de la infraestructura local
- **CIE-10**: Clasificación Internacional de Enfermedades, 10ª revisión (campo textual de diagnósticos)
- **CUPS**: Clasificación Única de Procedimientos en Salud (campo textual de procedimientos)
- **Res_2284**: Resolución 2284/2023 — Manual Único de Glosas del sistema de salud colombiano
- **Recall_Glosada**: Métrica primaria del proyecto — fracción de facturas glosadas correctamente identificadas
- **Concept_Drift**: Degradación del rendimiento del modelo por cambio en la distribución de datos (detectado en OOT ago-dic 2025: F1 cae de 0.74 a 0.59)
- **OOT**: Out-of-Time — conjunto de validación temporal posterior al periodo de entrenamiento

---

## Requisitos

### Requisito 1: Infraestructura LLM Local y Cumplimiento Ley 1581

**Historia de usuario:** Como responsable de cumplimiento de Savia Salud EPS, quiero que todos los modelos de lenguaje operen exclusivamente en infraestructura propia, para garantizar que los datos sensibles de salud nunca salgan de los servidores de la organización en cumplimiento de la Ley 1581 de Colombia.

#### Criterios de Aceptación

1. THE Privacy_Guard SHALL ejecutar todos los modelos LLM y de embeddings exclusivamente en infraestructura local mediante Ollama y Sentence Transformers, sin realizar llamadas a APIs externas de ningún tipo
2. THE Privacy_Guard SHALL bloquear cualquier intento de transmisión de datos de pacientes, diagnósticos CIE-10, códigos CUPS u observaciones clínicas hacia servicios externos
3. WHEN el sistema inicia, THE Privacy_Guard SHALL verificar que el servidor Ollama esté disponible en localhost y que los modelos requeridos estén descargados localmente antes de aceptar solicitudes
4. IF el servidor Ollama no está disponible en el inicio, THEN THE Audit_API SHALL retornar un error de configuración con código HTTP 503 e instrucciones de remediación, sin degradar el servicio de predicción XGBoost base
5. THE Privacy_Guard SHALL registrar en logs locales toda operación de inferencia LLM con timestamp, modelo utilizado y hash del input, sin almacenar el contenido textual de los datos clínicos en logs externos

---

### Requisito 2: Enriquecimiento de Features con Embeddings Semánticos

**Historia de usuario:** Como científico de datos, quiero incorporar representaciones semánticas de los campos textuales de las facturas (diagnósticos CIE-10, códigos CUPS, observaciones) al vector de features del modelo XGBoost, para capturar información semántica que las 48 features numéricas actuales no representan y mejorar el Recall Glosada.

#### Criterios de Aceptación

1. WHEN el Embedding_Service recibe un campo textual de diagnóstico CIE-10 o código CUPS, THE Embedding_Service SHALL producir un vector de embeddings de dimensión fija mediante un modelo Sentence Transformers ejecutado localmente
2. THE Feature_Pipeline SHALL concatenar los embeddings semánticos al vector de 48 features numéricas existentes para producir un vector enriquecido como entrada al XGBoost_Model
3. THE NLP_Pipeline SHALL preprocesar los campos textuales en español aplicando normalización, eliminación de stopwords médicos y lematización antes de generar embeddings
4. WHEN el campo textual de diagnóstico o CUPS está ausente o es nulo, THE Embedding_Service SHALL sustituirlo por un vector de ceros de la misma dimensión fija, sin interrumpir el pipeline
5. THE Feature_Pipeline SHALL mantener la división temporal obligatoria (no aleatoria) al generar embeddings para los conjuntos de entrenamiento, validación y test, previniendo data leakage
6. THE XGBoost_Model SHALL conservar RANDOM_STATE = 42 en todos los experimentos de reentrenamiento con features enriquecidas
7. WHEN se entrena el XGBoost_Model con features enriquecidas, THE Evaluator SHALL reportar F1-macro, AUC-ROC, Recall por clase y Matriz de Confusión comparando el modelo base (48 features) contra el modelo enriquecido

---

### Requisito 3: Evaluación Experimental de Modelos LLM Candidatos

**Historia de usuario:** Como científico de datos, quiero evaluar sistemáticamente múltiples modelos LLM candidatos (Llama 3.2, Mistral 7B, Qwen2.5, Phi-3 Mini) antes de seleccionar uno para producción, para tomar una decisión basada en evidencia sobre calidad de embeddings, calidad de explicaciones y viabilidad operacional.

#### Criterios de Aceptación

1. THE Evaluator SHALL evaluar cada modelo candidato (Llama 3.2, Mistral 7B, Qwen2.5, Phi-3 Mini) en las mismas métricas de negocio: Recall Glosada, F1-macro, AUC-ROC y Recall por clase
2. WHEN se evalúa la calidad de embeddings de un modelo candidato, THE Evaluator SHALL medir la separabilidad semántica entre clases (Auditada/Glosada/Devuelta) usando los embeddings generados sobre el conjunto de test
3. WHEN se evalúa la calidad de explicaciones de un modelo candidato, THE Evaluator SHALL medir coherencia con causales Res_2284 verificando que las narrativas generadas referencien al menos uno de los códigos relevantes (SO0101, SO0201, FA0101, FA0301, TA0101, DE16, DE01) cuando el modelo XGBoost predice Glosada o Devuelta
4. THE Evaluator SHALL medir la latencia de inferencia de cada modelo candidato en percentil p50 y p95 sobre una muestra representativa de 1,000 facturas
5. THE Evaluator SHALL producir un reporte comparativo en formato JSON con todas las métricas por modelo candidato, incluyendo la recomendación del modelo seleccionado con justificación cuantitativa
6. IF ningún modelo candidato supera el Recall Glosada del modelo base XGBoost (0.7261) con features enriquecidas, THEN THE Evaluator SHALL documentar el resultado negativo y recomendar continuar con el pipeline XGBoost base sin integración LLM

---

### Requisito 4: Generación de Explicaciones en Lenguaje Natural

**Historia de usuario:** Como auditor médico de Savia Salud EPS, quiero recibir explicaciones de las predicciones en lenguaje natural contextualizado con causales de la Resolución 2284/2023, para entender rápidamente por qué una factura fue clasificada como Glosada o Devuelta y tomar la acción correctiva apropiada.

#### Criterios de Aceptación

1. WHEN el XGBoost_Model predice clase Glosada (1) o Devuelta (2), THE Explanation_Generator SHALL producir una narrativa en español que incluya: (a) la predicción con nivel de confianza, (b) los tres factores de riesgo principales derivados de SHAP values, y (c) el causal probable de la Res_2284 más relevante
2. WHEN el XGBoost_Model predice clase Auditada (0), THE Explanation_Generator SHALL producir una narrativa breve en español confirmando el cumplimiento y los factores que respaldan la predicción
3. THE Explanation_Generator SHALL referenciar causales específicos de la Res_2284 (SO0101, SO0201, FA0101, FA0301, TA0101, DE16, DE01) en las explicaciones cuando los SHAP values de las features correspondientes superen el umbral de importancia definido en la configuración
4. THE Explanation_Generator SHALL producir explicaciones en lenguaje comprensible para auditores médicos sin conocimiento técnico de machine learning, evitando términos como "SHAP value", "feature importance" o "embedding"
5. WHEN el LLM_Local no está disponible, THE Explanation_Generator SHALL degradar graciosamente retornando las explicaciones SHAP técnicas existentes del sistema base, sin interrumpir el endpoint de predicción
6. THE Explanation_Generator SHALL completar la generación de una explicación individual en menos de 10 segundos en el percentil p95, medido desde la recepción de los SHAP values hasta la entrega de la narrativa

---

### Requisito 5: Integración con la API FastAPI Existente

**Historia de usuario:** Como desarrollador de integración, quiero que las nuevas capacidades LLM se expongan a través de la API FastAPI existente sin romper los contratos de los 4 endpoints actuales, para que los sistemas consumidores no requieran cambios en su integración.

#### Criterios de Aceptación

1. THE Audit_API SHALL mantener compatibilidad hacia atrás con los 4 endpoints existentes (`GET /v1/health`, `GET /v1/model/info`, `POST /v1/predict`, `POST /v1/predict/batch`) sin modificar sus schemas de request ni response actuales
2. WHEN se activa el modo de enriquecimiento LLM, THE Audit_API SHALL aceptar campos textuales opcionales (`diagnostico_texto`, `cups_texto`, `observaciones`) en el schema de `POST /v1/predict` y `POST /v1/predict/batch`
3. WHEN los campos textuales opcionales están presentes en la solicitud, THE Audit_API SHALL utilizar el Embedding_Service para enriquecer el vector de features antes de invocar el XGBoost_Model
4. WHEN los campos textuales opcionales están ausentes, THE Audit_API SHALL ejecutar el pipeline de predicción base con las 48 features numéricas sin degradación de rendimiento
5. THE Audit_API SHALL exponer un nuevo endpoint `GET /v1/llm/status` que retorne el estado del servidor Ollama, el modelo LLM activo, la versión del modelo de embeddings y las métricas de latencia p50/p95 de las últimas 1,000 inferencias
6. THE Audit_API SHALL incluir en el campo `explicacion_enriquecida` del response de `POST /v1/predict` la narrativa generada por el Explanation_Generator cuando el LLM_Local esté disponible

---

### Requisito 6: Métricas de Éxito y Criterios de Aceptación del Modelo

**Historia de usuario:** Como líder técnico del proyecto, quiero criterios de aceptación cuantitativos claros para la integración LLM, para decidir objetivamente si la integración aporta valor suficiente para justificar su despliegue en producción.

#### Criterios de Aceptación

1. THE Evaluator SHALL reportar que el modelo XGBoost enriquecido con embeddings alcanza Recall Glosada ≥ 0.80 en el conjunto de test (jun-jul 2025, n=40,193) como criterio primario de aceptación
2. THE Evaluator SHALL verificar que el F1-macro del modelo enriquecido no sea inferior al F1-macro del modelo base en más de 0.02 puntos (degradación máxima permitida: F1-macro ≥ 0.7290)
3. THE Evaluator SHALL verificar que el AUC-ROC del modelo enriquecido no sea inferior al AUC-ROC del modelo base en más de 0.01 puntos (AUC-ROC ≥ 0.8978)
4. THE Evaluator SHALL evaluar el modelo enriquecido en el conjunto OOT (ago-dic 2025, n=223,621) y reportar si la degradación por concept drift se reduce respecto al modelo base (F1-macro OOT base: 0.5941)
5. THE Evaluator SHALL reportar métricas separadas para los segmentos de alto valor (facturas >5M COP) donde el modelo base ya alcanza 89-93% de accuracy, para verificar que la integración no degrada estos segmentos críticos
6. WHEN se evalúa el modelo enriquecido, THE Evaluator SHALL generar la Matriz de Confusión completa 3×3 (Auditada/Glosada/Devuelta) y reportar la reducción en el error principal: facturas Glosadas clasificadas como Auditadas (error base: 3,306 en test set)
7. THE Evaluator SHALL mantener la tarea como multiclase (3 clases) en todos los experimentos, sin simplificar a clasificación binaria

---

### Requisito 7: Gestión del Concept Drift con Embeddings

**Historia de usuario:** Como científico de datos, quiero que los embeddings semánticos ayuden a mitigar el concept drift detectado en el periodo OOT (ago-dic 2025), donde el F1-macro cae de 0.74 a 0.59, para mejorar la estabilidad del modelo en producción.

#### Criterios de Aceptación

1. WHEN se detecta concept drift (PSI > 0.15 en features numéricas clave), THE Feature_Pipeline SHALL regenerar los embeddings semánticos sobre los datos recientes para capturar cambios en la distribución de diagnósticos y procedimientos
2. THE Embedding_Service SHALL soportar actualización incremental del modelo de embeddings sin requerir reentrenamiento completo del XGBoost_Model
3. THE Evaluator SHALL medir el PSI (Population Stability Index) de los embeddings generados entre el periodo de entrenamiento (mar-2024 a jul-2025) y el periodo OOT (ago-dic 2025) para cuantificar la estabilidad semántica
4. WHEN el PSI de embeddings supera 0.20, THE Audit_API SHALL incluir una advertencia en el campo `drift_warning` del response indicando que el modelo puede estar operando fuera de su distribución de entrenamiento
5. THE Feature_Pipeline SHALL documentar en el reporte de evaluación la hipótesis de que el aumento de Glosadas en OOT (de 40.2% a 67.5%) puede estar correlacionado con cambios semánticos en los diagnósticos CIE-10 facturados

---

### Requisito 8: Reproducibilidad y Trazabilidad Experimental

**Historia de usuario:** Como investigador de tesis, quiero que todos los experimentos de integración LLM sean completamente reproducibles y trazables, para documentar el proceso experimental en la tesis de maestría y permitir la replicación de resultados.

#### Criterios de Aceptación

1. THE Feature_Pipeline SHALL usar RANDOM_STATE = 42 en todos los componentes que involucren aleatoriedad, incluyendo inicialización de modelos de embeddings, muestreo de datos de evaluación y reentrenamiento de XGBoost
2. THE Evaluator SHALL registrar en un archivo de log estructurado (JSON) la versión exacta de cada modelo LLM candidato evaluado, los hiperparámetros utilizados, las métricas obtenidas y el timestamp de cada experimento
3. THE Feature_Pipeline SHALL mantener la división temporal obligatoria (no aleatoria por `fecha_radicacion`) en todos los experimentos, con los mismos cortes de fecha: Train (mar-2024 → mar-2025), Val (abr-2025 → may-2025), Test (jun-2025 → jul-2025)
4. THE Evaluator SHALL generar artefactos reproducibles: el vector de features enriquecido serializado en formato Parquet, los embeddings generados en formato NPZ, y el modelo XGBoost reentrenado en formato JSON
5. WHEN se ejecuta el pipeline de evaluación con los mismos datos y configuración, THE Evaluator SHALL producir métricas idénticas en ejecuciones sucesivas (determinismo garantizado por RANDOM_STATE = 42 y división temporal fija)
