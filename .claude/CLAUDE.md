# CLAUDE.MD — Proyecto: Analítica Predictiva para Auditoría Prestacional
## Savia Salud EPS · Universidad EAFIT · Ciencia de Datos

> **Propósito de este archivo:** Índice maestro y estado actual del proyecto. La documentación completa está modularizada en archivos temáticos en `.claude/` para optimizar el uso de la ventana de contexto. Carga solo los módulos necesarios según la tarea actual.

## ARQUITECTURA DE DOCUMENTACIÓN MODULAR

Este archivo maestro contiene solo el estado actual y referencias a módulos. Para trabajar en una tarea específica, carga el módulo correspondiente usando `#File` en el chat:

| Módulo | Archivo | Contenido | Cuándo cargar |
|---|---|---|---|
| **01** | `01_contexto_proyecto.md` | Contexto general, objetivos, problema, glosario | Inicio de sesión, contexto general |
| **02** | `02_base_datos.md` | Estructura BD, tablas, FKs, hallazgos de calidad | Trabajo con BD, queries, extracción de datos |
| **03** | `03_dataset_features.md` | Dataset, columnas, feature engineering, splits | Feature engineering, transformaciones |
| **04** | `04_modelo_metricas.md` | Entrenamiento, métricas, SHAP, validación | Modelado, evaluación, explicabilidad |
| **05** | `05_api_arquitectura.md` | API REST, endpoints, arquitectura AWS | Desarrollo API, despliegue, infraestructura |
| **06** | `06_testing_suite.md` | Suite de testing, fixtures, estrategia | Testing, validación de código |
| **07** | `07_reglas_codificacion.md` | Reglas de código, convenciones, instrucciones IA | Siempre cargar al inicio de sesión |
| **08** | `08_taxonomia_glosas.md` | Causales Res. 2284, reglas de validación | Trabajo con glosas, causales, validaciones |
| **09** | `09_historial_sesiones.md` | Log de sesiones por fecha | Revisar trabajo previo, continuidad |
| **10** | `10_integracion_llm_local.md` | **Integración LLM local, embeddings, Fase 8** | **Trabajo con LLM, embeddings, enriquecimiento semántico** |

**REGLA DE USO:** Carga solo los módulos necesarios para tu tarea actual. No cargues todos los módulos a la vez.

---

## 0. ESTADO ACTUAL DEL PROYECTO *(Actualizado: 2026-04-09)*

### Avance por Fase

| Fase | Nombre | Estado | Avance |
|---|---|---|---|
| **0** | Saneamiento y Entorno | ✅ Completada | 100% |
| **1A** | Mapeo BD + Confirmación de fuentes | ✅ Completada | 100% |
| **1B** | Distribución de clases confirmada | ✅ Completada | 100% |
| **1C** | Extracción y corrección dataset entrenamiento | ✅ Completada | 100% |
| **1D** | Parsers JSON RIPS + XML FEV | ⏳ Pendiente | 0% |
| **2** | EDA | ✅ Completada | 100% |
| **3** | Feature Engineering | ✅ Completada (right-censoring corregido y verificado) | 100% |
| **4** | Entrenamiento del Modelo | ✅ Completada (v1 — recall Glosada pendiente de optimizar) | 100% |
| **5** | Explicabilidad (XAI) | ✅ Completada | 100% |
| **5B** | Validación contra Datos Reales | ✅ Completada (con hallazgos críticos sobre IA Previa) | 100% |
| **6** | API REST | ✅ Completada | 100% |
| **6B** | Suite de Testing de Validación | ✅ Completada (147 tests, 96.85% cobertura) | 100% |
| **7** | Tesis (main.tex) | 🔄 **En progreso** | ~90% |
| **8** | **Integración LLM Local** | 🔄 **En progreso** | **44%** |

### Resumen Ejecutivo

**Modelo XGBoost v1:**
- F1-macro: 0.7490
- AUC-ROC: 0.9078
- Recall Glosada: 0.7261 (meta: 0.80)
- Dataset: 633K facturas (mar-2024 a jul-2025)
- Features: 48 variables (28 originales + 13 log + 4 flags + 6 interacciones)

**Integración LLM Local (Fase 8 - En Progreso):**
- Objetivo: Superar Recall Glosada ≥ 0.80 mediante enriquecimiento semántico
- Tecnologías: Ollama + Sentence Transformers + spaCy (100% local, Ley 1581)
- Progreso: 7/16 tareas completadas (44%)
- Componentes implementados: PrivacyGuard, EmbeddingService, LLMEvaluator, EmbeddingPipeline
- Candidatos LLM: Llama 3.2, Mistral 7B, Qwen2.5, Phi-3 Mini
- **Sistema de reentrenamiento:** Automático mensual con detección de drift
- Demo MVP: Interfaz gráfica atractiva con Streamlit (pendiente)
- Spec: `.kiro/specs/local-llm-integration/` (design-first workflow)

**API REST:**
- 4 endpoints FastAPI (/health, /model/info, /predict, /predict/batch)
- Explicabilidad SHAP integrada en tiempo real
- Pydantic v2 schemas con 48 features
- Dockerfile con healthcheck

**Testing Suite (completada):**
- 147 tests en 10 archivos (unit, integration, performance, security, regression, docker)
- Cobertura: 96.85% (meta: 80%)
- Ejecución: 21.76s (meta: <5min)
- CI/CD: GitHub Actions con 3 jobs (unit, integration, coverage)

### Próximos Pasos

1. **Fase 8 (Integración LLM Local):** Continuar implementación (44% completada)
   - Completar tests unitarios y property tests para componentes implementados
   - Finalizar reentrenamiento XGBoost con features enriquecidas
   - Implementar generador de explicaciones en lenguaje natural
   - **Sistema de reentrenamiento automático:** Programación mensual con detección de drift
   - Integrar con API FastAPI manteniendo compatibilidad hacia atrás
   - **Demo MVP:** Crear interfaz gráfica atractiva con Streamlit para demostración
   - Spec completa disponible en `.kiro/specs/local-llm-integration/tasks.md`

2. **Fase 7:** Pulir tesis main.tex (~10% restante)
   - Revisar referencias cruzadas y bibliografía
   - Verificar formato de tablas y figuras
   - Revisión final de arquitectura AWS

3. **Fase 1D (opcional):** Parsers JSON RIPS + XML FEV para producción

---

## 1. ESTRUCTURA DEL REPOSITORIO

```
Codigo/
├── .claude/                   ← Documentación modular del proyecto
│   ├── CLAUDE.MD              ← (este archivo) Índice maestro
│   ├── README.md              ← Guía de uso de la documentación modular
│   ├── 01_contexto_proyecto.md
│   ├── 02_base_datos.md
│   ├── 03_dataset_features.md
│   ├── 04_modelo_metricas.md
│   ├── 05_api_arquitectura.md
│   ├── 06_testing_suite.md
│   ├── 07_reglas_codificacion.md
│   ├── 08_taxonomia_glosas.md
│   └── 09_historial_sesiones.md
│
├── .kiro/specs/               ← Specs de features (metodología spec-driven)
│   ├── api-testing-suite/    ← Spec de suite de testing de validación
│   └── local-llm-integration/ ← Spec de integración LLM local (Fase 8)
│
├── data/
│   ├── raw/                   ← Datos ORIGINALES e INMUTABLES
│   ├── database_analysis/     ← Muestras y análisis de BD
│   ├── Info_data_base/        ← Metadata de BD
│   ├── interim/               ← Datos en proceso de transformación
│   └── processed/             ← Dataset final (Parquet)
│
├── notebooks/                 ← 9 notebooks (00-08)
├── src/                       ← Módulos Python refactorizados (Fase 6)
├── tests/                     ← Suite de testing (Fase 6B)
├── models/artifacts/          ← Modelos serializados
├── reports/                   ← Figuras y evaluaciones
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 2. REGLAS CRÍTICAS (SIEMPRE ACTIVAS)

> **⚠️ ESTAS REGLAS SON OBLIGATORIAS EN TODA SESIÓN**

### Seguridad de Datos

1. **BD solo lectura:** NUNCA generar código con `INSERT`, `UPDATE`, `DELETE`, `DROP`, `TRUNCATE`, `ALTER`, `CREATE TABLE`
2. **data/raw/ es sagrado:** NUNCA modificar, sobreescribir o eliminar archivos en `data/raw/`
3. **Anonimización:** NUNCA incluir `nombre_completo_afiliado` ni datos PII en features, logs o notebooks

### Calidad de Código

1. **Type hints obligatorios:** Toda función debe tener type hints completos
2. **Docstrings Google Style:** Obligatorio en toda función pública
3. **No print() en src/:** Usar `logging.getLogger(__name__)`
4. **Tests obligatorios:** Cobertura mínima 80% para módulos en `src/`
5. **Reproducibilidad:** `RANDOM_STATE = 42` en todos los modelos

### Modelado

1. **Tarea multiclase:** NUNCA simplificar a binaria sin aprobación
2. **Métrica primaria:** Recall de clase Glosada (clase=1)
3. **Métricas obligatorias:** F1-macro, AUC-ROC, Recall por clase, Matriz de Confusión
4. **División temporal:** NUNCA usar split aleatorio (prevenir data leakage)

---

## 3. GUÍA RÁPIDA DE USO

### Para iniciar una sesión de trabajo:

1. **Siempre cargar:** `#File .claude/07_reglas_codificacion.md`
2. **Según la tarea, cargar:**
   - Trabajo con BD → `#File .claude/02_base_datos.md`
   - Feature engineering → `#File .claude/03_dataset_features.md`
   - Modelado/evaluación → `#File .claude/04_modelo_metricas.md`
   - API/despliegue → `#File .claude/05_api_arquitectura.md`
   - Testing → `#File .claude/06_testing_suite.md`
   - Glosas/causales → `#File .claude/08_taxonomia_glosas.md`
3. **Para contexto general:** `#File .claude/01_contexto_proyecto.md`
4. **Para revisar historial:** `#File .claude/09_historial_sesiones.md`

### Para actualizar documentación:

- **Estado del proyecto:** Actualizar §0 en este archivo (CLAUDE.MD)
- **Historial de sesión:** Agregar entrada en `09_historial_sesiones.md`
- **Cambios técnicos:** Actualizar el módulo temático correspondiente

---

## 4. INFORMACIÓN DE CONTACTO Y VERSIÓN

| Atributo | Valor |
|---|---|
| **Última actualización** | 2026-04-09 |
| **Autor** | jcardonr |
| **Proyecto** | EAFIT — Maestría en Ciencia de Datos |
| **Organización** | Savia Salud EPS |
| **Repositorio** | `d:\Users\jcardonr\Documents\eafit\Codigo\` |
| **Versión documentación** | 2.1 (+ Fase 8 LLM Local) |

---

*Para más detalles sobre cualquier aspecto del proyecto, consulta los módulos temáticos en `.claude/`*
