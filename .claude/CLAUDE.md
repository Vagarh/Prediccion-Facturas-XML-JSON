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

## 0. ESTADO ACTUAL DEL PROYECTO *(Actualizado: 2026-09-11)*

### Avance por Fase

| Fase | Nombre | Estado | Avance |
|---|---|---|---|
| **0** | Saneamiento y Entorno | ✅ Completada | 100% |
| **1A** | Mapeo BD + Confirmación de fuentes | ✅ Completada | 100% |
| **1B** | Distribución de clases confirmada | ✅ Completada | 100% |
| **1C** | Extracción y corrección dataset entrenamiento | ✅ Completada | 100% |
| **1D** | Parsers JSON RIPS + XML FEV | ⏳ Pendiente | 0% |
| **2** | EDA | ✅ Completada | 100% |
| **3** | Feature Engineering | ✅ Completada (right-censoring re-aplicado, horizonte movido a jun-2026) | 100% |
| **4** | Entrenamiento del Modelo | ✅ Completada (**v3 — LightGBM, Recall Glosada 0.8635 test / 0.8599 OOT**) | 100% |
| **5** | Explicabilidad (XAI) | ✅ Completada (SHAP refrescado para v3) | 100% |
| **5B** | Validación contra Datos Reales | ✅ Completada — drift detectado en marzo **resuelto** en esta sesión | 100% |
| **6** | API REST | ✅ Completada (recableada al campeón v3) | 100% |
| **6B** | Suite de Testing de Validación | ✅ Completada (390 tests, 100% verde) | 100% |
| **7** | Tesis (main.tex) | 🔄 **En progreso** | ~95% |
| **8** | **Integración LLM Local** | 🔄 **En progreso** | **~60%** |

### Resumen Ejecutivo

**⚠️ Sesión 2026-09-10/11 — actualización de modelo tras 5 meses sin tocar el proyecto.**
Al revisar se encontró que el "modelo v2" documentado como productivo (Recall Glosada 0.8961)
**nunca estuvo realmente cableado a la API** (`src/config.py::MODEL_PATH` seguía apuntando a v1),
y que v2 era en realidad un modelo sub-ajustado (F1 cayó de 0.7490 a 0.7114) que heredaba el
concept drift nunca resuelto (ver Fase 5B, hallazgo de marzo 2026: degradación OOT del 20.1% en
F1, Glosada 40.2%→67.5%). Se ejecutó el ciclo completo de corrección: datos frescos → suite
comparativa de modelos → selección de campeón → SHAP → API → dashboard → tesis.

**Modelo XGBoost v1 (original, mar-2026):**
- F1-macro: 0.7490 | AUC-ROC: 0.9078 | Recall Glosada: 0.7261
- Dataset: 633K facturas (mar-2024 a jul-2025)

**Modelo XGBoost v2 base (abr-2026) — nunca desplegado en la API pese a estar documentado:**
- F1-macro: 0.7114 | AUC-ROC: 0.9028 | Recall Glosada: 0.8961 (modelo sub-ajustado, no una mejora real)

**Modelo LightGBM v3 (2026-09-10/11) — CAMPEÓN ACTUAL, desplegado en la API:**
- Test (feb-mar 2026, n=79,803): F1-macro **0.7577** | AUC-ROC **0.9060** | **Recall Glosada 0.8635**
- Out-of-Time (abr-may 2026, n=74,463): F1-macro **0.7243** | **Recall Glosada 0.8599** — degradación
  relativa de solo **0.4%** (vs. 25.2% de la versión anterior en su propia validación OOT)
- Dataset fresco: **1,084,637 facturas** (mar-2024 a may-2026) vía `scripts/extract_fresh_dataset.py`
  (Auditada 19.5% / Glosada 66.1% / Devuelta 14.4% — confirma el drift: Glosada era 40% en 2024-2025)
- Horizonte de etiquetas confiable: `scripts/check_label_horizon.py` — censura desde jun-2026
  (tasa de glosa cae de ~13% a ~9-10%, devoluciones se desploman) → corte en 2026-06-01
- Splits temporales: train 842,469 (mar24-nov25) · val 87,902 (dic25-ene26) · test 79,803 (feb-mar26)
  · oot 74,463 (abr-may26) — `data/processed/20260910_{train,val,test,oot}.parquet`
- Suite comparativa (`src/training/model_suite.py`, Optuna): xgb_v2_repro, XGBoost/LightGBM/CatBoost
  afinados, TabM, AutoGluon (parcial — interrumpido por una suspensión del equipo antes de completar
  todos los candidatos de `best_quality`; solo entrenó 3 de ~8-10 modelos base)
- **TabPFN v2 excluido:** exige aceptar licencia del modelo vía login interactivo/HF (browser auth),
  incompatible con ejecución automatizada y con el requisito 100% local (Ley 1581)
- Campeón elegido por empate técnico en F1 (XGBoost afinado 0.7595 vs. LightGBM afinado 0.7577,
  Δ=0.0018, dentro del margen de ruido) desempatado por Recall Glosada OOT — ver
  `src/training/evaluation.py::select_champion` (regla de desempate documentada en código)
- Artefacto: `models/artifacts/20260910_lgbm_tuned_champion.joblib` (+ `_metadata.json`)
- Reportes: `reports/evaluations/20260910_model_suite.json`, `reports/20260910_model_suite_comparison.md`
- Drift real medido (PSI, dataset viejo vs. fresco): `reports/20260911_drift_report.json` —
  mean_psi=0.247 (alert), top features: `total_facturas_nit` (PSI=3.30), `tiene_fev` (1.38),
  `dias_radicacion_promedio_nit` (0.94)
- SHAP refrescado: `reports/evaluations/20260910_shap_importance.json` + 4 figuras
  `reports/figures/shap_*_2026.png` — top-5 consistente con v1 (misma feature #1), confirma que
  la mejora vino de datos/tuning, no de un cambio en la lógica de negocio
- Modelo enriquecido con embeddings (`xgb_enriched`, Fase 8): en curso al cierre de esta sesión —
  ver `reports/evaluations/20260910_model_suite_enriched.json` para el resultado final

**Cableado a producción (esta sesión):**
- `src/config.py`: `MODEL_PATH`, `FEATURE_LIST_PATH`, `MODEL_VERSION=v3`, `MODEL_DATE`,
  `MODEL_ALGORITHM=LightGBM` (constante nueva), `MODEL_METRICS` — todos actualizados
- `src/models/predict.py::load_model`: ahora soporta `.joblib` genérico además de XGBoost-JSON nativo
  (detección por extensión) — necesario porque el campeón ya no es XGBoost
- `src/training/retraining_scheduler.py`: piso de Recall Glosada por defecto corregido de 0.7261
  (métrica de v1, heredada por error) a 0.80 (la meta operativa real del proyecto)
- Dashboard (`web-next/`): Hero, Resultados (reenfocado en estabilidad OOT, no solo el número),
  Drift (PSI y fechas reales), Solucion, Ventajas, Arquitectura, Explicabilidad (SHAP real), Nav,
  Footer y meta de `layout.tsx` — todos actualizados; build verificado sin errores

**Capa de datos del dashboard (refactor posterior):**
- `scripts/sync_dashboard_data.py` consolida `reports/**` + los parquet de splits en
  `web-next/data/metrics.json` (versionado). **Regla: ninguna cifra del sitio se escribe a mano**;
  si cambia un número se reentrena y se re-ejecuta el script (`npm run sync:metrics`).
  Necesario porque `/Codigo/reports/` está gitignored y Vercel construye desde GitHub.
- `web-next/lib/metrics.ts`: tipos + accesores (`pct`, `num`, `compact`, `duration`, `rangeLabel`).
  Los componentes de la landing ya no hardcodean métricas.
- Ruta nueva `/experimentos`: dataset y splits temporales, suite comparativa de 6 modelos con la
  regla de desempate, matriz de confusión e hiperparámetros por modelo (interactivo), PSI e
  historial v1→v2→v3. Componentes en `web-next/components/lab/`.
- ⚠️ `.gitignore` raíz: `lib/` se ancló a `/lib/` — sin anclar también ignoraba `web-next/lib/`,
  lo que habría roto el build de Vercel.
- Tesis (`main.tex`): resumen ES/EN, nueva sección de suite comparativa v3, métricas detalladas del
  campeón, discusión (subsección de validación empírica del ciclo MLOps), conclusiones y
  recomendaciones — actualizados

**Testing Suite:**
- 390 tests, **100% verde** tras esta sesión (2 aserciones obsoletas que esperaban "XGBoost"/.json
  corregidas para leer de config en vez de hardcodear; 3 tests de embeddings que fallaban por faltar
  el modelo spaCy `es_core_news_md` — instalado esta sesión, ahora pasan)

### Hallazgos y pendientes para rotar

- **Credenciales de BD hardcodeadas y commiteadas** en `notebooks/02_construccion_dataset.ipynb`,
  `07_Validacion_Datos_Reales.ipynb` y `extraccion_xml_json.ipynb` (host `10.250.5.36`,
  usuario/password en texto plano). El código nuevo (`src/db.py`, scripts de esta sesión) lee de
  `Codigo/.env` (gitignored). **Recomendación:** limpiar los notebooks y rotar la contraseña —
  no ejecutado en esta sesión, requiere decisión explícita del usuario.
- `resultados_generados.tex` (raíz del repo) es un artefacto huérfano con valores placeholder
  (nunca reales) que **no está `\input`eado en `main.tex`** — no afecta el PDF compilado. Si se
  quiere mantener consistente, regenerar con `generar_resultados_latex.py` apuntando a los JSON
  de `reports/evaluations/20260910_*.json`.

### Próximos Pasos

1. **Fase 8 (Integración LLM Local):** revisar resultado final de `xgb_enriched` y decidir si se
   documenta como mejora adicional o como "no superó al campeón de 48 features" (ambos son
   resultados válidos para la tesis). Pendiente: generador de explicaciones en lenguaje natural
   con Ollama, demo MVP con Streamlit. Spec: `.kiro/specs/local-llm-integration/tasks.md`.
2. **Fase 7:** confirmar compilación final de `main.tex` sin errores/warnings de referencias.
3. **Rotar credenciales de BD** (ver arriba) — pendiente de decisión del usuario.
4. **Publicar cambios:** commit + push a master (redespliega Vercel automáticamente) — pendiente
   de confirmación del usuario.
5. **Fase 1D (opcional):** Parsers JSON RIPS + XML FEV para producción.

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
