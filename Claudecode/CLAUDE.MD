# CLAUDE.MD — Proyecto: Analítica Predictiva para Auditoría Prestacional
## Savia Salud EPS · Universidad EAFIT · Ciencia de Datos

> **Propósito de este archivo:** Instrucciones canónicas para el asistente de IA que co-desarrolla este proyecto. Toda interacción de codificación debe respetar estrictamente las convenciones, fases y reglas aquí definidas. Este documento es la **fuente única de verdad** para el desarrollo y debe actualizarse al finalizar cada sesión de trabajo.

---

## 0. ESTADO ACTUAL DEL PROYECTO *(Actualizado: 2026-03-11 — Fase 7 Tesis sustancialmente avanzada)*

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
| **6** | API REST | ⏳ Pendiente | 0% |
| **7** | Tesis (main.tex) | 🔄 **En progreso** | 85% |

### Lo que está hecho

- [x] Conexión a BD MySQL Savia Salud (`system_savia`) funcional
- [x] Metadata de 554 tablas exportada en `data/Info_data_base/`
- [x] Muestras (2 filas) de 15 tablas críticas en `data/database_analysis/`
- [x] JOIN `cm_facturas → cm_detalles → cm_auditoria_motivos_glosas` confirmado funcional
- [x] **DISTRIBUCIÓN DE CLASES CONFIRMADA** → `data/database_analysis/01_distribucion_clases.csv`
- [x] Las 3 clases están presentes con volumen suficiente (4,5M facturas en 24 meses)
- [x] Notebooks base: `00_analisis_bd.ipynb`, `01_exploracion_tablas_bd.ipynb`
- [x] **DATASET DE ENTRENAMIENTO CREADO Y CORREGIDO** → `data/processed/20260304_dataset_entrenamiento.parquet`
- [x] `notebooks/02_construccion_dataset.ipynb` completado (celdas 1–11)
- [x] **EDA COMPLETADO** → `notebooks/03_EDA.ipynb` (14 celdas, 12 análisis, 11 figuras)
- [x] 11 figuras EDA generadas en `reports/figures/eda_*.png` a 300 dpi
- [x] Benchmark sistema IA Savia ejecutado (`eda_11_baseline_savia.png`)
- [x] **FEATURE ENGINEERING COMPLETADO** → `notebooks/04_Feature_Engineering.ipynb` (11 celdas)
- [x] **RIGHT-CENSORING CORREGIDO Y VERIFICADO** → Parquets en disco con 3 clases en todos los splits
- [x] Vector de features final con transformaciones log, flags y 6 features de interacción
- [x] Figura `fe_01_correlaciones_features.png` generada a 300 dpi
- [x] **MODEL TRAINING COMPLETADO (v1)** → `notebooks/05_Model_Training.ipynb` (16 celdas)
- [x] 3 modelos entrenados: XGBoost, LightGBM, RandomForest
- [x] Modelos serializados en `models/artifacts/` (XGBoost .json, LightGBM/RF .joblib)
- [x] Métricas exportadas en `reports/evaluations/20260309_metricas.json`
- [x] 3 figuras generadas: matrices de confusión, importancia de features, curva PR Glosada
- [x] Análisis de umbral para clase Glosada incluido en notebook
- [x] **EXPLICABILIDAD SHAP COMPLETADA** → `notebooks/06_SHAP_Explainability.ipynb` (10 celdas)
- [x] 4 figuras SHAP generadas en `reports/figures/shap_*.png` a 300 dpi
- [x] SHAP values calculados con TreeExplainer (1,998 facturas, 4.7s)
- [x] Función `explain_prediction()` con factores de riesgo en lenguaje natural
- [x] Importancia SHAP exportada en `reports/evaluations/20260309_shap_importance.json`
- [x] SHAP values exportados en `reports/evaluations/20260309_shap_values_sample.npz`
- [x] **VALIDACIÓN CONTRA DATOS REALES COMPLETADA** → `notebooks/07_Validacion_Datos_Reales.ipynb` (12 celdas)
- [x] Validación Parte A (Test Set jun-jul 2025): F1-macro=0.7432, confirmada consistente con entrenamiento
- [x] Validación Parte B (OOT ago-dic 2025): F1-macro=0.5941, degradación 20.1% — SIGNIFICATIVA
- [x] Concept drift detectado: Glosada subió de 40.2% a 67.5% en periodo OOT
- [x] 3 figuras de validación generadas: `val_01..03_*.png`
- [x] Reporte JSON: `reports/evaluations/20260310_validacion_datos_reales.json`
- [x] Reporte markdown: `reports/20260311_reporte_validacion_datos_reales.md`
- [x] **TESIS main.tex ACTUALIZADA (Fase 7)** — Resultados reales integrados en documento LaTeX
- [x] Resumen y Abstract actualizados con métricas reales (F1=0.7490, AUC=0.9078)
- [x] Stack tecnológico corregido (PostgreSQL→MySQL, TensorFlow→SHAP/LightGBM)
- [x] Sección Datos reescrita con volúmenes reales (633K facturas, splits temporales, 48 features)
- [x] Tablas de fuentes de datos con tablas MySQL reales (cm_facturas, cm_detalles, etc.)
- [x] Resultados con métricas empíricas y comparación de 3 algoritmos
- [x] Sección OOT con tabla de concept drift (Glosada 40.2%→67.5%)
- [x] Conclusiones reescritas con evidencia empírica (7 conclusiones)
- [x] Recomendaciones basadas en hallazgos (reentrenamiento trimestral, piloto triaje, etc.)
- [x] 9 figuras nuevas integradas en secciones CRISP-DM apropiadas (22 figuras totales)
- [x] Rutas de figuras corregidas (`reports/figures/` → `Codigo/reports/figures/`)
- [x] Errores de compilación LaTeX corregidos (backticks → `\texttt{}`)
- [x] PDF compila exitosamente (5.2 MB, 3 pasadas pdflatex)
- [x] **PROPUESTA_VALOR_MODELO.md** actualizada con análisis de disponibilidad de features, diagrama de flujo operativo, y sección de transparencia

### DISTRIBUCIÓN DE CLASES REAL (últimos 24 meses)

| Clase | Facturas | % | Valor Promedio COP |
|---|---|---|---|
| **0 — Auditada** | 3.772.917 | **83,6%** | $646.277 |
| **1 — Glosada** | 598.293 | **13,3%** | $6.816.873 |
| **2 — Devuelta** | 139.724 | **3,1%** | $2.033.731 |
| **TOTAL** | **4.510.934** | 100% | — |

> **Hallazgo clave:** Las facturas Glosadas tienen valor promedio **10,5× mayor** que las Auditadas ($6,8M vs $646K). El impacto financiero de un falso negativo clase 1 es crítico. Confirma recall de Glosada como métrica primaria.

---

## 0B. DATASET DE ENTRENAMIENTO — ESPECIFICACIÓN FINAL *(2026-03-05)*

### Archivo

`data/processed/20260304_dataset_entrenamiento.parquet` — **~26 MB · 728,289 filas · 39 columnas**

### Distribución en el dataset (sub-muestreada para balanceo)

| Clase | Filas | % en dataset | Estrategia de muestreo |
|---|---|---|---|
| **0 — Auditada** | 188,576 | 25,9% | `MOD(id, 20) = 0` (1 de cada 20) |
| **1 — Glosada** | 400,000 | 54,9% | `LIMIT 400000` sobre universo glosado |
| **2 — Devuelta** | 139,713 | 19,2% | 100% (universo completo) |

> El muestreo es **intencional** para reducir desbalance extremo (83%/13%/3% → 26%/55%/19%). Documentar en tesis como pre-procesamiento de balanceo de clases.

### Columnas del dataset (39)

| Columna | Tipo | Fuente | Notas |
|---|---|---|---|
| `factura_id` | int64 | cm_facturas.id | Llave primaria |
| `target` | int64 | construida | 0=Auditada, 1=Glosada, 2=Devuelta |
| `nit_prestador` | str | cm_facturas.nit | |
| `tipo_contrato_codigo` | str | cm_facturas | |
| `tipo_contrato` | str | cm_facturas | CAPITA / Evento / Paquete |
| `regimen_codigo` | str | cm_facturas | |
| `regimen` | str | cm_facturas | CONTRIBUTIVO / SUBSIDIADO |
| `valor_factura` | float64 | cm_facturas | Winsorizado al p99.9 ($348M COP) |
| `valor_factura_original` | float64 | cm_facturas | Sin winsorizar |
| `valor_copago` | float64 | cm_facturas | |
| `valor_cuota_moderadora` | float64 | cm_facturas | |
| `fecha_prestacion` | datetime | cm_facturas | |
| `fecha_radicacion` | datetime | cm_facturas | |
| `dias_hasta_radicacion` | int64 | calculada | Clipeado [0, 365] |
| `origen_factura` | int64 | cm_facturas | |
| `pbs` | object | cm_facturas | |
| `mes_radicacion` | int64 | calculada | |
| `trimestre_radicacion` | int64 | calculada | |
| `anio_radicacion` | int64 | calculada | |
| `num_items` | float64 | cm_detalles | COUNT(*) por factura |
| `num_cups_distintos` | float64 | cm_detalles | |
| `num_dx_distintos` | float64 | cm_detalles | **Extraído de nombre_dx** (codigo_dx tiene NULLs masivos) |
| `sum_valor_facturado` | float64 | cm_detalles | |
| `max_valor_item` | float64 | cm_detalles | |
| `avg_valor_item` | float64 | cm_detalles | |
| `num_items_pbs` | float64 | cm_detalles | |
| `num_items_con_autorizacion` | float64 | cm_detalles | |
| `num_tipos_servicio` | float64 | cm_detalles | |
| `num_ambitos_distintos` | float64 | cm_detalles | |
| `total_facturas_nit` | int64 | calculada | Histórico 24 meses |
| `tasa_devolucion_nit` | float64 | calculada | % histórico del prestador |
| `tasa_glosa_nit` | float64 | calculada | % histórico del prestador |
| `valor_promedio_nit` | float64 | calculada | |
| `dias_radicacion_promedio_nit` | float64 | calculada | |
| `porcentaje_mayor_item` | float64 | calculada | Clipeado [0, 1] |
| `discrepancia_valor` | float64 | calculada | `|sum_detalles - valor_factura|` |
| `tiene_fev` | int32 | calculada | 1 si tiene FEV electrónica |
| `tipo_contrato_encoded` | int32 | calculada | CAPITA=0, Evento=1, Paquete=2 |
| `regimen_encoded` | int32 | calculada | SUBSIDIADO=0, CONTRIBUTIVO=1 |

### Hallazgos de calidad de datos confirmados

| Columna | Hallazgo | Resolución aplicada |
|---|---|---|
| `codigo_dx` (cm_detalles) | NULLs masivos — código CIE-10 está en `nombre_dx` como prefijo (`K083-descripción`) | Usar `SUBSTRING_INDEX(nombre_dx, '-', 1)` |
| `valor_copago_item` (cm_detalles) | Columna existe en schema pero todos los valores son 0/NULL en producción | Eliminada del dataset (copago ya está en `cm_facturas.valor_copago`) |
| `dias_hasta_radicacion` | Max 4,045 días (11 años) — outliers por errores de captura | Clipeado a [0, 365] — 19,024 filas afectadas |
| `porcentaje_mayor_item` | Max 1,454 — inconsistencia `sum_detalles > valor_factura` | Cap a 1.0 |
| `valor_factura` | Max $11.97 billones COP — outliers extremos | Winsorizado al p99.9 = $348M COP — 729 filas |
| `num_dx_distintos` | Era 0 en todas las filas (bug de query) | Corregido: max=2531, 656K filas con dx>0 |

### Optimización de rendimiento confirmada

La re-extracción de `num_dx_distintos` vía chunks de IN (728K IDs → 15 queries de 50K) tardó **146 minutos**. La versión optimizada con JOIN por fecha (1 sola query) tardó **17.8 minutos** (8.2× más rápido). **Regla para futuras queries masivas sobre cm_detalles:** usar JOIN + filtro por `fecha_radicacion`, nunca `WHERE cm_facturas_id IN (N)` con N > 10K.

### EDA COMPLETADO (Fase 2) — `notebooks/03_EDA.ipynb` *(2026-03-05)*

**12 análisis ejecutados · 11 figuras generadas en `reports/figures/`:**
1. Distribución de clases (dataset vs BD real)
2. `valor_factura` por clase (boxplot + KDE log)
3. `dias_hasta_radicacion` por clase
4. Tasas históricas del prestador (scatter + boxplot)
5. Heatmap correlaciones Spearman
6. Top-20 NITs con mayor tasa de glosa
7. Serie temporal por mes y clase
8. Completitud y outliers
9. Features discriminadoras clave (6 boxplots)
10. Variables categóricas (tipo contrato, régimen, FEV, origen)
11. Benchmark sistema IA Savia (baseline)
12. Resumen de hallazgos para tesis

### FEATURE ENGINEERING COMPLETADO (Fase 3) — `notebooks/04_Feature_Engineering.ipynb` *(2026-03-09)*

**Pipeline de transformación aplicado:**

1. **Columnas eliminadas (11):** IDs, strings, varianza cero (`num_items_pbs`, `num_ambitos_distintos`, `origen_factura`), redundantes (`valor_factura_original`)
2. **Imputación:** NaNs en variables de detalles → 0, NaNs en variables NIT → mediana
3. **Transformaciones log1p (13 variables):** `valor_factura`, `num_items`, `sum_valor_facturado`, `max_valor_item`, etc.
4. **Flags binarios (4):** `is_extemporanea` (>22d, base DE16), `tiene_copago`, `tiene_cuota_mod`, `sin_detalles`
5. **Features de interacción (6):**
   - `ratio_valor_vs_nit` — factura vs promedio histórico del prestador
   - `log_valor_x_tasa_glosa` — valor × tasa glosa (riesgo compuesto)
   - `dias_x_log_valor` — radicación tardía × alto valor
   - `ratio_autorizacion` — % ítems con autorización previa
   - `desviacion_dias_nit` — días vs promedio del prestador
   - `ratio_cups_items` — diversidad de servicios (causal FA0301)
6. **División temporal** (por `fecha_radicacion`, NO aleatoria — previene data leakage)

**Archivos generados (verificados en disco 2026-03-09):**
- `data/processed/20260309_train.parquet` — 515,145 filas × 50 cols
- `data/processed/20260309_val.parquet` — 77,843 filas × 50 cols
- `data/processed/20260309_test.parquet` — 40,193 filas × 50 cols
- `data/processed/20260309_feature_list.txt` — 48 features
- `reports/figures/fe_01_correlaciones_features.png`

**Vector de features final (48 features):**
- 28 features originales (numéricas del dataset base, sin IDs/strings/varianza-cero)
- 13 transformaciones log1p
- 4 flags binarios
- 6 features de interacción

### RIGHT-CENSORING DETECTADO Y CORREGIDO *(2026-03-09)*

> **Causa raíz:** Las facturas radicadas desde agosto 2025 **no han completado el ciclo de auditoría de glosas**. El proceso de auditoría toma meses, por lo que sus etiquetas "Auditada" (clase 0) son en realidad "pendiente de auditar" — labels no confiables. Evidencia: 0 facturas Glosadas desde ago-2025, mientras meses anteriores promedian ~25K/mes.

**Solución aplicada en `04_Feature_Engineering.ipynb` (celdas 7, 10, 11):**
- Truncar dataset al periodo con labels confiables (mar-2024 a jul-2025)
- ~95K facturas censuradas descartadas (todas clase 0 o 2)
- Split temporal re-hecho dentro del rango confiable

| Split | Periodo | Auditada (0) | Glosada (1) | Devuelta (2) | Total |
|---|---|---|---|---|---|
| **Train** | mar-2024 → mar-2025 | 90,193 (17.5%) | 330,712 (64.2%) | 94,240 (18.3%) | 515,145 |
| **Val** | abr-2025 → may-2025 | 17,098 (22.0%) | 53,145 (68.3%) | 7,600 (9.8%) | 77,843 |
| **Test** | jun-2025 → jul-2025 | 13,723 (34.1%) | 16,143 (40.2%) | 10,327 (25.7%) | 40,193 |

- Las **3 clases presentes en todos los splits** — verificado en disco
- Documentar en tesis como hallazgo metodológico: "sesgo de censura por la derecha en datos de auditoría"

### MODEL TRAINING COMPLETADO (Fase 4) — `notebooks/05_Model_Training.ipynb` *(2026-03-09)*

**3 modelos entrenados y evaluados en conjunto de Test (40,193 facturas):**

| Modelo | F1-macro | AUC-ROC | Recall Auditada | Recall Glosada | Recall Devuelta |
|---|---|---|---|---|---|
| **XGBoost** | **0.7490** | **0.9078** | 0.7544 | 0.7261 | 0.7628 |
| LightGBM | 0.7441 | 0.9063 | 0.7686 | 0.7124 | 0.7433 |
| RandomForest | 0.7217 | 0.8966 | 0.8131 | 0.7406 | 0.5854 |

**Criterios de aceptación:**

| Criterio | Meta | XGBoost (mejor) | Estado |
|---|---|---|---|
| Recall Glosada | ≥ 0.80 | 0.7261 | Pendiente optimizar (ajuste umbral o tuning) |
| F1-macro | ≥ 0.75 | 0.7490 | Cercano (~0.75) |
| AUC-ROC | ≥ 0.85 | 0.9078 | **CUMPLIDO** |

> **Hallazgo:** XGBoost es el mejor modelo global. AUC-ROC excelente (0.91). Recall de Glosada (0.73) no alcanza la meta de 0.80 — puede mejorarse con ajuste de umbral de decisión (análisis incluido en celda 15 del notebook) o tuning de hiperparámetros.

**Top-5 features más importantes (XGBoost):**
1. `num_dx_distintos` — Número de diagnósticos distintos por factura
2. `num_tipos_servicio` — Diversidad de tipos de servicio
3. `num_items_con_autorizacion` — Ítems con autorización previa
4. `tasa_devolucion_nit` — Tasa histórica de devolución del prestador
5. `log_num_items_con_autorizacion` — Log de ítems con autorización

**Archivos generados:**
- `models/artifacts/20260309_xgboost_v1.json` — 9.7 MB
- `models/artifacts/20260309_lightgbm_v1.joblib` — 8.1 MB
- `models/artifacts/20260309_randomforest_v1.joblib` — 172 MB
- `reports/evaluations/20260309_metricas.json`
- `reports/figures/mt_01_confusion_matrices.png`
- `reports/figures/mt_02_feature_importance.png`
- `reports/figures/mt_03_precision_recall_glosada.png`

**Configuración XGBoost v1 (usada):**
```python
{
    'objective': 'multi:softprob', 'num_class': 3,
    'eval_metric': 'mlogloss', 'random_state': 42,
    'n_estimators': 1000, 'early_stopping_rounds': 50,
    'learning_rate': 0.05, 'max_depth': 6,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_weight': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
}
```

### EXPLICABILIDAD SHAP COMPLETADA (Fase 5) — `notebooks/06_SHAP_Explainability.ipynb` *(2026-03-10)*

**Método:** `shap.TreeExplainer` — cálculo exacto de SHAP values para XGBoost.
**Muestra:** 1,998 facturas estratificadas (666 por clase) del conjunto de test.
**Tiempo de cálculo:** 4.7 segundos (515 árboles × 48 features × 1,998 muestras).

**Top-5 features por importancia SHAP global (mean|SHAP| promediado sobre 3 clases):**

| # | Feature | mean|SHAP| | Interpretación |
|---|---|---|---|
| 1 | `log_num_items_con_autorizacion` | 1.2284 | Factor más discriminante globalmente |
| 2 | `log_num_dx_distintos` | 0.4209 | Complejidad diagnóstica |
| 3 | `tiene_fev` | 0.3828 | Presencia de factura electrónica |
| 4 | `log_valor_x_tasa_glosa` | 0.2939 | Riesgo compuesto valor × historial |
| 5 | `num_items_con_autorizacion` | 0.2339 | Autorización previa |

**Hallazgos SHAP por clase:**
- **Auditada (0):** `num_tipos_servicio` bajo y `log_valor_x_tasa_glosa` alto favorecen esta clase
- **Glosada (1):** `log_valor_x_tasa_glosa` alto y `tasa_glosa_nit` alta son los principales drivers
- **Devuelta (2):** `log_num_items_con_autorizacion` bajo (=0, sin detalles) es el factor dominante; `tiene_fev` alto también favorece

**Figuras generadas (4):**
- `reports/figures/shap_01_importancia_global.png` — Top 20 features SHAP
- `reports/figures/shap_02_beeswarm_por_clase.png` — Beeswarm plots 3 clases
- `reports/figures/shap_03_waterfall_por_clase.png` — Waterfall plots individuales
- `reports/figures/shap_04_dependence_glosada.png` — Dependence plots top-4 Glosada

**Función `explain_prediction()`:**
- 16 features con descripciones en lenguaje natural para auditores
- 9 features mapeadas a causales de la Res. 2284/2023
- Output compatible con schema API `POST /v1/predict` (Fase 6)
- Ejemplo: *"Predigo GLOSADA porque: (1) Combinación de alto valor y alta tasa de glosa, (2) Muchos ítems con autorización previa (138), (3) Alta diversidad de tipos de servicio (4)"*

**Artefactos exportados:**
- `reports/evaluations/20260309_shap_importance.json` — Importancia SHAP global y por clase
- `reports/evaluations/20260309_shap_values_sample.npz` — SHAP values comprimidos

### VALIDACIÓN CONTRA DATOS REALES (Fase 5B) — `notebooks/07_Validacion_Datos_Reales.ipynb` *(2026-03-11)*

**Notebook:** 12 celdas. Validación en 2 partes: Test Set + Out-of-Time.

**Parte A — Test Set (jun-jul 2025, n=4,999 muestra estratificada):**

| Métrica | Modelo XGBoost | IA Previa Savia |
|---|---|---|
| F1-macro | **0.7432** | 0.1697 |
| Recall Glosada | 0.7206 | 0.0000 |

**Parte B — Out-of-Time (ago-dic 2025, n=223,621):**

| Métrica | Test Set | OOT | Degradación |
|---|---|---|---|
| F1-macro | 0.7432 | **0.5941** | **-20.1%** |
| Recall Glosada | 0.7206 | 0.5389 | -25.2% |

**Concept drift detectado:** Proporción de Glosada subió de 40.2% (test) a 67.5% (OOT). Principal error: 46,869 facturas Glosadas clasificadas como Auditadas (31% de Glosadas OOT).

#### HALLAZGO CRÍTICO: Columnas `ejecucion_glosa_aut_ia` y `estado_glosa_aut` *(2026-03-11)*

> **⚠️ CORRECCIÓN A DOCUMENTACIÓN ANTERIOR:** Estas columnas NO son lo que se pensaba inicialmente.

| Columna | Valores reales encontrados | Significado real |
|---|---|---|
| `ejecucion_glosa_aut_ia` | `1=FA0101`, `2=FA0201`, `3=FA0701`, `4=FA5701`, `5=FA5801` | **Códigos de causal de glosa de facturación (Res. 2284)** — NO son predicciones de clase |
| `estado_glosa_aut` | `0=Pendiente`, `1=bot en proceso`, `2=bot ejecutado`, `3=IA en proceso`, `4=IA ejecutado`, `5=No Aplica` | **Estado del pipeline de automatización** — NO es el resultado de auditoría |
| `estado_factura` | `0=Radicado`, `1=En Auditoría`, `2=Auditada-Glosas`, `3=Cierre Auditoría`, `4=Conciliada`, `5=Devuelta`, `6-13=otros estados` | **Estado administrativo del flujo** — NO es equivalente a nuestro target |

> **Implicaciones confirmadas:**
> 1. **El target actual (construido desde `cm_auditoria_devoluciones` + `cm_auditoria_motivos_glosas`) ES CORRECTO.** Refleja el resultado real de la auditoría.
> 2. **`estado_factura` NO es mejor target** — es un estado de flujo administrativo (Radicado→Auditoría→Conciliada), no el resultado de auditoría.
> 3. **`ejecucion_glosa_aut_ia` contiene códigos FA (causales de glosa por facturación)**, NO predicciones de 3 clases. Los códigos FA0101, FA0201, etc. corresponden a la taxonomía de la Res. 2284 (§5 de este documento).
> 4. **`estado_glosa_aut` indica si el bot/IA ya procesó la factura**, NO el resultado.
> 5. **La comparación "IA Previa" del notebook 07 tiene un error de mapeo**: el mapeo por texto (`'glos'`, `'devol'`) sobre códigos numéricos/FA siempre cae al default → todo se mapea como Auditada. **La conclusión de que la IA Previa clasifica 100% como Auditada es un artefacto del mapeo incorrecto, no del sistema real.**
> 6. **Para una comparación válida**, se necesita investigar qué significan los códigos FA en el contexto del bot de Savia y cómo mapearlos correctamente al esquema {0,1,2}.

**Artefactos generados:**
- `reports/evaluations/20260310_validacion_datos_reales.json`
- `reports/20260311_reporte_validacion_datos_reales.md`
- `reports/figures/val_01_confusion_matrices_comparacion.png`
- `reports/figures/val_02_metricas_por_clase_comparacion.png`
- `reports/figures/val_03_oot_confusion_estabilidad.png`

### Próximos pasos

**Opción A — Fase 6 (API REST):** Crear MVP funcional con FastAPI
1. `src/api/main.py`: `POST /v1/predict`, `POST /v1/predict/batch`, `GET /v1/model/info`
2. `src/api/schemas.py`: Schemas Pydantic
3. Integrar `explain_prediction()` desde notebook 06 SHAP
4. Refactorizar código de notebooks a módulos `src/`

**Opción B — Fase 7 (Tesis, 15% restante):** Pulir main.tex
1. Revisar que todas las referencias cruzadas (`\ref{}`, `\cite{}`) sean correctas
2. Verificar bibliografía completa
3. Revisar formato de tablas y figuras (captions, labels)
4. Agregar notebook `08_Analisis_Profundo_Comparacion.ipynb` si aporta contenido

**Opcional:** Optimizar recall de Glosada via tuning de hiperparámetros (meta: 0.80, actual: 0.73).

---

## 1. CONTEXTO DEL PROYECTO

| Atributo | Valor |
|---|---|
| **Proyecto** | Estrategia de Analítica Predictiva para Optimización de Auditoría Prestacional |
| **Organización** | Savia Salud EPS — Colombia |
| **Marco regulatorio** | Resolución 2275 de 2023 (RIPS JSON/XML FEV) · Resolución 2284 de 2023 (Manual Único de Glosas) |
| **Repositorio local** | `d:\Users\jcardonr\Documents\eafit\Codigo\` |
| **Documento tesis** | `d:\Users\jcardonr\Documents\eafit\main.tex` |
| **Entregable técnico** | MVP funcional (modelo + API local) + propuesta de arquitectura futura en AWS |
| **Arquitectura futura** | AWS SageMaker (entrenamiento) + Lambda (inferencia) + S3 (feature store) + RDS (BD prod) |
| **Fecha de inicio** | 2026-03-02 |
| **Lenguaje principal** | Python 3.11+ |
| **Fuente datos entrenamiento** | BD MySQL `system_savia` (host `10.250.5.36`) |
| **Fuente datos producción** | JSON RIPS + XML FEV (URLs en Azure Blob Storage via `cm_fe_rips_cargas.url_json` / `url_xml`) |

### Problema Central
El modelo de auditoría de Savia Salud EPS es **reactivo y manual**, generando cuellos de botella en flujo de caja e incertidumbre en el ciclo de ingresos. La implementación del estándar de interoperabilidad (JSON RIPS + XML FEV) crea la oportunidad de construir un sistema de **clasificación predictiva multiclase** de facturas antes de que agoten los ciclos de revisión humana.

### Objetivo del Modelo
Clasificar cada factura/cuenta médica en uno de **tres estados** (tarea multiclase):

| Clase | Código | Descripción | Acción Proactiva |
|---|---|---|---|
| **Auditada** | `0` | Alta probabilidad de cumplimiento total. Sin objeciones previsibles. | Fast-track / Auditoría automática / Pago programado |
| **Glosada** | `1` | Inconsistencia en pertinencia clínica, tarifas o soportes. Objeción parcial o total. | Enviar a auditor médico especializado |
| **Devuelta** | `2` | Error de forma: radicación extemporánea, pagador incorrecto u otros causales técnicos. | Notificación inmediata al prestador para corrección |

---

## 1B. HALLAZGOS CONFIRMADOS DE LA BASE DE DATOS *(2026-03-04)*

### Volumen de datos disponible

| Tabla | Registros | Rol en el modelo |
|---|---|---|
| `cm_facturas` | **12.471.525** | Unidad de análisis (1 fila = 1 factura) |
| `cm_detalles` | **260.125.272** | Líneas de servicio (CUPS + CIE-10 + valores) |
| `cm_auditoria_motivos_glosas` | **71.778.090** | TARGET clase 1 (Glosada) |
| `cm_auditoria_devoluciones` | **550.833** | TARGET clase 2 (Devuelta) |
| `cm_fe_rips_cargas` | **2.050.588** | FEV electrónico (URLs JSON/XML) |
| `cm_fe_rips_carga_contenidos` | **6.150.308** | Contenido JSON RIPS + XML almacenado en BD |
| `aseg_afiliados` | **3.603.076** | Datos del paciente (género, edad) |
| `cnt_contratos` | **2.915** | Contratos vigentes e históricos |
| `cnt_prestadores` | **53.563** | Prestadores (IPS) registrados |

### Arquitectura de la variable TARGET confirmada

```
TARGET(factura) =
  2 (Devuelta)  si EXISTS en cm_auditoria_devoluciones WHERE cm_facturas_id = f.id
  1 (Glosada)   si EXISTS en cm_auditoria_motivos_glosas
                  JOIN cm_detalles WHERE cm_detalles.cm_facturas_id = f.id
  0 (Auditada)  en caso contrario
```

> **CRÍTICO:** `cm_auditoria_devoluciones` tiene FK directa `cm_facturas_id`.
> `cm_auditoria_motivos_glosas` tiene FK `cm_detalles_id` (JOIN doble necesario).
> El SQL en `02_queries_dataset_entrenamiento.sql` debe ajustarse — versión corregida en `02b_query_corregido.sql`.

### Campos clave confirmados por tabla

**`cm_facturas`** (67 cols):
- `id`, `nit`, `ips` — identificadores (nit = NIT del prestador, denormalizado)
- `cm_fe_rips_cargas_id` — FK al RIPS electrónico ✅
- `valor_factura`, `valor_inicial_glosa`, `valor_pagado_factura`
- `fecha_prestacion`, `fecha_radicacion` — para calcular `dias_hasta_radicacion`
- `estado_factura` (int), `estado_glosa_aut` — estados del proceso
- `mae_tipo_contrato_valor` — 'CAPITA' / 'Evento' / 'Paquete'
- `mae_regimen_valor` — 'CONTRIBUTIVO' / 'SUBSIDIADO'
- `ejecucion_glosa_aut_ia` — **⚠️ Ya existe un sistema de IA para glosas. Contextualizar en tesis.**
- `numero_contrato`, `numero_contrato_auditoria`

**`cm_detalles`** (63 cols):
- `cm_facturas_id` — FK a factura padre ✅
- `aseg_afiliados_id` — FK al afiliado ✅
- `ma_servicio_codigo` — Código CUPS del servicio
- `ma_servicio_valor` — Nombre descriptivo del servicio
- `codigo_dx` — Diagnóstico CIE-10
- `valor_facturado`, `valor_glosa`, `valor_pagado`
- `aplica_glosa` (bit), `aplica_autorizacion` (bit), `aplica_pbs` (bit)
- `tipo_servicio` — Clasificación del tipo de servicio
- ⚠️ `nombre_completo_afiliado` — **DATO PII. NUNCA incluir en features ni en CSV publicados.**

**`cm_auditoria_motivos_glosas`** (19 cols):
- `cm_detalles_id` — FK a línea de detalle ✅
- `mae_motivo_codigo` — Código de motivo (e.g., `2`=Tarifas)
- `mae_motivo_valor` — Descripción ('Tarifas', 'Soportes', etc.)
- `mae_motivo_especifico_codigo` — Causal específica (e.g., `202`)
- `mae_motivo_especifico_valor` — Descripción del causal específico
- `mae_motivo_glosa_aplicacion_codigo` — Resolución aplicada (e.g., `RE3047`)
- `valor_motivo` — Monto glosado en COP
- `tipologia` — Clasificación de la glosa
- `observacion` — Texto libre del auditor

**`cm_auditoria_devoluciones`** (30 cols):
- `cm_facturas_id` — FK directa a factura ✅ (NO pasa por cm_detalles)
- `mae_motivo_devolucion_codigo` — Código del motivo
- `mae_motivo_devolucion_valor` — Descripción (e.g., 'Factura no cumple requisitos legales')
- `mae_devolucion_motivo_general_codigo` — Resolución base (e.g., `3047`)
- `fecha_devolucion` — Fecha del rechazo
- `observacion` — Texto detallado de la razón

**`cm_fe_rips_cargas`** (62 cols):
- `cnt_prestador_sedes_id` — FK a sede del prestador ✅
- `cnt_contratos_id` — FK al contrato ✅
- `factura_numero`, `factura_valor` — Número y valor de la FEV
- `url_xml` — **URL Azure Blob con el archivo XML FEV** (SAS token, expirable)
- `url_json` — **URL Azure Blob con el archivo JSON RIPS** (SAS token, expirable)
- `de1601_eps_erronea` (bit) — Validación DIAN: EPS incorrecta
- `de4401_profesional_red` (bit) — Profesional no en red
- `de5001_pagada` (bit) — Factura ya pagada
- `de5601_soporte_fe` (bit) — Soporte de facturación electrónica

> **Nota arquitectónica:** Los JSON RIPS y XML FEV NO están en el servidor local. Están en **Azure Blob Storage** con URLs SAS (temporales). Para el modelo de producción se debe acceder via URL o usar `cm_fe_rips_carga_contenidos` que tiene el contenido ya almacenado en BD.

### Sistema de IA/Bot paralelo en Savia — ACTUALIZADO *(2026-03-11)*

La tabla `cm_facturas` contiene columnas de un **sistema automatizado de glosas preexistente** en Savia:

| Columna | Valores | Significado real confirmado |
|---|---|---|
| `ejecucion_glosa_aut_ia` | `1=FA0101`, `2=FA0201`, `3=FA0701`, `4=FA5701`, `5=FA5801` | **Código de causal de glosa** que el bot/IA detectó (códigos FA de la Res. 2284) |
| `estado_glosa_aut` | `0=Pendiente`, `1=bot en proceso`, `2=bot ejecutado`, `3=IA en proceso`, `4=IA ejecutado`, `5=No Aplica` | **Estado del pipeline de automatización** (si ya corrió o no) |
| `estado_factura` | `0-13`: Radicado, En Auditoría, Auditada-Glosas, Cierre Auditoría, Conciliada, Devuelta, etc. | **Estado administrativo del flujo** de la factura |

> **Conclusiones confirmadas (sesión 2026-03-11):**
> 1. **`estado_glosa_aut` NO es ground truth de auditoría** — solo indica si el bot ya procesó la factura. **NO es candidato a target.**
> 2. **`ejecucion_glosa_aut_ia` NO contiene predicciones de 3 clases** — contiene códigos FA de causales de glosa por facturación. **NO es comparable directamente con nuestro esquema {0,1,2}.**
> 3. **`estado_factura` es un estado de flujo administrativo**, no el resultado de la auditoría. Una factura en estado "2=Auditada-Glosas" ya pasó por auditoría y tiene glosas, pero el valor numérico no corresponde a nuestras clases.
> 4. **El target correcto sigue siendo el construido desde JOINs** a `cm_auditoria_devoluciones` (Devuelta) y `cm_auditoria_motivos_glosas` (Glosada).
> 5. **Para comparar con el sistema IA de Savia**, se necesita mapear los códigos FA a nuestro esquema. Los códigos FA indican detección de glosas por facturación, lo que implica que si `ejecucion_glosa_aut_ia` tiene valor, el bot detectó una posible glosa (clase 1).
> 6. **⚠️ Estas columnas NO deben usarse como features** — son resultados del proceso automatizado (data leakage).

### Corrección al SQL de entrenamiento

El archivo `data/Info_data_base/02_queries_dataset_entrenamiento.sql` tiene un error en el JOIN de devoluciones:

```sql
-- ❌ INCORRECTO (cm_auditoria_devoluciones NO tiene cm_detalles_id):
INNER JOIN cm_auditoria_devoluciones cad ON cad.cm_detalles_id = cd.id

-- ✅ CORRECTO (FK directa a cm_facturas):
LEFT JOIN cm_auditoria_devoluciones cad ON cad.cm_facturas_id = f.id
```

---

## 2. ESTRUCTURA DEL REPOSITORIO

```
Codigo/
├── CLAUDE.MD                  ← (este archivo) — Reglas e instrucciones canónicas
├── README.md                  ← Descripción general del proyecto
├── requirements.txt           ← Dependencias Python
├── reglas.md                  ← Reglas de negocio específicas de glosas (Res. 2284)
│
├── agents/                    ← Definiciones de subagentes Claude con funciones específicas
│   ├── 01_data_extractor.md   ← Subagente: extracción de datos desde BD (solo lectura)
│   ├── 02_eda_analyst.md      ← Subagente: análisis exploratorio de datos
│   ├── 03_feature_engineer.md ← Subagente: ingeniería de características
│   ├── 04_model_trainer.md    ← Subagente: entrenamiento y evaluación del modelo
│   └── 05_thesis_writer.md    ← Subagente: generación de contenido LaTeX para tesis
│
├── data/
│   ├── raw/                   ← Datos ORIGINALES e INMUTABLES. NUNCA modificar.
│   │   ├── json_rips/         ← Objetos JSON RIPS recibidos de IPS (local extraído)
│   │   └── xml_fev/           ← Archivos XML de Facturación Electrónica de Venta
│   ├── database_analysis/     ← Muestras y análisis de BD (solo lectura — extraído)
│   ├── Info_data_base/        ← Metadata de BD: estructura, relaciones, queries
│   ├── interim/               ← Datos en proceso de transformación (ETL parcial)
│   └── processed/             ← Dataset final aplanado y listo para ML (Parquet)
│
├── notebooks/
│   ├── 00_analisis_bd.ipynb           ← Mapeo BD y relaciones FK
│   ├── 01_exploracion_tablas_bd.ipynb ← Muestras de tablas + distribución de clases
│   ├── 02_construccion_dataset.ipynb  ← Extracción dataset de entrenamiento
│   ├── 03_EDA.ipynb                   ← Análisis Exploratorio de Datos
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Model_Training.ipynb        ← Entrenamiento + evaluación de modelos
│   ├── 06_SHAP_Explainability.ipynb    ← Explicabilidad SHAP (XAI)
│   ├── 07_Validacion_Datos_Reales.ipynb ← Validación contra BD real + OOT
│   └── extraccion_xml_json.ipynb       ← Legacy: exploración XML/JSON
│
├── src/                          ← ⚠️ Directorios creados pero SIN código .py aún
│   ├── data/                    ←   (todo el código está en notebooks por ahora)
│   ├── features/                ←   Se refactorizará a módulos .py en Fase 6
│   ├── models/
│   ├── api/                     ← PENDIENTE: FastAPI endpoints (Fase 6)
│   │   ├── main.py              ←   POST /v1/predict, POST /v1/predict/batch
│   │   └── schemas.py           ←   Pydantic schemas
│   ├── explainability/
│   │   └── shap_explanations.py ←   explain_prediction() desde notebook 06
│   └── visualization/
│
├── latex/                     ← Scripts Python para generar contenido LaTeX automáticamente
│   ├── generar_tablas.py      ← Exporta métricas del modelo como tablas .tex
│   ├── generar_figuras.py     ← Copia y formatea gráficos para reports/figures/
│   └── exportar_resultados.py ← Orquestador: genera todos los archivos .tex de resultados
│
├── models/
│   ├── artifacts/             ← Modelos serializados (.pkl / .joblib / .json booster)
│   └── feature_store/         ← Features pre-calculadas por NIT prestador
│
└── reports/
    ├── figures/               ← Gráficos para tesis LaTeX (PNG/PDF 300 dpi)
    └── evaluations/           ← Reportes de métricas en JSON/HTML
```

---

## 3. REGLAS DE CODIFICACIÓN (OBLIGATORIAS)

### 3.1 Reglas Generales
- **Idioma del código:** Inglés para nombres de variables, funciones y clases. Español para docstrings, comentarios y mensajes de log.
- **Estilo:** Seguir PEP 8 estrictamente. Usar `black` para formateo y `ruff` para linting.
- **Tipado estático:** Toda función debe tener **type hints** completos (parámetros y retorno).
- **Docstrings:** Formato Google Style. Obligatorio en toda función pública.
- **Logging:** Usar el módulo estándar `logging` de Python. No usar `print()` en código de producción (`src/`).
- **Manejo de errores:** Usar excepciones específicas. Nunca capturar `Exception` base sin re-lanzar o loggear.
- **Tests:** Todo módulo en `src/` debe tener su test correspondiente en `tests/` (pytest). Cobertura mínima: 80%.
- **Reproducibilidad:** Fijar semilla aleatoria con `RANDOM_STATE = 42` en todos los modelos.

### 3.2 Reglas de Datos (CRÍTICAS)

> **🔒 REGLA DE SEGURIDAD MÁXIMA — BASE DE DATOS:**
> El asistente de IA **NUNCA debe generar código que modifique datos en la base de datos** de producción `system_savia`.
> - Solo se permiten queries `SELECT` (solo lectura).
> - Están **completamente prohibidos**: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `TRUNCATE`, `ALTER`, `CREATE TABLE`.
> - Todo el trabajo de transformación se realiza **localmente** sobre datos extraídos en `data/`.
> - Si una tarea requiere escribir en BD, detener y consultar al investigador antes de proceder.

- **Los datos en `data/raw/` son SAGRADOS.** El asistente de IA NUNCA debe proponer código que modifique, sobreescriba o elimine archivos dentro de `data/raw/`.
- **Flujo de datos obligatorio:** BD (solo lectura) → `data/raw/` o `data/database_analysis/` → `data/interim/` → `data/processed/`. Nunca en sentido inverso.
- **Anonimización:** Ningún dato que contenga nombre de paciente, número de documento o dirección debe aparecer en notebooks publicados ni en logs. Usar solo `factura_id` y `nit_prestador` como identificadores. La columna `nombre_completo_afiliado` de `cm_detalles` **nunca** se extrae.
- **Validación pre-procesamiento (REGLA DE NEGOCIO #1):** La suma de los valores de los servicios en el JSON RIPS debe coincidir exactamente (a nivel de centavos) con el valor neto a pagar del XML FEV. Cualquier discrepancia debe generar un registro en `data/interim/inconsistencias_fiscales.csv` y excluir la factura del conjunto de entrenamiento principal, clasificándola automáticamente como **Devuelta (clase 2)**.
- **Versionar el dataset:** Usar prefijos de fecha `YYYYMMDD_` en todos los archivos de `data/processed/`.

### 3.3 Reglas de Modelado
- **Algoritmo preferido:** XGBoost (`xgboost>=2.0`) o LightGBM para producción. Scikit-learn solo para baselines y comparación.
- **Tarea:** Clasificación multiclase (`num_class=3`). NUNCA simplificar a binaria sin aprobación explícita del investigador.
- **Métrica primaria de optimización:** `recall` de la clase Glosada (`clase=1`). El costo de un falso negativo (glosa no detectada) supera largamente el de un falso positivo.
- **Métricas obligatorias de reporte:** F1-Score macro, AUC-ROC (OvR), Recall por clase, Matriz de Confusión normalizada.
- **Umbral de decisión:** Por defecto `0.75` para la clase Glosada. Este valor es configurable en `src/models/predict.py`.
- **Serialización de modelos:** Usar `joblib` para pipelines sklearn y el formato nativo `.json` de XGBoost booster.

### 3.4 Reglas de la API
- **Framework:** FastAPI (ya en `requirements.txt`).
- **Endpoint principal:** `POST /predict` — Acepta JSON RIPS + XML FEV codificado en base64, retorna `{ "estado": "glosada", "clase": 1, "probabilidades": {...}, "factores_riesgo": [...] }`.
- **Validación de entrada:** Pydantic v2. Schemas en `src/api/schemas.py`.
- **Latencia objetivo:** < 200ms por predicción en modo batch diario; < 500ms en tiempo real.
- **Versión de API:** Siempre incluir prefijo `/v1/` en las rutas.

---

## 4. PLAN DE TRABAJO — FASES END-TO-END

### FASE 0 · Configuración del Entorno *(Semana 1)* — ✅ COMPLETADA

**Entregables completados:**
- [x] Conexión a BD MySQL funcional
- [x] Estructura de carpetas base creada
- [x] Notebooks base: `00_analisis_bd.ipynb`, `01_exploracion_tablas_bd.ipynb`
- [x] Metadata de BD exportada: `data/Info_data_base/`
- [x] Muestras de tablas: `data/database_analysis/`
- [ ] `requirements.txt` con versiones exactas — **pendiente**
- [ ] Entorno virtual `.venv` — **pendiente**
- [ ] Pre-commit hooks (`black`, `ruff`) — **pendiente**

**Reglas específicas de la fase:**
- El asistente debe generar datos sintéticos realistas que respeten la estructura del Anexo Técnico No. 1 de la Res. 2275 (JSON RIPS) y la extensión de salud DIAN (XML FEV).
- Los datos sintéticos deben incluir los tres estados de la variable objetivo en proporción aproximada: 60% Auditada, 30% Glosada, 10% Devuelta (distribución típica en EPS colombianas).

**Comandos de referencia:**
```bash
# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
pip install black ruff pytest pytest-cov shap lightgbm
```

---

### FASE 1A · Mapeo BD y Construcción del Dataset *(Semanas 1-2)* — ✅ COMPLETADA

**Objetivo:** Construir el dataset de entrenamiento etiquetado con las 3 clases desde la BD.

**Completado:**
- [x] Schema de tablas críticas documentado (ver §1B)
- [x] JOIN confirmado funcional en `99_test_join_glosas.csv`
- [x] SQL propuesto en `data/Info_data_base/02_queries_dataset_entrenamiento.sql`
- [x] FK corregida: `cm_auditoria_devoluciones.cm_facturas_id` (directa)

**Pendiente:**
- [x] Distribución de clases ejecutada: 83,6% / 13,3% / 3,1% → `01_distribucion_clases.csv`
- [x] FK corregida en `01_exploracion_tablas_bd.ipynb` celda 11
- [ ] Crear `notebooks/02_construccion_dataset.ipynb` con SQL corregido
- [ ] Extraer dataset ≥ 100.000 facturas → `data/processed/20260304_dataset_entrenamiento.parquet`

---

### FASE 1B · ETL y Parsers JSON RIPS + XML FEV *(Semanas 2-3)*
**Objetivo:** Pipeline de ingesta para producción (API recibirá JSON + XML de las IPS).

> **Nota:** Los archivos JSON/XML están en Azure Blob Storage. URLs disponibles via `cm_fe_rips_cargas.url_json` y `url_xml`. Para entrenamiento usar `cm_fe_rips_carga_contenidos` (contenido ya en BD).

---

### FASE 2 (original) · Ingesta y ETL *(Semanas 2-3)*
**Objetivo:** Pipeline robusto de extracción, validación y transformación de JSON RIPS y XML FEV.

**Módulos a desarrollar:**
- `src/data/ingest_json.py` — Carga y aplanamiento (`json_normalize`) de objetos RIPS jerárquicos
- `src/data/ingest_xml.py` — Parseo XML con `lxml`, extracción de extensión de salud (11 campos)
- `src/data/validator.py` — Reglas de validación (ver abajo)
- `src/data/etl_pipeline.py` — Orquestador: cruza JSON + XML, produce DataFrame unificado

**Reglas de validación obligatorias en `validator.py`:**

| ID Regla | Descripción | Causal Glosa/Dev. | Clase Asignada |
|---|---|---|---|
| `VAL-001` | Suma JSON ≠ Valor neto XML (diferencia > $0.01 COP) | MUV / Inconsistencia fiscal | Devuelta (2) |
| `VAL-002` | Fecha radicación > Fecha factura + 22 días hábiles | DE16 Radicación Extemporánea | Devuelta (2) |
| `VAL-003` | CUPS no autorizado para nivel de complejidad de la IPS | TA (Tarifas) | Glosada (1) |
| `VAL-004` | Diagnóstico CIE-10 incongruente con sexo del paciente | SO (Soportes) | Glosada (1) |
| `VAL-005` | Cirugía compleja sin honorarios de anestesiólogo (SO0303) | SO0303 | Glosada (1) |
| `VAL-006` | Epicrisis ausente o incompleta en estancia de urgencias | SO0101 | Glosada (1) |
| `VAL-007` | NIT prestador no coincide con pagador del contrato | DE (Devolución) | Devuelta (2) |
| `VAL-008` | Procedimiento de parto con diagnóstico no obstétrico | SO (Pertinencia) | Glosada (1) |

**Salida esperada de la Fase 1:**
```python
# DataFrame resultante (una fila por factura)
{
  "numero_factura": str,
  "nit_prestador": str,
  "fecha_factura": datetime,
  "fecha_radicacion": datetime,
  "modalidad_pago": str,       # Evento / Capitación / Paquete
  "numero_contrato": str,
  "valor_total_json": float,
  "valor_neto_xml": float,
  "discrepancia_fiscal": float,
  "dias_radicacion": int,
  # ... (ver Feature Engineering)
  "estado_label": int           # 0=Auditada, 1=Glosada, 2=Devuelta
}
```

---

### FASE 2 · Análisis Exploratorio de Datos — EDA *(Semana 3)* — ✅ COMPLETADA
**Objetivo:** Entender la distribución de los datos, detectar valores atípicos y documentar hallazgos para la tesis.

**Notebook:** `notebooks/03_EDA.ipynb` — 14 celdas, 12 análisis, 11 figuras generadas.

**Análisis completados:**
1. Distribución de clases (dataset vs BD real) — `eda_01_distribucion_clases.png`
2. Valor factura por clase (boxplot + KDE log) — `eda_02_valor_factura.png`
3. Días hasta radicación por clase — `eda_03_dias_radicacion.png`
4. Tasas históricas del prestador — `eda_04_tasas_prestador.png`
5. Heatmap correlaciones Spearman — `eda_05_correlaciones.png`
6. Top-20 NITs con mayor tasa de glosa — `eda_06_top20_nit_glosa.png`
7. Serie temporal por mes y clase — `eda_07_serie_temporal.png`
8. Completitud y outliers — `eda_08_completitud.png`
9. Features discriminadoras clave — `eda_09_features_discriminadoras.png`
10. Variables categóricas — `eda_10_variables_categoricas.png`
11. Benchmark sistema IA Savia — `eda_11_baseline_savia.png`
12. Resumen de hallazgos para tesis (texto)

**Reglas del EDA:**
- Todos los gráficos guardados en `reports/figures/` en formato PNG (300 dpi).
- El EDA concluye con tabla de variables candidatas para Feature Engineering.

---

### FASE 3 · Ingeniería de Características *(Semanas 4-5)*
**Objetivo:** Construir el vector de características óptimo para el modelo predictivo.

**Módulos a desarrollar:**
- `src/features/clinical_features.py`
- `src/features/provider_features.py`
- `src/features/financial_features.py`

#### 3.1 Variables de Pertinencia Clínica (`clinical_features.py`)

| Feature | Tipo | Descripción |
|---|---|---|
| `cie10_cups_match_prob` | float | Probabilidad histórica del par (CIE-10, CUPS) en la IPS |
| `dx_sexo_inconsistente` | bool | Flag: diagnóstico incompatible con sexo del paciente |
| `num_dx_secundarios` | int | Número de comorbilidades (diagnósticos secundarios) |
| `nivel_complejidad_cups` | int | Nivel de complejidad requerido por el CUPS (1-4) |
| `nivel_habilitacion_ips` | int | Nivel de habilitación de la IPS reportante |
| `complejidad_mismatch` | bool | Flag: nivel CUPS > nivel IPS |
| `tiene_anestesiologia` | bool | Cirugía mayor con honorarios de anestesiólogo reportados |
| `tiene_epicrisis` | bool | Urgencias con epicrisis en soportes |
| `cups_autorizado_contrato` | bool | CUPS cubierto por el contrato específico del pagador |

#### 3.2 Variables de Comportamiento del Prestador (`provider_features.py`)

| Feature | Tipo | Descripción |
|---|---|---|
| `tasa_glosa_historica_nit` | float | % de glosas históricas del NIT en los últimos 12 meses |
| `tasa_devolucion_historica_nit` | float | % de devoluciones históricas del NIT |
| `valor_promedio_factura_nit` | float | Valor promedio de facturación del prestador |
| `dias_promedio_radicacion_nit` | float | Días promedio de radicación del prestador |
| `concentracion_cups_prestador` | float | Índice HHI de concentración de CUPS del prestador |
| `trimestre_facturacion` | int | Trimestre del año (1-4): detecta patrones de fin de mes/año |

#### 3.3 Variables de Concentración y Severidad (`financial_features.py`)

| Feature | Tipo | Descripción |
|---|---|---|
| `valor_total_factura` | float | Valor total de la factura en COP |
| `num_items_factura` | int | Número total de ítems facturados |
| `num_medicamentos_alto_costo` | int | Medicamentos CUM de alto costo incluidos |
| `modalidad_pago_encoded` | int | Evento=0, Capitación=1, Paquete=2 |
| `porcentaje_mayor_item` | float | % del ítem de mayor valor sobre el total |
| `case_mix_index` | float | Índice de complejidad ponderado de la factura |
| `discrepancia_fiscal_abs` | float | |Valor JSON - Valor XML| en COP |
| `dias_hasta_radicacion` | int | Días entre fecha de factura y fecha de radicación |

**Regla de Feature Store:**
Las features de comportamiento del prestador (`provider_features.py`) deben calcularse **una sola vez por período** y almacenarse en `models/feature_store/provider_stats_YYYYMM.parquet`. El módulo de predicción debe leerlas desde allí, no recalcularlas en tiempo real.

---

### FASE 4 · Entrenamiento del Modelo *(Semanas 5-6)*
**Objetivo:** Entrenar, validar y seleccionar el modelo predictivo multiclase óptimo.

**Módulos a desarrollar:**
- `src/models/train.py`
- `src/models/evaluate.py`

**Protocolo de entrenamiento:**

```python
# Configuración base XGBoost
XGBOOST_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": ["mlogloss", "merror"],
    "use_label_encoder": False,
    "random_state": 42,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": "auto"  # Manejar desbalance de clases
}

# División temporal del dataset (NO aleatoria - previene data leakage)
# Train: facturas del período T-12 a T-2
# Validation: facturas del período T-2 a T-1
# Test: facturas del mes T (último mes disponible)
```

**Manejo de desbalance de clases:**
- Aplicar `scale_pos_weight` en XGBoost O usar `class_weight='balanced'` en el pipeline.
- Evaluar SMOTE solo si el recall de la clase minoritaria (Devuelta) es < 0.50 tras balanceo nativo.

**Evaluación obligatoria (notebook `04_Model_Evaluation.ipynb`):**
1. Matriz de confusión (normalizada por fila)
2. Classification report (precision, recall, F1 por clase)
3. AUC-ROC multiclase (One-vs-Rest)
4. Curva Precision-Recall para clase Glosada
5. Análisis de umbrales: impacto económico de variar el threshold sobre recall de Glosada
6. Comparación de modelos (tabla): XGBoost vs LightGBM vs RandomForest (baseline)

**Criterio de aceptación del modelo:**
- Recall clase Glosada ≥ 0.80
- F1-Score macro ≥ 0.75
- AUC-ROC ≥ 0.85

---

### FASE 5 · Explicabilidad (XAI) *(Semana 6)*
**Objetivo:** Proporcionar explicaciones auditables para cada predicción (adopción por auditores médicos).

**Módulo:** `src/explainability/shap_explanations.py`
**Notebook:** `notebooks/05_SHAP_Explainability.ipynb`

**Entregables:**
- Summary plot SHAP global (importancia de características)
- Waterfall plot SHAP por factura individual (explicación local)
- Función `explain_prediction(factura_id) -> dict` que retorna los 5 factores de riesgo principales en lenguaje natural
- Reporte de ejemplo: `"Predigo GLOSA porque: (1) la IPS Nivel 1 facturó CUPS de cirugía Nivel 3, (2) el diagnóstico CIE-10 no corresponde al sexo del paciente, (3) la tasa histórica de glosas del NIT es 45%."`

---

### FASE 6 · API REST y Despliegue *(Semanas 7-8)*
**Objetivo:** Exponer el modelo como servicio en tiempo real.

**Módulos:** `src/api/main.py`, `src/api/schemas.py`

**Endpoints requeridos:**

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/v1/predict` | Predicción individual: recibe JSON RIPS + XML FEV |
| `POST` | `/v1/predict/batch` | Predicción batch: array de facturas |
| `GET` | `/v1/model/info` | Versión del modelo, fecha de entrenamiento, métricas |
| `GET` | `/v1/health` | Health check del servicio |

**Schema de respuesta `POST /v1/predict`:**
```json
{
  "numero_factura": "FEV-2024-001234",
  "estado_predicho": "glosada",
  "clase": 1,
  "confianza": 0.87,
  "probabilidades": {
    "auditada": 0.08,
    "glosada": 0.87,
    "devuelta": 0.05
  },
  "factores_riesgo": [
    "CUPS 89.07.1 no autorizado para Nivel I de complejidad",
    "Diagnóstico Z34 en paciente masculino",
    "Tasa histórica de glosas del prestador: 42%"
  ],
  "causal_probable": "SO0101",
  "accion_sugerida": "Enviar a auditoría médica especializada",
  "timestamp_prediccion": "2026-03-02T07:49:37-05:00"
}
```

**Infraestructura MLOps:**
- Crear `Dockerfile` en la raíz del proyecto.
- La imagen debe incluir el modelo serializado y el feature store.
- Variable de entorno `MODEL_VERSION` para seleccionar el artefacto del modelo.

---

### FASE 7 · Documentación e Integración Tesis *(Semana 8)*
**Objetivo:** Exportar resultados, gráficos y tablas para integración en `main.tex`.

**Reglas de integración con LaTeX:**
- Todos los gráficos se generan en `reports/figures/` a 300 dpi, formato PDF o PNG.
- Las tablas de métricas se exportan como fragmentos `.tex` en `reports/evaluations/` para `\input{}` directo.
- El script `generar_resultados_latex.py` (ya existente) debe actualizarse para incorporar las métricas del modelo.

---

## 5. TAXONOMÍA DE CAUSALES DE GLOSA (Res. 2284/2023)

El modelo usará estas categorías como contexto para las predicciones. Los códigos son cruciales para el campo `causal_probable` en la respuesta de la API.

### Glosas por Soportes (SO)
| Código | Descripción |
|---|---|
| `SO0101` | Ausencia o inconsistencia en epicrisis de urgencias |
| `SO0201` | Historia clínica incompleta |
| `SO0303` | Ausencia de honorarios de anestesiólogo en cirugía mayor |
| `SO0401` | Consentimiento informado ausente |

### Glosas por Facturación (FA)
| Código | Descripción |
|---|---|
| `FA0101` | Código CUPS inexistente o errado |
| `FA0201` | Diferencia entre lo prescrito y lo facturado |
| `FA0301` | Duplicidad en la facturación del servicio |

### Glosas por Tarifas (TA)
| Código | Descripción |
|---|---|
| `TA0101` | Tarifa cobrada superior a la del contrato |
| `TA0201` | CUPS no incluido en el manual tarifario pactado |

### Devoluciones (DE)
| Código | Descripción |
|---|---|
| `DE16` | Radicación extemporánea (> 22 días hábiles) |
| `DE01` | Radicación ante pagador incorrecto |
| `DE09` | Factura sin número de autorización |

---

## 6. CONVENCIONES DE COMMITS Y VERSIONAMIENTO

```
feat(fase-X): descripción breve del cambio
fix(modulo): descripción del bug corregido
data(raw): actualización de datos de muestra
model(train): ajuste de hiperparámetros v2
docs(tesis): actualización de resultados LaTeX

# Ejemplos:
feat(fase-1): implementar parser JSON RIPS con aplanamiento anidado
fix(validator): corregir cálculo de días hábiles para VAL-002
model(evaluate): agregar curva precision-recall para clase glosada
```

---

## 7. INSTRUCCIONES DIRECTAS AL ASISTENTE DE IA

> Estas instrucciones guían el comportamiento del asistente durante el desarrollo.

1. **Fase primero, pregunta después.** Antes de escribir código, confirma en qué fase y módulo estamos trabajando.
2. **No generar código de una fase futura** sin completar los entregables de la fase actual.
3. **Siempre incluir type hints y docstrings** en Google Style en todo código Python generado.
4. **Los datos en `data/raw/` son intocables.** Todo código generado debe leer de `data/raw/` pero escribir solo en `data/interim/` o `data/processed/`.
5. **Cuando generes código de validación** (`validator.py`), siempre referencia qué regla de negocio (VAL-XXX) y qué causal de la Res. 2284 estás implementando.
6. **Para cada función de Feature Engineering**, justifica la variable con su base clínica o regulatoria en el docstring.
7. **Reporta métricas siempre en el formato estándar** del §4 (Fase 4). Nunca reportes solo accuracy.
8. **Los gráficos para la tesis** siempre deben guardarse en `reports/figures/` con el script Python, no mostrados inline solamente.
9. **Cuando actualices `requirements.txt`**, incluye siempre la versión exacta (`paquete==X.Y.Z`).
10. **No usar `print()` en módulos `src/`.** Usar `logging.getLogger(__name__)`.
11. **Si detectas una inconsistencia entre los datos y este CLAUDE.MD**, señálala explícitamente antes de proceder.
12. **El idioma de los comentarios en notebooks** es español. El código fuente en `src/` usa comentarios en español también, variables en inglés.

---

## 8. GLOSARIO TÉCNICO CLAVE

| Término | Definición |
|---|---|
| **RIPS** | Registro Individual de Prestación de Servicios de Salud |
| **FEV** | Factura Electrónica de Venta (estándar DIAN) |
| **MUV** | Mecanismo Único de Validación (DIAN — valida FEV antes de radicación) |
| **CUPS** | Clasificación Única de Procedimientos en Salud |
| **CIE-10** | Clasificación Internacional de Enfermedades, 10ª revisión |
| **CUM** | Código Único de Medicamento |
| **NIT** | Número de Identificación Tributaria (identificador del prestador IPS) |
| **IPS** | Institución Prestadora de Servicios de Salud |
| **EPS** | Entidad Promotora de Salud |
| **Glosa** | Objeción total o parcial a una factura médica por parte del pagador |
| **Devolución** | Rechazo de forma de una factura (error técnico/administrativo) |
| **Case Mix** | Índice de complejidad ponderada del portafolio de servicios facturados |
| **XAI** | Explainable Artificial Intelligence (IA explicable) |
| **SHAP** | SHapley Additive exPlanations (técnica de explicabilidad de modelos ML) |
| **Feature Store** | Repositorio centralizado de características pre-calculadas para ML |
| **ETL** | Extract, Transform, Load (proceso de ingesta y transformación de datos) |

---

*Última actualización: 2026-03-11 · Autor: jcardonr · Proyecto EAFIT*
*Sesión 2026-03-04: Estructura agents/ + latex/ creada · Sistema IA Savia documentado · Regla BD solo-lectura añadida · 02_construccion_dataset.ipynb creado*
*Sesión 2026-03-05: Dataset 728K filas extraído y corregido · 5 bugs de calidad de datos resueltos · codigo_dx→nombre_dx documentado · query optimizada 8.2× · Fase 1C COMPLETADA*
*Sesión 2026-03-05: EDA completo (03_EDA.ipynb) · 12 análisis · 11 figuras a 300dpi · Benchmark IA Savia ejecutado · Fase 2 COMPLETADA*
*Sesión 2026-03-09: CLAUDE.MD actualizado — Fase 2 marcada como completada · Próximo paso: Fase 3 Feature Engineering*
*Sesión 2026-03-09: Feature Engineering completo (04_Feature_Engineering.ipynb) · 11 columnas eliminadas · 13 log-transforms · 4 flags · 6 features de interacción · Split temporal train/val/test · Fase 3 COMPLETADA*
*Sesión 2026-03-09: Notebook FE ejecutado y verificado · HALLAZGO CRÍTICO: val/test sin clase Glosada por right-censoring (ago-2025+ sin auditar)*
*Sesión 2026-03-09: RIGHT-CENSORING CORREGIDO · Dataset truncado a periodo confiable (mar-2024 a jul-2025) · Splits corregidos: train=515K, val=78K, test=40K · 3 clases en todos los splits · Notebook 04_FE celdas 7,10,11 corregidas y ejecutadas via nbconvert*
*Sesión 2026-03-09: Parquets verificados en disco · Fase 3 COMPLETADA al 100% · Listo para Fase 4 (Model Training)*
*Sesión 2026-03-09: MODEL TRAINING COMPLETADO (Fase 4) · XGBoost/LightGBM/RF entrenados · XGBoost mejor: F1=0.749, AUC=0.908, Recall Glosada=0.726 · 3 modelos serializados en models/artifacts/ · Métricas en reports/evaluations/ · 3 figuras a 300dpi · Análisis de umbral incluido · Recall Glosada pendiente de optimizar (meta 0.80)*
*Sesión 2026-03-10: EXPLICABILIDAD SHAP COMPLETADA (Fase 5) · 06_SHAP_Explainability.ipynb implementado y ejecutado · TreeExplainer exacto · 4 figuras SHAP a 300dpi · explain_prediction() con 16 descripciones en lenguaje natural y 9 causales Res.2284 · Top feature: log_num_items_con_autorizacion (SHAP=1.23) · Artefactos JSON+NPZ exportados · Fase 5 COMPLETADA al 100% · Listo para Fase 6 (API REST)*
*Sesión 2026-03-10: Auditoría de artefactos en disco · Corregido naming de notebooks en §2 (06_Model_Evaluation→no existe, 07_SHAP→06_SHAP) · Documentado que src/ está vacío (código en notebooks) · 19 figuras totales confirmadas (11 EDA + 1 FE + 3 MT + 4 SHAP) · 4 artefactos SHAP pendientes de commit*
*Sesión 2026-03-11: VALIDACIÓN CONTRA DATOS REALES (Fase 5B) · 07_Validacion_Datos_Reales.ipynb completado (12 celdas) · Test F1=0.7432 (consistente) · OOT F1=0.5941 (-20.1% degradación, SIGNIFICATIVA) · Concept drift: Glosada 40%→67.5% · 3 figuras val_*.png · Reporte JSON+MD generados*
*Sesión 2026-03-11: HALLAZGO CRÍTICO sobre columnas IA Savia · `ejecucion_glosa_aut_ia` contiene códigos FA (FA0101-FA5801), NO predicciones de clase · `estado_glosa_aut` es estado del pipeline (Pendiente/bot/IA ejecutado), NO ground truth · `estado_factura` es flujo administrativo (0-13), NO resultado de auditoría · Target actual (JOINs cm_auditoria_*) confirmado CORRECTO · Comparación "IA Previa" del notebook 07 tiene error de mapeo (todo cae a Auditada por mapeo textual sobre códigos numéricos) · Sección §1B corregida*
*Sesión 2026-03-11: TESIS main.tex ACTUALIZADA (Fase 7 ~85%) · Resumen/Abstract con métricas reales · Stack tecnológico corregido · Datos con volúmenes reales · 9 figuras integradas (22 total) · Rutas corregidas · Conclusiones y recomendaciones empíricas · Errores LaTeX (backticks) corregidos · PDF compila OK (5.2MB) · PROPUESTA_VALOR_MODELO.md actualizada con disponibilidad de features T=0 y diagrama de flujo operativo*
