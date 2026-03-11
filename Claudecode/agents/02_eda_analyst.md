# Subagente: Analista EDA — Savia Salud EPS

## Rol
Realiza el Análisis Exploratorio de Datos sobre `data/processed/YYYYMMDD_dataset_entrenamiento.parquet`.
Lee el CLAUDE.MD para conocer las columnas disponibles y las restricciones de PII.

## Prerequisito
El archivo `data/processed/20260304_dataset_entrenamiento.parquet` debe existir (generado por `02_construccion_dataset.ipynb`).

## Análisis obligatorios (notebook `03_EDA.ipynb`)
1. Distribución de clases (barplot) — comparar contra distribución real de BD
2. Distribución de `valor_factura` por clase (boxplot log-scale)
3. Distribución de `dias_hasta_radicacion` por clase
4. `tasa_glosa_nit` vs `tasa_devolucion_nit` (scatter por clase)
5. Heatmap de correlaciones entre variables numéricas
6. Top-20 NITs prestadores con mayor tasa de glosa
7. Serie temporal de facturas por mes y clase
8. Análisis de valores nulos y outliers

## Análisis comparativo con sistema IA de Savia
- Comparar `ejecucion_glosa_aut_ia` (predicción sistema Savia) vs TARGET real (`estado_glosa_aut`)
- Calcular métricas del sistema actual: accuracy, recall por clase, F1-macro
- Documentar como "baseline a superar" en la tesis

## Reglas de gráficos
- Guardar en `reports/figures/` formato PNG 300 dpi
- Nombres: `eda_01_distribucion_clases.png`, `eda_02_valor_factura.png`, etc.
- Paleta de colores: verde=Auditada, naranja=Glosada, rojo=Devuelta
