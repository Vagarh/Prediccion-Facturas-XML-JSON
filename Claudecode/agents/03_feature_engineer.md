# Subagente: Ingeniero de Features — Savia Salud EPS

## Rol
Construye el vector de características final para el modelo a partir del dataset base.

## Input
`data/processed/YYYYMMDD_dataset_entrenamiento.parquet`

## Features a construir (ver CLAUDE.MD §3 Fase 3)

### Clínicas (`src/features/clinical_features.py`)
- `dx_sexo_inconsistente` — diagnóstico CIE-10 incompatible con sexo del paciente
- `complejidad_mismatch` — nivel CUPS > nivel habilitación IPS
- `tiene_anestesiologia` — cirugía mayor con anestesiólogo
- `cups_autorizado_contrato` — CUPS cubierto por el contrato

### Prestador (`src/features/provider_features.py`)
- `tasa_glosa_historica_nit` — % glosas históricas del NIT (últimos 12 meses)
- `tasa_devolucion_historica_nit` — % devoluciones históricas del NIT
- `concentracion_cups_prestador` — índice HHI de CUPS
- `trimestre_facturacion` — patrones de fin de período

### Financieras (`src/features/financial_features.py`)
- `discrepancia_fiscal_abs` — |Valor JSON - Valor XML|
- `dias_hasta_radicacion` — días entre fecha factura y radicación
- `porcentaje_mayor_item` — % del ítem de mayor valor sobre total
- `case_mix_index` — índice de complejidad ponderado

## Reglas anti-leakage (CRÍTICO)
NUNCA incluir como features:
- `valor_inicial_glosa`, `valor_pagado_factura`, `estado_factura`
- `ejecucion_glosa_aut_ia`, `estado_glosa_aut` (son resultados, no inputs)
- `nombre_completo_afiliado` (PII)
- Cualquier columna generada DESPUÉS de la radicación de la factura

## Output
`data/processed/YYYYMMDD_features_engineered.parquet`
