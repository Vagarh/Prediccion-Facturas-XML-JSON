# Analítica Predictiva para Auditoría Prestacional en Salud
## Universidad EAFIT · Maestría en Ciencia de Datos · 2026
### Juan Felipe Cardona Arango · Savia Salud EPS

---

## El Problema

El sistema de salud colombiano exige que las EPS auditen **cada factura** antes de pagarla. En Savia Salud EPS esto significa revisar manualmente más de **600.000 facturas al año**, generando:

- Rezagos de hasta **90 días** en el pago a prestadores
- Equipos de auditores saturados revisando el 100% del volumen
- Incertidumbre financiera para clínicas y hospitales de la red
- Riesgo de glosas no detectadas con pérdidas de hasta **10.5× el valor promedio de una factura auditada**

La Resolución 2275 de 2023 cambió las reglas: ahora toda factura llega en formato digital estandarizado (JSON/XML), con datos estructurados disponibles **antes de que un auditor la toque**. Esto abre la puerta a la predicción automática.

---

## La Solución

Un modelo de **inteligencia artificial** que predice, en el momento de la radicación, si una factura será:

| Clase | Descripción |
|-------|-------------|
| **Auditada** | Aprobada sin observaciones |
| **Glosada** | Rechazada parcial o totalmente (mayor impacto financiero) |
| **Devuelta** | Devuelta por errores formales |

El modelo procesa **48 características** extraídas automáticamente de los datos de radicación — sin intervención humana — y genera una predicción con explicación en segundos.

---

## Datos y Metodología

- **633.000 facturas** reales de Savia Salud EPS (marzo 2024 – julio 2025)
- División temporal estricta (nunca aleatoria) para evitar data leakage:
  - Train: 515.145 facturas
  - Validación: 77.843 facturas
  - Test: 40.193 facturas (junio–julio 2025)
- Metodología: **CRISP-DM** adaptada al dominio de salud colombiano
- Algoritmos evaluados: XGBoost, LightGBM, Random Forest

---

## Resultados del Modelo

### Comparativa de algoritmos

| Algoritmo | F1-Score Macro | AUC-ROC | Recall Glosada |
|-----------|---------------|---------|---------------|
| XGBoost v1 | 0.7490 | 0.9078 | 0.7261 |
| LightGBM | 0.7441 | 0.9063 | 0.7124 |
| Random Forest | 0.7217 | 0.8966 | 0.7406 |
| **XGBoost v2 (enriquecido)** | 0.7114 | 0.9028 | **0.8961 ✅** |

> **Meta operativa: Recall Glosada ≥ 0.80 — SUPERADA**

### ¿Qué significa Recall Glosada = 0.8961?

De cada 100 facturas que realmente serán glosadas, el modelo detecta **89 antes de que el auditor las revise**. Esto permite triaje inteligente: el auditor prioriza automáticamente las facturas de mayor riesgo.

---

## Hallazgo Crítico: Deriva del Concepto

La validación con 223.621 facturas de agosto–diciembre 2025 reveló una degradación del **20.1% en F1-Score**:

| Métrica | Test (jun–jul 2025) | OOT (ago–dic 2025) | Cambio |
|---------|--------------------|--------------------|--------|
| F1-Macro | 0.749 | 0.599 | -20.1% |
| Recall Glosada | 0.726 | 0.543 | -25.2% |

**Causa:** La proporción de facturas glosadas saltó del 40.2% al 67.5% en el segundo semestre de 2025, cambio atribuido a nuevos requisitos de la Circular 007 de 2025.

**Implicación:** Un modelo de ML en producción **no es un artefacto estático** — requiere monitoreo continuo y reentrenamiento periódico.

---

## Explicabilidad con SHAP

El modelo no es una caja negra. Cada predicción viene acompañada de una explicación que identifica qué variables influyeron y en qué dirección:

### Variables más importantes globalmente

1. **Riesgo compuesto** (`log_valor_x_tasa_glosa`): valor de factura × tasa histórica de glosa del prestador
2. **Complejidad administrativa** (`tiene_autorizacion_items`): presencia de ítems con autorización previa
3. **Diversidad diagnóstica** (`log_num_dx_distintos`): número de diagnósticos CIE-10 distintos
4. **Perfil del prestador** (`log_tasa_glosa_prestador`): historial de glosas del prestador

### ¿Por qué esto importa?
El auditor puede ver en tiempo real: *"Esta factura fue clasificada como Glosada principalmente porque el prestador tiene una tasa histórica de glosa del 68% y el valor de la factura es 5× el promedio"*. Esto genera **confianza** en el sistema y permite supervisión humana informada.

---

## Arquitectura Técnica

```
[Radicación FEV/RIPS]
        │
        ▼
[API REST FastAPI]  ──→  [Feature Engineering (48 vars)]
        │                        │
        │                        ▼
        │              [XGBoost v2 Model]
        │                        │
        ▼                        ▼
[Respuesta JSON]  ◄──  [SHAP Explainer]
  • clase predicha
  • probabilidades
  • explicación SHAP
```

**Stack tecnológico:**
- Modelo: XGBoost + Sentence Transformers (embeddings clínicos)
- API: FastAPI + Pydantic v2
- Explicabilidad: SHAP
- Privacidad: 100% local, sin APIs externas (Ley 1581 Colombia)
- Infraestructura: AWS (ECS Fargate + Lambda + S3 + RDS)
- Testing: 380 pruebas automatizadas, 83.24% cobertura

---

## Impacto Esperado

| Indicador | Sin modelo | Con modelo | Mejora |
|-----------|-----------|------------|--------|
| Facturas revisadas manualmente | 100% | ~15–20% (triaje) | -80% carga |
| Tiempo hasta primera auditoría | >30 días | <1 día | -97% |
| Detección de glosas | Depende del auditor | 89.6% automático | Estandarizado |
| Glosas no detectadas (falsos negativos) | Variable | ~10.4% | Cuantificado |

---

## Contribución Original

Este es el **primer estudio** que integra variables clínicas y financieras de los estándares FEV-RIPS del sistema de salud colombiano para clasificación multiclase de estados de auditoría prestacional.

Los hallazgos son replicables para cualquier EPS que opere bajo la Resolución 2275 de 2023.

---

## Próximos Pasos

1. **Piloto controlado** en Savia Salud EPS: triaje de facturas en 3 colas de riesgo (bajo/medio/alto)
2. **Integración con MUV** (Mecanismo Único de Validación) para detección de errores formales en tiempo real
3. **Reentrenamiento trimestral** automatizado con detección de drift (sistema ya implementado)
4. **Extensión a predicción de monto**: no solo el estado, sino el valor esperado de la glosa

---

## Información del Proyecto

| Atributo | Detalle |
|----------|---------|
| Autor | Juan Felipe Cardona Arango |
| Institución | Universidad EAFIT · Maestría en Ciencia de Datos |
| Organización | Savia Salud EPS |
| Año | 2026 |
| Repositorio | Disponible bajo solicitud |
| Tecnologías | Python · XGBoost · FastAPI · SHAP · Sentence Transformers · AWS |
