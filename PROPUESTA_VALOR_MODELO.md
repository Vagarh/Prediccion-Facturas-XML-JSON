# Sistema Inteligente de Auditoría de Facturas Médicas
## Propuesta de Valor — Savia Salud EPS

---

> **Este documento tiene dos públicos:** la sección §1–§5 está dirigida al equipo directivo y de auditoría de Savia Salud EPS. La sección §6 es un resumen ejecutivo del estado del sistema para uso interno.

---

## 1. El Problema que Resuelve

El proceso de auditoría de cuentas médicas en Savia Salud EPS enfrenta un cuello de botella estructural:

- Una factura puede tardar **hasta 90 días** en obtener un estado definitivo (Aprobada, Glosada, Devuelta)
- La revisión depende **casi exclusivamente** de la capacidad humana del equipo auditor, sin distinción de complejidad o riesgo
- Los problemas —glosas, devoluciones— se detectan **tarde**, cuando ya consumieron tiempo valioso del equipo

**El resultado:** retrasos en pagos a prestadores, dificultad para planificación financiera y subutilización del talento especializado del equipo auditor.

---

## 2. La Solución Construida

Se desarrolló un **sistema de Machine Learning productivo** que analiza cada factura en el momento de su radicación y genera un perfil de riesgo completo antes de que un auditor la toque.

### Qué hace el sistema

| Capacidad | Detalle |
|---|---|
| **Predicción en tiempo real** | Clasifica cada factura en: Auditada / Glosada / Devuelta |
| **Probabilidades por clase** | No da un único veredicto — da `P(Auditada)`, `P(Glosada)`, `P(Devuelta)` |
| **Explicabilidad SHAP** | Indica *por qué* predice lo que predice, en lenguaje operacional |
| **API REST productiva** | Endpoint `/v1/predict` integrable con cualquier sistema existente |
| **Triaje automático** | Prioriza la cola de auditoría sin intervención humana |

### Qué insumos usa (ya disponibles en Savia Salud)

El modelo usa **48 características** extraídas exclusivamente de datos que el sistema ya tiene en el momento de la radicación:

| Fuente | Features | Disponibilidad |
|---|---|---|
| Cabecera de factura (`cm_facturas`) | 16 | Inmediata al radicar |
| Detalle de servicios (`cm_detalles`) | 21 | Automática al cargar XML/JSON |
| Perfil histórico del prestador | 11 | Pre-calculado, lookup instantáneo |

**No requiere que ningún auditor haya tocado la factura.**

---

## 3. Resultados Obtenidos

El modelo fue entrenado con **633,000 facturas reales** de Savia Salud (marzo 2024 – julio 2025).

### Métricas de desempeño

| Métrica | Resultado | Referencia |
|---|---|---|
| **AUC-ROC (OvR)** | **0.9078** | Meta: ≥ 0.85 ✅ |
| **F1-macro** | **0.7490** | Línea base: 0.0 (sin modelo) |
| **Recall Glosada** | **72.6%** | De cada 100 glosas, detecta 73 |
| **Precisión Glosada** | Alta | Baja tasa de falsos positivos |

### Ejemplos reales de predicción

**Factura de alto riesgo:**
```
P(Glosada):   97.9%  ← El auditor debe revisarla primero
P(Auditada):   2.1%
P(Devuelta):   0.0%
Factores: Prestador con tasa histórica de glosa 42% + 18 diagnósticos distintos
```

**Factura de bajo riesgo:**
```
P(Auditada):  92.9%  ← Candidata a revisión rápida o muestreo
P(Glosada):    7.1%
P(Devuelta):   0.0%
```

### Impacto económico estimado

Con un promedio histórico de **$6.8M COP por glosa detectada** y el modelo capturando el 72.6% de las glosas antes de que escalen:

> **1 glosa detectada proactivamente = ~$6.8M COP recuperados o evitados**
> Costo de infraestructura del sistema: **~$150K COP/mes** (~$38 USD)
> **ROI: 1 glosa = 45× el costo mensual del sistema**

---

## 4. Cómo se Integra al Flujo Actual

```
PRESTADOR radica factura (XML/JSON)
         ↓
SISTEMA procesa → cm_facturas + cm_detalles
(automático, segundos, sin intervención humana)
         ↓
MODELO ejecuta sobre 48 features
Resultado: [P(Auditada), P(Glosada), P(Devuelta)] + explicación SHAP
         ↓
    ┌────────────────────────────────────────┐
    │ BAJO RIESGO   → Revisión rápida/muest │
    │ ALTO RIESGO   → Auditor especializado │
    │               + factores SHAP         │
    │ ERROR DE FORMA→ Alerta al prestador   │
    └────────────────────────────────────────┘

Tiempo desde radicación hasta triaje: minutos (vs. hasta 90 días)
```

---

## 5. Estado del Sistema y Próximos Pasos

El sistema está **listo para piloto**. No es un prototipo en notebook — es código de producción.

### Lo que está construido y funcionando

| Componente | Estado | Descripción |
|---|---|---|
| Modelo XGBoost v1 | ✅ Productivo | Serializado, cargable, versionado |
| API REST (FastAPI) | ✅ Productivo | 4 endpoints, Pydantic v2, Dockerfile |
| Explicabilidad SHAP | ✅ Integrada | En tiempo real, en español, causales Res. 2284 |
| Suite de pruebas | ✅ 147 tests | 96.85% cobertura, 6 capas de validación |
| CI/CD | ✅ Activo | GitHub Actions, 3 pipelines automatizados |
| Arquitectura AWS | ✅ Diseñada | Lambda + SageMaker + CloudWatch + Step Functions |

### Lo que requiere para operar en producción

1. **Acceso de lectura** a `cm_facturas` y `cm_detalles` desde el entorno de despliegue
2. **Definición del umbral de triaje** con el equipo auditor (ajustable sin reentrenar)
3. **Reentrenamiento trimestral** — el sistema detecta drift automáticamente y alerta

---

## 6. Transparencia: Limitaciones Conocidas

| Limitación | Impacto | Mitigación |
|---|---|---|
| Concept drift detectado (OOT F1=0.59) | Degradación sin reentrenamiento | Reentrenamiento trimestral automático |
| Dataset de una sola EPS | Generalización no validada en otras EPS | Piloto acotado a Savia Salud |
| Recall Glosada = 72.6% (meta 80%) | ~27% de glosas no detectadas | Umbral ajustable, mejora con más datos |
| No incluye texto libre de auditores | Pierde señal clínica cualitativa | NLP como fase futura |

---

*Sistema desarrollado en el marco del proyecto de tesis de Maestría en Ciencia de Datos — Universidad EAFIT, 2026.*
*Datos proporcionados por Savia Salud EPS bajo acuerdo de confidencialidad.*
