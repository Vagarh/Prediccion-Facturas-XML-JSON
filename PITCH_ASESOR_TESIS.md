# Propuesta de Asesoría — Tesis de Maestría
## "Sistema Predictivo para la Auditoría Prestacional en Savia Salud EPS"
### Maestría en Ciencia de Datos — Universidad EAFIT · 2026

---

## Lo esencial en 30 segundos

**El trabajo ya está hecho.** No busco a alguien que me ayude a construir el proyecto — busco un asesor que lo revise, lo valide académicamente y lo avale. El código corre, el modelo está entrenado, la tesis está redactada y el PDF compila.

Su rol sería de **revisor y validador**, no de co-constructor.

---

## El proyecto en números

| Qué | Cuánto |
|---|---|
| Datos reales de Savia Salud EPS | 633,000 facturas (16 meses) |
| Modelo XGBoost entrenado | AUC-ROC **0.9078**, F1-macro **0.749** |
| Features de entrada | 48 variables (ingeniería de features completa) |
| Líneas de código de producción | ~2,500 líneas en módulos `src/` |
| Tests automatizados | **147 tests**, cobertura **96.85%** |
| Páginas de tesis redactadas | ~120 páginas, PDF compila sin errores |
| Notebooks de análisis | 9 notebooks (EDA → Modelado → Validación → API) |

---

## Lo que está 100% completo

### Datos y Modelado
- [x] Extracción de datos reales desde BD MySQL de Savia Salud (554 tablas mapeadas)
- [x] EDA completo con 11 figuras a 300dpi
- [x] Feature engineering: 48 features (log-transforms, flags binarios, interacciones)
- [x] Corrección de right-censoring en splits temporales (hallazgo crítico documentado)
- [x] Entrenamiento comparativo: XGBoost vs LightGBM vs Random Forest
- [x] Explicabilidad SHAP con TreeExplainer exacto (1,998 facturas, 4.7s)
- [x] Validación OOT (ago-dic 2025): concept drift documentado y analizado
- [x] Cruce con ground truth de `cm_auditoria_motivos_glosas` confirmado correcto

### Sistema de Software
- [x] API REST productiva (FastAPI, Pydantic v2, 4 endpoints)
- [x] Dockerfile con healthcheck
- [x] Suite de 147 pruebas automatizadas (unit, integration, performance, security, regression)
- [x] 96.85% de cobertura de código
- [x] CI/CD con GitHub Actions (3 pipelines)
- [x] Arquitectura AWS diseñada (Lambda, SageMaker, CloudWatch, Step Functions)

### Tesis
- [x] Marco teórico con referencias IEEE
- [x] Marco metodológico (8 fases documentadas)
- [x] Resultados con tablas y figuras integradas
- [x] Capítulo de Discusión (viabilidad, concept drift, limitaciones)
- [x] Conclusiones y recomendaciones
- [x] PDF compilado, 5.2 MB, sin errores LaTeX

---

## Lo que falta (su rol)

Lo que genuinamente necesito de un asesor:

1. **Revisión del enfoque metodológico** — ¿el diseño del experimento es sólido académicamente?
2. **Validación de las afirmaciones** — ¿las conclusiones están respaldadas por los resultados?
3. **Aval institucional** — firma para entrega formal a EAFIT y a Savia Salud EPS
4. **Feedback puntual** sobre el capítulo de Discusión y Limitaciones

Estimo que la carga de revisión es de **4-6 horas en total**, distribuidas en 2-3 reuniones.

---

## Por qué este proyecto es atractivo para un asesor

### Impacto real y verificable
- Datos reales de la EPS pública más grande de Antioquia
- Problema con impacto económico directo (glosas en salud = millones de pesos)
- Sistema desplegable, no un prototipo académico

### Rigor técnico demostrable
- Código abierto en repositorio git con historial completo
- 147 tests con 96.85% de cobertura — estándar de ingeniería de software profesional
- Documentación de hallazgos críticos (right-censoring, concept drift, ground truth)
- Comparación de 3 algoritmos con métricas estándar de la literatura

### Novedad académica
- Aplicación de XAI (SHAP) a auditoría prestacional colombiana — no hay papers en español sobre esto específicamente con datos de RIPS/FEV
- Alineación con Resolución 2284 (causales de glosa) como framework de explicabilidad
- Análisis de concept drift en el contexto del sistema de salud colombiano (post-COVID cambio de patrones)

### Convenio con industria
- Existe obligación formal de entrega a Savia Salud EPS (acuerdo por uso de datos)
- El asesor quedaría asociado a un proyecto con impacto institucional en el sector salud

---

## Sobre el concepto drift (la pregunta difícil respondida de antemano)

El modelo baja de F1=0.74 (test) a F1=0.59 (OOT, ago-dic 2025). Este es un hallazgo, no un defecto oculto.

**Por qué esto es fortaleza académica, no debilidad:**
- Está documentado, analizado y explicado en el capítulo de Discusión
- La causa es identificada: la proporción de facturas glosadas pasó de 40% a 67% en ese período (cambio estructural post-auditoría)
- La solución está diseñada: reentrenamiento trimestral automático vía SageMaker
- Es coherente con la literatura de healthcare fraud detection (Chan et al., 2022; Joshi & Yesha, 2020)
- **Un modelo que ignora el drift sería el problema; este lo detecta y lo gestiona**

---

## Siguiente paso

Si le interesa revisar el material antes de comprometerse:

- **PDF de la tesis:** disponible inmediatamente (120+ páginas, compila sin errores)
- **Demo de la API:** el sistema corre localmente en 2 minutos con Docker
- **Repositorio:** historial git completo con cada fase del proyecto

Propongo una primera reunión de 45 minutos para revisar el PDF y el demo en vivo.

---

*Proyecto desarrollado íntegramente por el candidato. Los datos de Savia Salud EPS fueron provistos bajo acuerdo de confidencialidad para uso académico exclusivo.*
