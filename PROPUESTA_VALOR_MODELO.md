# Propuesta de Valor: Sistema Inteligente para la Auditoría de Facturas Médicas

## 1. El Problema: Un Proceso de Auditoría Lento y Reactivo

El proceso actual de auditoría de cuentas médicas en Savia Salud EPS, aunque riguroso, enfrenta tres desafíos críticos que limitan la eficiencia y generan incertidumbre financiera:

*   **Ciclos de Decisión Extensos:** Una factura puede tardar **hasta 90 días** en ser auditada y tener un estado definitivo (Aprobada, Glosada o Devuelta).
*   **Proceso Manual Intensivo:** La auditoría depende de una revisión humana para casi la totalidad de las facturas, sin importar su complejidad o riesgo inherente. Esto consume una cantidad enorme de recursos de personal especializado.
*   **Gestión Reactiva:** Los problemas (glosas, devoluciones) se descubren tarde en el ciclo, cuando la factura ya ha consumido tiempo valioso del equipo de auditoría. No existe un mecanismo para anticipar el riesgo en el momento de la radicación.

Esta situación crea un cuello de botella operativo, retrasa los pagos a la red de prestadores y dificulta la planificación financiera precisa.

## 2. La Solución: Un Modelo Predictivo con Explicabilidad Integrada

Hemos desarrollado un modelo de Machine Learning que analiza cada factura **tan pronto como el sistema procesa su radicación**. El modelo utiliza 48 características extraídas de tres fuentes que están disponibles sin intervención humana:

| Fuente de datos | # Features | Disponibilidad | Ejemplo |
|---|---|---|---|
| **Cabecera de factura** (`cm_facturas`) | 16 | Inmediata al radicar | Valor, régimen, tipo contrato, FEV |
| **Detalle de servicios** (`cm_detalles`) | 21 | Automática al cargar XML/JSON | # CUPS, diagnósticos, autorizaciones, valores por línea |
| **Perfil histórico del prestador** (lookup) | 11 | Pre-calculado | Tasa de glosa histórica, volumen, valor promedio |

**Ninguna de estas características requiere que un auditor haya revisado la factura.** Todas provienen del documento electrónico del prestador y del historial acumulado del sistema. Esto significa que el modelo puede ejecutarse **antes de que la factura entre a la cola de auditoría**, eliminando la espera de 90 días para conocer el riesgo.

En lugar de simplemente predecir un único resultado, el modelo asigna una **probabilidad a cada uno de los tres posibles estados**:

*   **% de probabilidad de ser Auditada** (cumple con los requisitos)
*   **% de probabilidad de ser Glosada** (contiene inconsistencias de fondo)
*   **% de probabilidad de ser Devuelta** (contiene errores de forma)

Esto significa que, de manera inmediata, se obtiene un perfil de riesgo completo para cada factura. Por ejemplo, los resultados reales del modelo muestran perfiles de riesgo muy claros:

*   **Perfil de Alto Riesgo de Glosa:**
    *   `Probabilidad de ser Glosada: 97.9%`
    *   `Probabilidad de ser Auditada: 2.1%`
    *   `Probabilidad de ser Devuelta: 0.0%`

*   **Perfil de Bajo Riesgo (Alta probabilidad de Auditoría):**
    *   `Probabilidad de ser Auditada: 92.9%`
    *   `Probabilidad de ser Glosada: 7.1%`
    *   `Probabilidad de ser Devuelta: 0.0%`

El modelo, basado en XGBoost, alcanza un **AUC-ROC (OvR) de 0.9078**, lo que demuestra una alta capacidad para discriminar entre las diferentes clases de facturas.

### Precisión sobre el momento de predicción

Es importante ser transparentes: el modelo no opera sobre el XML crudo del prestador de forma aislada. Requiere que el sistema de radicación haya procesado el documento y poblado las tablas `cm_facturas` y `cm_detalles` en la base de datos. En el flujo actual de Savia Salud, este procesamiento es **automático y ocurre como parte de la radicación**, por lo que el modelo puede ejecutarse inmediatamente después sin esperar intervención humana.

El cuello de botella de 90 días no es por falta de datos — **es por la cola de trabajo del equipo de auditoría**. El modelo ataca exactamente ese cuello de botella: permite priorizar la cola antes de que un auditor toque la factura.

Sin embargo, el verdadero valor del modelo no reside solo en su capacidad de predicción.

## 3. El Diferenciador Clave: De la Predicción a la Acción Inteligente

Este no es un modelo de "caja negra". Gracias al análisis de explicabilidad (XAI) mediante SHAP, podemos saber **exactamente por qué** el modelo toma cada decisión. El sistema no solo dice "qué" va a pasar, sino que explica el "porqué", transformando la auditoría en un proceso inteligente.

### ¿Cómo funciona en la práctica?

El modelo identifica los "impulsores de riesgo" para cada factura. En lugar de reglas simplistas, el modelo ha aprendido patrones complejos a partir de los datos históricos. Por ejemplo:

*   **¿Por qué se predice una GLOSA?**
    No es solo por el alto valor. El modelo lo predice por una **combinación de factores** como:
    1.  **Riesgo Compuesto (`log_valor_x_tasa_glosa`):** La factura tiene un valor alto **Y** el prestador tiene un historial de alta tasa de glosas.
    2.  **Complejidad Clínica (`log_num_dx_distintos`):** La factura involucra una gran cantidad de diagnósticos diferentes, lo que incrementa la probabilidad de una inconsistencia clínica.
    3.  **Complejidad Administrativa (`log_num_items_con_autorizacion`):** La factura contiene muchos servicios que requirieron autorización previa, aumentando los puntos de posible fallo.

*   **¿Por qué se predice una DEVOLUCIÓN?**
    El modelo detecta patrones de **errores de forma**, como:
    1.  **Falta de Detalles (`log_num_items_con_autorizacion` = 0):** Una factura sin detalles de autorización es un fuerte predictor de que será devuelta por información incompleta.
    2.  **Indicadores de FEV (`tiene_fev`):** La ausencia de una Factura Electrónica de Venta (FEV) correctamente estructurada es una señal de alerta.

### Hacia un Nuevo Flujo de Auditoría Inteligente

Esta capacidad de explicación sienta las bases para rediseñar el proceso, pasando de una cola única a un **triaje inteligente**. Aunque requiere mayor desarrollo y validación antes de una automatización completa, el modelo ya permite visualizar un flujo futuro:

1.  **Facturas de Bajo Riesgo (Predicción: Auditada):**
    *   **Acción Potencial:** Identificar estas facturas para un **posible flujo de revisión rápida** o una auditoría por muestreo, en lugar de una revisión exhaustiva individual.
    *   **Beneficio a Futuro:** Liberar al personal de auditoría para que se concentre en casos de mayor complejidad.

2.  **Facturas con Indicios de Error de Forma (Predicción: Devuelta):**
    *   **Acción Potencial:** Generar **alertas tempranas** para que un equipo administrativo verifique el error formal y notifique al prestador, antes de que la factura entre al ciclo de auditoría médica.
    *   **Beneficio a Futuro:** Reducir el ciclo de corrección de semanas a días, evitando que errores de forma consuman tiempo del equipo de auditoría.

3.  **Facturas de Alto Riesgo (Predicción: Glosada):**
    *   **Acción Inmediata:** Asignar estas facturas con **prioridad a un auditor médico especializado**. El auditor recibe la factura junto con los **factores de riesgo clave identificados por el modelo** (p. ej., "Riesgo alto por combinación de valor y tasa de glosa del prestador").
    *   **Beneficio Inmediato:** El auditor enfoca su atención en los puntos críticos, reduciendo el tiempo de revisión por caso.

## 4. Flujo Operativo Propuesto

```
┌─────────────────────────────────────────────────────────────────┐
│  PRESTADOR radica factura (XML/JSON)                            │
└──────────────────────┬──────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  SISTEMA procesa automáticamente → cm_facturas + cm_detalles    │
│  (segundos a minutos, sin intervención humana)                  │
└──────────────────────┬──────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  MODELO PREDICTIVO ejecuta sobre las 48 features                │
│  Resultado: [P(Auditada), P(Glosada), P(Devuelta)]             │
│  + Factores explicativos SHAP por factura                       │
└──────┬───────────────┬──────────────────┬───────────────────────┘
       ▼               ▼                  ▼
  ┌─────────┐   ┌────────────┐   ┌──────────────┐
  │ BAJO    │   │ ALTO       │   │ ERROR DE     │
  │ RIESGO  │   │ RIESGO     │   │ FORMA        │
  │ (Audit) │   │ (Glosa)    │   │ (Devolución) │
  └────┬────┘   └─────┬──────┘   └──────┬───────┘
       ▼              ▼                  ▼
  Revisión        Auditor            Alerta al
  rápida o        especializado      prestador
  muestreo        + factores SHAP    para corrección
```

**Tiempo desde radicación hasta triaje: minutos (vs. hasta 90 días en el flujo actual).**

## 5. Beneficios Tangibles para Savia Salud EPS

La implementación de este sistema representa un salto cualitativo en la gestión, con beneficios directos:

*   **Optimización de Recursos:** El talento humano especializado (auditores médicos) se enfoca exclusivamente en los casos complejos y de alto valor donde su experiencia es crucial.
*   **Reducción Drástica de Tiempos:** El ciclo de vida de una factura se reduce significativamente, mejorando el flujo de caja y la relación con la red de prestadores.
*   **Decisiones Basadas en Datos:** La auditoría deja de ser un arte subjetivo para convertirse en una ciencia informada por patrones históricos y análisis de riesgo cuantitativo.
*   **Gestión Proactiva del Riesgo:** Permite identificar sistemáticamente a los prestadores o los tipos de factura que generan más glosas, abriendo la puerta a intervenciones y capacitaciones dirigidas.

## 6. Consideraciones de Transparencia

Para una adopción responsable, es importante documentar las limitaciones conocidas:

*   **Concept drift detectado:** La distribución de clases cambia en el tiempo (la proporción de glosas pasó de 40% a 67% entre junio y diciembre de 2025). Esto implica que el modelo requiere **re-entrenamiento periódico** (recomendado: trimestral) para mantener su precisión.
*   **Rendimiento OOT:** En datos fuera de periodo de entrenamiento (ago-dic 2025), el F1-score baja de 0.74 a 0.59. Esto es esperado por el drift y se mitiga con reentrenamiento.
*   **Dependencia del procesamiento:** El modelo requiere que `cm_detalles` esté poblado. Si hay retrasos en la carga de detalles del XML, la predicción se retrasa proporcionalmente.
*   **No reemplaza al auditor:** El modelo es una herramienta de **triaje y priorización**, no un sistema de decisión autónoma. La decisión final de glosar o devolver sigue siendo del auditor médico.

## 7. Conclusión

Este proyecto no es solo un modelo predictivo; es una **herramienta estratégica de inteligencia de negocio**. Permite a Savia Salud EPS transformar su proceso de auditoría de un centro de costo reactivo a un sistema de gestión de riesgo proactivo, eficiente e inteligente, alineado con los desafíos del sector salud actual.

La pregunta clave no es "¿puede el modelo predecir?" — ya demostró que sí (AUC-ROC 0.91). La pregunta es: **¿cuánto valor se pierde cada día que las facturas esperan en cola sin priorización?** Este modelo elimina esa espera.
