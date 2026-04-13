# Demo MVP — Auditoría Predictiva Savia Salud EPS

Interfaz web interactiva para demostrar el pipeline completo de clasificación
predictiva de facturas médicas con XGBoost + embeddings semánticos + Ollama.

## Requisitos

```bash
pip install streamlit plotly pandas numpy
# Componentes LLM (opcionales):
pip install sentence-transformers spacy ollama
python -m spacy download es_core_news_md
```

Ollama (para explicaciones enriquecidas):
```bash
# Instalar: https://ollama.ai
ollama pull llama3.2
ollama serve
```

## Ejecución

Desde la raíz del proyecto `Codigo/`:

```bash
streamlit run demo/streamlit_app.py
```

La app abre en `http://localhost:8501`.

## Casos de demostración

| # | Caso | Descripción |
|---|------|-------------|
| 1 | Factura típica | Modelo base XGBoost sin campos textuales |
| 2 | Diagnóstico CIE-10 complejo | Enriquecimiento semántico con embeddings |
| 3 | Observaciones clínicas | Explicación narrativa LLM |
| 4 | Simulación de drift | Alertas PSI con distribución atípica |
| 5 | Comparación de versiones | Modelo base vs enriquecido |

## Tabs de la interfaz

- **Predicción** — Formulario de entrada + resultado con SHAP + explicación LLM
- **Monitoreo de Drift** — PSI por feature con umbrales visuales
- **Comparación de Modelos** — Métricas v1 vs v2 con gráficos interactivos
- **Administración** — Estado del registry, reentrenamiento, rollback
