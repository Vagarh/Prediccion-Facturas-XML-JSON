# Proyecto de Grado - Ciencia de Datos

Estructura de carpetas para el desarrollo del modelo predictivo de auditoría prestacional.

## Estructura del DirectorIO

- `data/`
    - `raw/`: Datos originales e inmutables (JSON RIPS, XML FEV).
    - `interim/`: Datos transformados y limpios en etapas intermedias.
    - `processed/`: El conjunto de datos final listo para el entrenamiento del modelo.
- `notebooks/`: Jupyter Notebooks para Análisis Exploratorio de Datos (EDA) y prototipado.
- `src/`: Código fuente modularizado para ser utilizado en el proyecto.
    - `data/`: Scripts para cargar o generar datos.
    - `features/`: Scripts para transformar datos crudos en características para modelado.
    - `models/`: Scripts para entrenar modelos y realizar predicciones.
    - `visualization/`: Scripts para crear visualizaciones exploratorias y de resultados.
- `models/`: Modelos entrenados y serializados (.pkl, .h5), predicciones o resúmenes.
- `reports/`: Análisis generados como HTML, PDF, etc.
    - `figures/`: Gráficos y figuras generadas para ser usadas en la tesis de LaTeX.
- `requirements.txt`: Archivo de requisitos para reproducir el entorno de análisis.
