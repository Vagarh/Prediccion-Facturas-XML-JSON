FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias primero para aprovechar cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar artefactos del modelo
COPY models/artifacts/20260309_xgboost_v1.json models/artifacts/
COPY data/processed/20260309_feature_list.txt data/processed/

# Copiar código fuente
COPY src/ src/

# Variables de entorno
ENV SAVIA_BASE_DIR=/app
ENV MODEL_VERSION=v1
ENV PYTHONPATH=/app

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
