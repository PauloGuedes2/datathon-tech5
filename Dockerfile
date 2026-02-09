# --- Estágio 1: Builder (Compilação) ---
FROM python:3.11-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# --- Estágio 2: Runner (Execução Leve) ---
FROM python:3.11-slim as runner

WORKDIR /app

RUN groupadd -r appuser && useradd -r -g appuser appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Código da aplicação
COPY app/src/ ./src/
COPY app/main.py .

# Modelos treinados
COPY app/models/ ./models/

# ✅ Dados históricos (CSV / XLSX)
COPY app/data/ ./data/

RUN chown -R appuser:appuser /app

USER appuser

CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
