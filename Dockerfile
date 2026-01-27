# --- Estágio 1: Builder (Compilação) ---
FROM python:3.11-slim as builder

# Evita arquivos .pyc e buffer de logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instala dependências do sistema necessárias para compilar pacotes Python (se houver)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Cria ambiente virtual para isolar dependências
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instala dependências do projeto
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Estágio 2: Runner (Execução Leve) ---
FROM python:3.11-slim as runner

WORKDIR /app

# Cria um usuário não-root por segurança
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copia o ambiente virtual do estágio anterior
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Variáveis de ambiente para o Python e Render
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Copia o código fonte da aplicação
COPY src/ ./src/
COPY main.py .

# Copia a pasta de dados e modelos iniciais (necessário para a API subir na primeira vez)
# NOTA: Crie a pasta 'models' localmente e coloque um .joblib lá antes de buildar, 
# ou o build pode falhar ou a API não subir.
COPY models/ ./models/
# COPY data/ ./data/  <-- Opcional, se precisar de dados de referência iniciais

# Ajusta permissões para o usuário não-root (caso precise escrever logs na pasta temporária)
# No Render, logs devem ir para stdout, mas se salvar em arquivo localmente:
RUN chown -R appuser:appuser /app

# Muda para o usuário seguro
USER appuser

# O Render injeta a variável PORT automaticamente. 
# Usamos o shell form para expandir a variável $PORT.
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"