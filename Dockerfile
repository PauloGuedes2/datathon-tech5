FROM python:3.11-slim

WORKDIR /app

# Instala dependências do SO necessárias
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copia e instala requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código fonte
COPY . .

# Cria as pastas necessárias para evitar erro de permissão/existência
RUN mkdir -p app/models logs app/monitoring

# Define o PYTHONPATH para incluir a raiz
ENV PYTHONPATH=/app

# Comando de execução
# Ajustado para: uvicorn app.main:app (já que o main.py está dentro da pasta app)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]