"""
Ponto de entrada da API FastAPI.

Responsabilidades:
- Configurar a aplicação FastAPI
- Registrar rotas e eventos
- Inicializar recursos no startup
"""

import os

import uvicorn
from fastapi import FastAPI, HTTPException

from src.api.controller import ControladorPredicao
from src.api.monitoring_controller import ControladorMonitoramento
from src.infrastructure.model.model_manager import GerenciadorModelo
from src.util.logger import logger

app = FastAPI(
    title="Passos Mágicos - API de Risco",
    description="API com Monitoramento de Data Drift (Evidently).",
    version="2.1.0",
)


@app.on_event("startup")
async def evento_inicializacao():
    """
    Executa ações de inicialização da aplicação.

    Responsabilidades:
    - Registrar log de inicialização
    - Carregar o modelo na memória

    Retorno:
    - None: não retorna valor
    """
    logger.info("Inicializando recursos da API...")
    GerenciadorModelo().carregar_modelo()


controlador_predicao = ControladorPredicao()
app.include_router(controlador_predicao.roteador, prefix="/api/v1", tags=["Predição"])

controlador_monitoramento = ControladorMonitoramento()
app.include_router(controlador_monitoramento.roteador, prefix="/api/v1/monitoring", tags=["Observabilidade"])


@app.get("/health", tags=["Infraestrutura"])
def checar_saude():
    """
    Endpoint de health check.

    Retorno:
    - dict: status da aplicação
    """
    try:
        GerenciadorModelo().obter_modelo()
        return {"status": "ok"}
    except Exception as erro:
        raise HTTPException(status_code=503, detail=str(erro))


if __name__ == "__main__":
    porta = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=porta)
