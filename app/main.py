import os

import uvicorn
from fastapi import FastAPI

from src.api.controller import PredictionController
from src.api.monitoring_controller import MonitoringController
from src.infrastructure.model.model_manager import ModelManager
from src.util.logger import logger

app = FastAPI(
    title="Passos Mágicos - API de Risco",
    description="API com Monitoramento de Data Drift (Evidently).",
    version="2.1.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("Inicializando recursos da API...")
    # Carrega o modelo na memória IMEDIATAMENTE ao iniciar o servidor
    ModelManager().load_model()

# Rota de Predição
prediction_controller = PredictionController()
app.include_router(prediction_controller.router, prefix="/api/v1", tags=["Predição"])

# Rota de Monitoramento (Nova)
monitoring_controller = MonitoringController()
app.include_router(monitoring_controller.router, prefix="/api/v1/monitoring", tags=["Observabilidade"])


@app.get("/health", tags=["Infraestrutura"])
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
