import os

import uvicorn
from fastapi import FastAPI

from src.api.controller import PredictionController


class App:
    def __init__(self):
        # 1. Inicializa o FastAPI
        self.app = FastAPI(
            title="Passos Mágicos - API de Previsão de Risco",
            description="API para predição de risco de defasagem escolar utilizando Machine Learning.",
            version="1.0.0"
        )

        # 2. Configura as Rotas
        self._configure_routes()

    def _configure_routes(self):
        """
        Instancia a Controller e registra suas rotas na aplicação.
        """
        prediction_controller = PredictionController()

        self.app.include_router(
            prediction_controller.router,
            prefix="/api/v1",
            tags=["Previsão"]
        )

        self.app.add_api_route("/health", self.health_check, methods=["GET"], tags=["Health"])

    @staticmethod
    def health_check():
        """Endpoint para verificar se a API está online."""
        return {"status": "ok", "service": "passos-magicos-api"}

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Inicia o servidor Uvicorn."""
        port = int(os.getenv("PORT", port))

        # Roda o servidor
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    application = App()
    application.run()
