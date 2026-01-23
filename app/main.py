import os

import uvicorn
from fastapi import FastAPI

from src.api.controller import PredictionController


class App:
    def __init__(self):
        """
        Inicializa a aplicação FastAPI com configurações e rotas.
        
        Configura:
            - Instância FastAPI com metadados
            - Registro de todas as rotas da aplicação
        """
        self.app = FastAPI(
            title="Passos Mágicos - API de Previsão de Risco",
            description="API para predição de risco de defasagem escolar utilizando Machine Learning.",
            version="1.0.0"
        )

        self._configure_routes()

    def _configure_routes(self):
        """
        Configura e registra todas as rotas da aplicação.
        
        Registra:
            - Rotas do PredictionController com prefixo /api/v1
            - Endpoint de health check na raiz
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
        """
        Endpoint de health check para monitoramento da API.
        
        Returns:
            Dict com status da aplicação
        """
        return {"status": "ok", "service": "passos-magicos-api"}

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Inicia o servidor Uvicorn para servir a API.
        
        Args:
            host: Endereço IP para bind (padrão: 0.0.0.0)
            port: Porta para servir a API (padrão: 8000)
            
        Note:
            A porta pode ser sobrescrita pela variável de ambiente PORT
        """
        port = int(os.getenv("PORT", port))
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    application = App()
    application.run()
