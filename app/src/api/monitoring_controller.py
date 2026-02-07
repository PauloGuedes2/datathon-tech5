from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from src.application.monitoring_service import MonitoringService


def get_monitoring_service():
    """ Dependência para obter uma instância do MonitoringService."""
    return MonitoringService()


class MonitoringController:
    """ Controller para endpoints relacionados ao monitoramento do modelo e dashboard do Evidently AI."""
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route(
            "/dashboard",
            self._get_dashboard,
            methods=["GET"],
            response_class=HTMLResponse  # Importante: Retorna HTML, não JSON
        )

    @staticmethod
    async def _get_dashboard(service: MonitoringService = Depends(get_monitoring_service)):
        """
        Retorna o Dashboard do Evidently AI.
        """
        html_content = service.generate_dashboard()
        return HTMLResponse(content=html_content, status_code=200)
