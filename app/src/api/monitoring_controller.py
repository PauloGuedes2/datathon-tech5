from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from src.application.monitoring_service import MonitoringService

def get_monitoring_service():
    return MonitoringService()

class MonitoringController:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route(
            "/dashboard",
            self.get_dashboard,
            methods=["GET"],
            response_class=HTMLResponse # Importante: Retorna HTML, não JSON
        )

    async def get_dashboard(self, service: MonitoringService = Depends(get_monitoring_service)):
        """
        Retorna o Dashboard do Evidently AI.
        """
        html_content = service.generate_dashboard()
        return HTMLResponse(content=html_content, status_code=200)