from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.monitoring_controller import MonitoringController, get_monitoring_service


def test_dashboard_endpoint():
    app = FastAPI()
    controller = MonitoringController()

    service = Mock()
    service.generate_dashboard.return_value = "<html>ok</html>"

    def override():
        return service

    app.dependency_overrides[get_monitoring_service] = override
    app.include_router(controller.router, prefix="/api/v1/monitoring")

    client = TestClient(app)

    response = client.get("/api/v1/monitoring/dashboard")

    assert response.status_code == 200
    assert "<html>ok</html>" in response.text
