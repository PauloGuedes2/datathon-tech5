from fastapi import APIRouter, HTTPException, Depends
from src.application.risk_service import RiskService
from src.domain.student import Student


def get_risk_service():
    """Factory para injeção de dependência."""
    return RiskService()


class PredictionController:
    def __init__(self):
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(
            path="/predict",
            endpoint=self.predict,
            methods=["POST"],
            response_model=dict,
        )

    @staticmethod
    async def predict(student: Student, service: RiskService = Depends(get_risk_service)):
        """
        Endpoint para predição de risco.
        O FastAPI usa a classe Student para validar o JSON de entrada automaticamente.
        """
        try:
            return service.predict_risk(student.model_dump())

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))