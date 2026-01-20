from fastapi import APIRouter, HTTPException, Depends

from src.api.schemas import StudentDTO
from src.application.risk_service import RiskService
from src.domain.student import Student


def get_risk_service():
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
    def predict(data: StudentDTO, service: RiskService = Depends(get_risk_service)):
        try:
            student = Student(data=data.dict())
            return service.predict_risk(student)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

