from fastapi import APIRouter, HTTPException, Depends
from src.application.risk_service import RiskService
from src.domain.student import Student
from src.infrastructure.model.model_manager import ModelManager

# Instância global do gerenciador
model_manager = ModelManager()

def get_risk_service():
    """
    Factory atualizada:
    1. Obtém o modelo já carregado da memória (rápido).
    2. Injeta no Service.
    """
    try:
        model = model_manager.get_model()
        return RiskService(model=model)
    except RuntimeError as e:
        # Se o modelo não estiver disponível, a API deve retornar 503 (Service Unavailable)
        raise HTTPException(status_code=503, detail="Modelo de ML não inicializado.")

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
        try:
            return service.predict_risk(student.model_dump())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))