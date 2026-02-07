from fastapi import APIRouter, HTTPException, Depends

from src.application.risk_service import RiskService
from src.domain.student import Student, StudentInput
from src.infrastructure.model.model_manager import ModelManager

model_manager = ModelManager()


def get_risk_service():
    """ Dependência para obter uma instância do RiskService. Verifica se o modelo está disponível antes de criar o serviço."""
    try:
        model = model_manager.get_model()
        return RiskService(model=model)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Modelo de ML não inicializado. {str(e)}")

""" Controller de predição, responsável por expor as rotas de API para predição de risco."""
class PredictionController:
    def __init__(self):
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        """Registra as rotas de predição."""
        self.router.add_api_route(
            path="/predict/full",
            endpoint=self._predict,
            methods=["POST"],
            response_model=dict,
        )

        self.router.add_api_route(
            path="/predict/smart",
            endpoint=self._predict_smart,
            methods=["POST"],
            response_model=dict,
            summary="Predição com busca automática de histórico"
        )

    @staticmethod
    async def _predict(student: Student, service: RiskService = Depends(get_risk_service)):
        """Predição tradicional, onde o cliente envia o modelo completo do aluno."""
        try:
            return service.predict_risk(student.model_dump())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def _predict_smart(student_input: StudentInput, service: RiskService = Depends(get_risk_service)):
        """" Predição inteligente, onde o cliente envia apenas os dados básicos e o sistema busca o histórico automaticamente."""
        try:
            return service.predict_risk_smart(student_input)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
