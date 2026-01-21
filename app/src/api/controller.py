from fastapi import APIRouter, HTTPException, Depends

from src.api.schemas import StudentDTO
from src.application.risk_service import RiskService
from src.domain.student import Student


def get_risk_service():
    """
    Factory function para injeção de dependência do RiskService.
    
    Returns:
        Instância do RiskService para uso nos endpoints
    """
    return RiskService()


class PredictionController:
    def __init__(self):
        """
        Inicializa o controller de predições.
        
        Configura automaticamente todas as rotas relacionadas a predições.
        
        Attributes:
            router: APIRouter do FastAPI com rotas configuradas
        """
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        """
        Registra todas as rotas do controller no router.
        
        Rotas configuradas:
            - POST /predict: Endpoint de predição de risco
        """
        self.router.add_api_route(
            path="/predict",
            endpoint=self.predict,
            methods=["POST"],
            response_model=dict,
        )

    @staticmethod
    def predict(data: StudentDTO, service: RiskService = Depends(get_risk_service)):
        """
        Endpoint para predição de risco de defasagem escolar.
        
        Args:
            data: Dados do estudante validados pelo schema
            service: Instância do serviço de predição (injetada)
            
        Returns:
            Dict com resultado da predição
            
        Raises:
            HTTPException: Em caso de erro interno (status 500)
        """
        try:
            student = Student(data=data.dict())
            return service.predict_risk(student)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

