import pandas as pd

from src.config.settings import Settings
from src.domain.student import Student
from src.infrastructure.model.ml_pipeline import MLPipeline


class RiskService:
    def __init__(self):
        """
        Inicializa o serviço de predição de risco.
        
        Carrega automaticamente o modelo ML treinado para uso.
        
        Attributes:
            ml_pipeline: Instância do pipeline de ML carregado
        """
        self.ml_pipeline = MLPipeline()
        self.ml_pipeline.load()

    def predict_risk(self, student: Student) -> dict:
        """
        Prediz o risco de defasagem escolar para um estudante.
        
        Args:
            student: Objeto Student com dados do estudante
            
        Returns:
            Dict com probabilidade, label e mensagem explicativa
            
        Formato de retorno:
            {
                "risk_probability": float,
                "risk_label": str,
                "message": str
            }
        """
        df = pd.DataFrame([student.data])

        prob = self.ml_pipeline.predict_proba(df)

        label = "ALTO RISCO" if prob > Settings.RISK_THRESHOLD else "BAIXO RISCO"

        return {
            "risk_probability": round(prob, 4),
            "risk_label": label,
            "message": f"O estudante possui {prob:.1%} de chance de defasagem."
        }
