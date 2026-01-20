import pandas as pd

from src.config.settings import Settings
from src.domain.student import Student
from src.infrastructure.model.ml_pipeline import MLPipeline


class RiskService:
    def __init__(self):
        self.ml_pipeline = MLPipeline()
        self.ml_pipeline.load()  # Tenta carregar ao iniciar

    def predict_risk(self, student: Student) -> dict:
        df = pd.DataFrame([student.data])

        prob = self.ml_pipeline.predict_proba(df)

        label = "ALTO RISCO" if prob > Settings.RISK_THRESHOLD else "BAIXO RISCO"

        return {
            "risk_probability": round(prob, 4),
            "risk_label": label,
            "message": f"O estudante possui {prob:.1%} de chance de defasagem."
        }
