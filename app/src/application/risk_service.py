import pandas as pd

from src.application.feature_processor import FeatureProcessor
from src.config.settings import Settings
from src.domain.student import StudentInput, Student
from src.infrastructure.data.historical_repository import HistoricalRepository
from src.infrastructure.logging.prediction_logger import PredictionLogger
from src.util.logger import logger

""" Módulo de Serviço para Predição de Risco de Defasagem Acadêmica."""
class RiskService:
    def __init__(self, model):
        self.model = model
        self.processor = FeatureProcessor()
        self.logger = PredictionLogger()
        self.repository = HistoricalRepository()

    def predict_risk(self, student_data: dict) -> dict:
        """Realiza a predição de risco com base nos dados do aluno."""
        if not self.model:
            raise RuntimeError("Serviço indisponível: Modelo não inicializado.")

        try:
            raw_df = pd.DataFrame([student_data])
            features_df = self.processor.process(raw_df)

            prob_risk = self.model.predict_proba(features_df)[:, 1][0]
            prediction_class = int(prob_risk > Settings.RISK_THRESHOLD)
            risk_label = "ALTO RISCO" if prediction_class == 1 else "BAIXO RISCO"

            result = {
                "risk_probability": round(float(prob_risk), 4),
                "risk_label": risk_label,
                "prediction": prediction_class
            }

            features_dict = features_df.to_dict(orient="records")[0]

            self.logger.log_prediction(
                features=features_dict,
                prediction_data=result
            )

            return result

        except Exception as e:
            logger.error(f"Erro na inferência: {e}")
            raise e

    def predict_risk_smart(self, input_data: StudentInput) -> dict:
        """
        Método inteligente que busca histórico automaticamente.
        """

        history_features = self.repository.get_student_history(input_data.RA)

        if history_features:
            logger.info(f"Histórico encontrado para RA: {input_data.RA}")
        else:
            logger.info(f"Aluno novo ou sem histórico (RA: {input_data.RA})")
            history_features = {
                "INDE_ANTERIOR": 0.0,
                "IAA_ANTERIOR": 0.0,
                "IEG_ANTERIOR": 0.0,
                "IPS_ANTERIOR": 0.0,
                "IDA_ANTERIOR": 0.0,
                "IPP_ANTERIOR": 0.0,
                "IPV_ANTERIOR": 0.0,
                "IAN_ANTERIOR": 0.0,
                "ALUNO_NOVO": 1
            }

        full_student_data = input_data.model_dump()
        full_student_data.update(history_features)

        student_domain = Student(**full_student_data)

        return self.predict_risk(student_domain.model_dump())
