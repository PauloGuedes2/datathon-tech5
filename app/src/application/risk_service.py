import pandas as pd

from src.application.feature_processor import FeatureProcessor
from src.config.settings import Settings
from src.infrastructure.logging.prediction_logger import PredictionLogger
from src.util.logger import logger


class RiskService:
    def __init__(self, model):
        self.model = model
        self.processor = FeatureProcessor()  # Instancia o processador
        self.logger = PredictionLogger()  # Instancia o Logger Seguro

    def predict_risk(self, student_data: dict) -> dict:
        if not self.model:
            raise RuntimeError("Serviço indisponível: Modelo não inicializado.")

        try:
            # 1. Processamento (Já corrigido no Passo 2)
            raw_df = pd.DataFrame([student_data])
            features_df = self.processor.process(raw_df)

            # 2. Inferência
            prob_risk = self.model.predict_proba(features_df)[:, 1][0]
            prediction_class = int(prob_risk > Settings.RISK_THRESHOLD)
            risk_label = "ALTO RISCO" if prediction_class == 1 else "BAIXO RISCO"

            result = {
                "risk_probability": round(float(prob_risk), 4),
                "risk_label": risk_label,
                "prediction": prediction_class
            }

            # 3. Logging Seguro (Requisito 3)
            # Convertemos o DataFrame de features para dict para salvar no JSON
            features_dict = features_df.to_dict(orient="records")[0]

            # Chamada Thread-Safe
            self.logger.log_prediction(
                features=features_dict,
                prediction_data=result
            )

            return result

        except Exception as e:
            logger.error(f"Erro na inferência: {e}")
            raise e
