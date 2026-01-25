import os
from datetime import datetime

import pandas as pd
from joblib import load

from src.config.settings import Settings
from src.util.logger import logger


class RiskService:
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        try:
            return load(Settings.MODEL_PATH)
        except Exception as e:
            logger.critical(f"Erro fatal: Modelo não encontrado em {Settings.MODEL_PATH}. Treine-o primeiro.")
            return None

    def predict_risk(self, student_data: dict) -> dict:
        if not self.model:
            raise RuntimeError("Serviço indisponível: Modelo não carregado.")

        # 1. Preparação
        df = pd.DataFrame([student_data])
        df = self._prepare_features(df)

        try:
            # 2. Inferência
            # Probabilidade da classe positiva (1 = Alto Risco)
            prob_risk = self.model.predict_proba(df)[:, 1][0]
            prediction_class = int(prob_risk > Settings.RISK_THRESHOLD)  # 0 ou 1

            risk_label = "ALTO RISCO" if prediction_class == 1 else "BAIXO RISCO"

            # 3. Observabilidade (Logging)
            self._save_prediction_log(df, prediction_class, prob_risk)

            return {
                "risk_probability": round(float(prob_risk), 4),
                "risk_label": risk_label,
                "prediction": prediction_class
            }

        except Exception as e:
            logger.error(f"Erro na inferência: {e}")
            raise e

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Feature Engineering em tempo real (igual ao treino)
        ano_atual = datetime.now().year
        if "ANO_INGRESSO" in df.columns:
            ano_ingresso = pd.to_numeric(df["ANO_INGRESSO"], errors="coerce")
            # Aqui usamos um valor default seguro se vier vazio
            df["TEMPO_NA_ONG"] = ano_atual - ano_ingresso.fillna(ano_atual)
            df["TEMPO_NA_ONG"] = df["TEMPO_NA_ONG"].clip(lower=0)
        else:
            df["TEMPO_NA_ONG"] = 0

        # Garante estrutura exata do modelo
        required = Settings.FEATURES_NUMERICAS + Settings.FEATURES_CATEGORICAS
        for col in required:
            if col not in df.columns:
                df[col] = 0 if col in Settings.FEATURES_NUMERICAS else "N/A"

        return df[required]

    def _save_prediction_log(self, df: pd.DataFrame, pred_class: int, prob: float):
        """Salva log para o Evidently ler depois como 'Current Data'."""
        try:
            log_df = df.copy()
            # Precisamos salvar a 'prediction' (classe) para o TargetDrift funcionar
            log_df["prediction"] = pred_class
            log_df["probability"] = prob
            log_df["timestamp"] = datetime.now().isoformat()

            header = not os.path.exists(Settings.LOG_PATH)
            os.makedirs(os.path.dirname(Settings.LOG_PATH), exist_ok=True)
            log_df.to_csv(Settings.LOG_PATH, mode='a', header=header, index=False)

        except Exception as e:
            logger.warning(f"Falha no logging (não afeta resposta da API): {e}")
