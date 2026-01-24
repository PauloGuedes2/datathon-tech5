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
            logger.critical(f"Erro ao carregar modelo: {e}")
            return None

    def predict_risk(self, student_data: dict) -> dict:
        if not self.model:
            raise RuntimeError("Modelo indisponível.")

        # 1. Cria DataFrame e Prepara Features
        df = pd.DataFrame([student_data])
        df = self._prepare_features(df)

        try:
            # 2. Predição
            prob = self.model.predict_proba(df)[:, 1][0]
            risk_label = "ALTO RISCO" if prob > Settings.RISK_THRESHOLD else "BAIXO RISCO"

            # 3. LOG DE MONITORAMENTO (A novidade está aqui)
            # Salva os dados de entrada + a predição feita
            self._save_prediction_log(df, prob)

            return {
                "risk_probability": round(float(prob), 4),
                "risk_label": risk_label,
                "message": f"Probabilidade de risco: {prob:.1%}"
            }

        except Exception as e:
            logger.error(f"Erro na inferência: {e}")
            raise e

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        ano_atual = datetime.now().year

        if "ANO_INGRESSO" in df.columns:
            ano_ingresso = pd.to_numeric(df["ANO_INGRESSO"], errors="coerce")
            df["ANO_INGRESSO"] = ano_ingresso.fillna(ano_ingresso.median())
            df["TEMPO_NA_ONG"] = ano_atual - df["ANO_INGRESSO"]
            df["TEMPO_NA_ONG"] = df["TEMPO_NA_ONG"].clip(lower=0)
        else:
            df["TEMPO_NA_ONG"] = 0

        for col in Settings.FEATURES_NUMERICAS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        required_features = (
                Settings.FEATURES_NUMERICAS +
                Settings.FEATURES_CATEGORICAS
        )

        for col in required_features:
            if col not in df.columns:
                df[col] = 0 if col in Settings.FEATURES_NUMERICAS else "N/A"

        return df[required_features]

    def _save_prediction_log(self, df: pd.DataFrame, prob: float):
        """Salva a predição em um CSV acumulativo para o Evidently ler depois."""
        try:
            # Adiciona a predição e timestamp ao log
            log_df = df.copy()
            log_df["prediction"] = float(prob)
            log_df["timestamp"] = datetime.now()

            # Cria diretório se não existir
            os.makedirs(os.path.dirname(Settings.LOG_PATH), exist_ok=True)

            # Salva (append mode 'a'). Se arquivo não existe, escreve header.
            header = not os.path.exists(Settings.LOG_PATH)
            log_df.to_csv(Settings.LOG_PATH, mode='a', header=header, index=False)

        except Exception as e:
            # Erro de log não deve parar a API, apenas avisar
            logger.warning(f"Falha ao salvar log de monitoramento: {e}")
