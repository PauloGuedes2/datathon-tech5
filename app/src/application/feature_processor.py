from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd

from src.config.settings import Settings

""" Módulo responsável por processar os dados de entrada, gerando as features necessárias para o modelo."""
class FeatureProcessor:
    @staticmethod
    def process(df: pd.DataFrame, snapshot_date: Optional[datetime] = None, stats: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Processa o DataFrame de entrada, gerando as features necessárias para o modelo."""
        df = df.copy()

        if "ANO_REFERENCIA" in df.columns:
            referencia = pd.to_numeric(df["ANO_REFERENCIA"], errors="coerce")
        else:
            current_date = snapshot_date or datetime.now()
            referencia = current_date.year

        if "ANO_INGRESSO" in df.columns:
            ano_ingresso = pd.to_numeric(df["ANO_INGRESSO"], errors="coerce")

            if ano_ingresso.isnull().any():
                if stats and "mediana_ano_ingresso" in stats:
                    mediana = stats["mediana_ano_ingresso"]
                else:
                    mediana = ano_ingresso.median() if not ano_ingresso.isnull().all() else 2020

                ano_ingresso = ano_ingresso.fillna(mediana)

            df["TEMPO_NA_ONG"] = referencia - ano_ingresso
            df["TEMPO_NA_ONG"] = df["TEMPO_NA_ONG"].clip(lower=0)
        else:
            df["TEMPO_NA_ONG"] = 0

        required_cols = Settings.FEATURES_NUMERICAS + Settings.FEATURES_CATEGORICAS

        for col in required_cols:
            if col not in df.columns:
                df[col] = 0 if col in Settings.FEATURES_NUMERICAS else "N/A"

        df_processed = df[required_cols].copy()

        for col in Settings.FEATURES_NUMERICAS:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)

        for col in Settings.FEATURES_CATEGORICAS:
            df_processed[col] = df_processed[col].astype(str).replace('nan', 'N/A')

        return df_processed
