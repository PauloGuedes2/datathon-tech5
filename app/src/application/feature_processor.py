from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd

from src.config.settings import Settings


class FeatureProcessor:
    """
    Centraliza a lógica de engenharia de features para garantir consistência
    entre Treino (Historical) e Inferência (Live).
    """

    def process(self, df: pd.DataFrame, snapshot_date: Optional[datetime] = None,
                stats: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Transforma dados brutos em features prontas para o modelo.

        Args:
            df: DataFrame com dados brutos.
            snapshot_date: Data de referência para cálculos temporais (usado na inferência).
                           Se None, usa a data atual do sistema.
            stats: Dicionário com estatísticas calculadas no TREINO (ex: mediana_ano_ingresso).
                   Essencial para evitar Data Leakage.
        """
        df = df.copy()

        # Define o ano de referência
        if "ANO_REFERENCIA" in df.columns:
            referencia = pd.to_numeric(df["ANO_REFERENCIA"], errors="coerce")
        else:
            current_date = snapshot_date or datetime.now()
            referencia = current_date.year

        # --- Feature: TEMPO_NA_ONG ---
        # Correção de Data Leakage: Usa estatística do treino se fornecida
        if "ANO_INGRESSO" in df.columns:
            ano_ingresso = pd.to_numeric(df["ANO_INGRESSO"], errors="coerce")

            if ano_ingresso.isnull().any():
                if stats and "mediana_ano_ingresso" in stats:
                    # Usa valor fixo do treino (Correto)
                    mediana = stats["mediana_ano_ingresso"]
                else:
                    # Fallback (Inferência sem stats ou treino inicial)
                    mediana = ano_ingresso.median() if not ano_ingresso.isnull().all() else 2020

                ano_ingresso = ano_ingresso.fillna(mediana)

            df["TEMPO_NA_ONG"] = referencia - ano_ingresso
            df["TEMPO_NA_ONG"] = df["TEMPO_NA_ONG"].clip(lower=0)
        else:
            df["TEMPO_NA_ONG"] = 0

        # --- Garantia de Colunas (Schema Enforcement) ---
        required_cols = Settings.FEATURES_NUMERICAS + Settings.FEATURES_CATEGORICAS

        for col in required_cols:
            if col not in df.columns:
                df[col] = 0 if col in Settings.FEATURES_NUMERICAS else "N/A"

        # Remove colunas que não devem estar ali (apenas features finais)
        # Nota: Mantemos df original separado se precisarmos dele depois,
        # mas aqui retornamos o dataset processado.
        df_processed = df[required_cols].copy()

        # --- Tratamento de Nulos Final ---
        for col in Settings.FEATURES_NUMERICAS:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)

        for col in Settings.FEATURES_CATEGORICAS:
            df_processed[col] = df_processed[col].astype(str).replace('nan', 'N/A')

        return df_processed
