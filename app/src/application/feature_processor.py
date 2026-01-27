from datetime import datetime
from typing import Optional

import pandas as pd

from src.config.settings import Settings


class FeatureProcessor:
    """
    Centraliza a lógica de engenharia de features para garantir consistência
    entre Treino (Historical) e Inferência (Live).
    """

    def process(self, df: pd.DataFrame, snapshot_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Transforma dados brutos em features prontas para o modelo.

        Args:
            df: DataFrame com dados brutos.
            snapshot_date: Data de referência para cálculos temporais (usado na inferência).
                           Se None, usa a data atual do sistema.
        """
        df = df.copy()

        # Define o ano de referência
        # LÓGICA ANTI-LEAKAGE:
        # 1. Se o dataset já tem 'ANO_REFERENCIA' (Treino histórico), usamos ele.
        # 2. Se não tem (Inferência), usamos o snapshot_date ou hoje.
        if "ANO_REFERENCIA" in df.columns:
            # Garante que é numérico para subtração
            referencia = pd.to_numeric(df["ANO_REFERENCIA"], errors="coerce")
        else:
            # Modo Inferência
            current_date = snapshot_date or datetime.now()
            referencia = current_date.year

        # --- Feature: TEMPO_NA_ONG ---
        if "ANO_INGRESSO" in df.columns:
            ano_ingresso = pd.to_numeric(df["ANO_INGRESSO"], errors="coerce")

            # Preenche ingresso vazio com a mediana (ou lógica de negócio)
            # Nota: Em produção real, idealmente usamos valores salvos do treino,
            # mas aqui usaremos a lógica simplificada para manter compatibilidade.
            if ano_ingresso.isnull().any():
                mediana = ano_ingresso.median() if not ano_ingresso.isnull().all() else referencia
                ano_ingresso = ano_ingresso.fillna(mediana)

            # O Cálculo Mágico unificado
            df["TEMPO_NA_ONG"] = referencia - ano_ingresso
            df["TEMPO_NA_ONG"] = df["TEMPO_NA_ONG"].clip(lower=0)  # Evita tempo negativo
        else:
            df["TEMPO_NA_ONG"] = 0

        # --- Garantia de Colunas (Schema Enforcement) ---
        required_cols = Settings.FEATURES_NUMERICAS + Settings.FEATURES_CATEGORICAS

        for col in required_cols:
            if col not in df.columns:
                # Se for feature numérica (ex: INDE_ANTERIOR) e não vier, preenche com 0
                df[col] = 0 if col in Settings.FEATURES_NUMERICAS else "N/A"

        # Remove colunas que não devem estar ali (Sanitização)
        # Garante que só passa o que está na whitelist
        df = df[required_cols]

        # --- Tratamento de Nulos Final ---
        # Numéricos -> 0 (ou média, conforme sua regra de negócio original)
        for col in Settings.FEATURES_NUMERICAS:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Categóricos -> 'N/A'
        for col in Settings.FEATURES_CATEGORICAS:
            df[col] = df[col].astype(str).replace('nan', 'N/A')

        return df[required_cols]
