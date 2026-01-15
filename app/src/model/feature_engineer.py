import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from src.config.params import Params


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}

    def fit(self, X, y=None):
        # Aprender codificação para variáveis categóricas
        for col in Params.FEATURES_CATEGORICAS:
            if col in X.columns:
                le = LabelEncoder()
                # Converter para string para garantir
                le.fit(X[col].astype(str))
                self.label_encoders[col] = le
        return self

    def transform(self, X):
        X_out = X.copy()

        # Aplica encoding
        for col, le in self.label_encoders.items():
            if col in X_out.columns:
                # Trata valores não vistos no treino
                X_out[col] = X_out[col].astype(str).map(
                    lambda s: le.transform([s])[0] if s in le.classes_ else -1
                )

        return X_out

    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria a variável alvo 'RISCO_DEFASAGEM'.
        """

        # Verifica se as colunas existem antes de aplicar a regra
        if 'INDE_22' in df.columns and 'IDA' in df.columns:
            condition = (df['INDE_22'] < 6.0) | (df['IDA'] < 6.0)
            df['RISCO_DEFASAGEM'] = np.where(condition, 1, 0)
        else:
            # Fallback caso algo ainda esteja errado, evita crash
            print("Aviso: Colunas para cálculo do target não encontradas. Criando target zerado.")
            df['RISCO_DEFASAGEM'] = 0

        return df