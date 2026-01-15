import pandas as pd

from src.config.params import Params
from src.logger.logger import logger


class DataLoader:

    @staticmethod
    def load_data() -> pd.DataFrame:
        """Carrega os dados do arquivo local (CSV ou Excel)."""
        try:
            path = Params.DATA_PATH
            logger.info(f"Carregando dados de: {path}")

            if path.endswith('.csv'):
                df = pd.read_csv(path, sep=';', encoding='utf-8')
            else:
                df = pd.read_excel(path)

            logger.info(f"Dados carregados: {df.shape[0]} linhas e {df.shape[1]} colunas.")
            return df
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            raise

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Realiza limpeza básica inicial baseada no Dicionário."""
        # Filtrar apenas dados relevantes de 2022 para treino
        df.columns = [str(col).strip().upper().replace(" ", "_") for col in df.columns]

        logger.info(f"Colunas normalizadas: {list(df.columns)}")

        # Converter colunas numéricas que podem estar como string
        for col in Params.FEATURES_NUMERICAS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Tratamento de Nulos
        df = df.fillna(0)

        return df
