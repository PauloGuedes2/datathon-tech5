import pandas as pd
import unicodedata

from src.config.settings import Settings
from src.util.logger import logger


class DataLoader:
    def load_data(self) -> pd.DataFrame:
        logger.info(f"Carregando dados de: {Settings.DATA_PATH}")
        df = pd.read_excel(Settings.DATA_PATH)
        return self._clean_columns(df)

    @staticmethod
    def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
        def normalize(col):
            col = str(col).strip().upper().replace(" ", "_")
            col = unicodedata.normalize("NFKD", col).encode("ASCII", "ignore").decode("utf-8")
            return col

        df.columns = [normalize(col) for col in df.columns]
        df = df.fillna(0)
        logger.info(f"Colunas normalizadas: {list(df.columns)}")
        return df
