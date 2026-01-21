import re

import pandas as pd
import unicodedata

from src.config.settings import Settings
from src.util.logger import logger


class DataLoader:
    """
    Responsável pelo carregamento e pré-processamento dos dados.
    
    Funcionalidades:
        - Carregamento de arquivos Excel
        - Normalização de colunas
        - Tratamento de valores nulos
    """

    def load_data(self) -> pd.DataFrame:
        """
        Carrega e processa o dataset principal.
        
        Returns:
            DataFrame com dados limpos e colunas normalizadas
            
        Features:
            - Carregamento de arquivo Excel
            - Normalização de nomes de colunas
            - Preenchimento de valores nulos com 0
        """
        logger.info(f"Carregando dados de: {Settings.DATA_PATH}")
        df = pd.read_excel(Settings.DATA_PATH)
        return self._clean_columns(df)

    @staticmethod
    def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza nomes de colunas e trata valores nulos.
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame com colunas normalizadas e valores nulos preenchidos
            
        Transformações:
            - Remove acentos e caracteres especiais
            - Converte para maiúsculo
            - Substitui espaços por underscore
            - Preenche NaN com 0
        """

        def normalize(col):
            col = str(col).strip().upper()
            # Substitui múltiplos espaços/tabs/quebras por um único underscore
            col = re.sub(r'\s+', '_', col)
            col = unicodedata.normalize("NFKD", col).encode("ASCII", "ignore").decode("utf-8")
            return col

        df.columns = [normalize(col) for col in df.columns]
        df = df.fillna(0)
        logger.info(f"Colunas normalizadas: {list(df.columns)}")
        return df
