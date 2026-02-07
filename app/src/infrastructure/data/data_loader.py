import glob
import os
import re

import pandas as pd
import unicodedata

from src.config.settings import Settings
from src.util.logger import logger


class DataLoader:
    """
    Responsável pelo carregamento, limpeza e unificação dos dados históricos (2022, 2023, 2024).
    """

    def load_data(self) -> pd.DataFrame:
        """
        Busca arquivos Excel na pasta de dados, processa as abas por ano e unifica em um único DataFrame.
        """
        search_path = os.path.join(Settings.DATA_DIR, "*.xlsx")
        excel_files = glob.glob(search_path)

        logger.info(f"Buscando arquivos em: {search_path}")

        if not excel_files:
            try:
                contents = os.listdir(Settings.DATA_DIR)
                logger.error(f"Conteúdo encontrado em {Settings.DATA_DIR}: {contents}")
            except Exception as e:
                pass
            raise FileNotFoundError(f"Nenhum arquivo .xlsx encontrado em {Settings.DATA_DIR}")

        file_path = excel_files[0]
        logger.info(f"Carregando arquivo Excel: {file_path}")

        try:
            sheets_dict = pd.read_excel(file_path, sheet_name=None)
        except Exception as e:
            logger.error(f"Erro crítico ao ler o Excel: {e}")
            raise e

        all_data = []

        for sheet_name, df_sheet in sheets_dict.items():
            ano_match = re.search(r'202\d', sheet_name)

            if not ano_match:
                logger.warning(f"Aba '{sheet_name}' ignorada (não contém ano no nome).")
                continue

            ano_completo = int(ano_match.group())
            logger.info(f"Processando aba: {sheet_name} (Ano {ano_completo})")

            df_sheet = self._process_dataframe(df_sheet, ano_completo)
            df_sheet['ANO_REFERENCIA'] = ano_completo

            all_data.append(df_sheet)

        if not all_data:
            raise RuntimeError("Nenhuma aba válida carregada do Excel.")

        try:
            final_df = pd.concat(all_data, ignore_index=True)
        except Exception as e:
            logger.error(f"Erro ao concatenar os dados: {e}")
            raise e

        final_df = final_df.fillna(0)

        logger.info(f"Dataset Total Unificado: {final_df.shape}")
        return final_df

    @staticmethod
    def _process_dataframe(df: pd.DataFrame, ano_full: int) -> pd.DataFrame:
        new_cols = []
        ano_short = int(str(ano_full)[-2:])

        for col in df.columns:
            col_clean = str(col).upper().strip()
            col_clean = unicodedata.normalize('NFKD', col_clean).encode('ASCII', 'ignore').decode('utf-8')
            col_clean = re.sub(f'[ _]{ano_full}', '', col_clean)
            col_clean = re.sub(f'[ _]{ano_short}$', '', col_clean)

            if col_clean in ['RA', 'ID_ALUNO', 'CODIGO_ALUNO', 'MATRICULA']:
                col_clean = 'RA'
            elif col_clean in ['MAT', 'MATEM', 'MATEMATICA']:
                col_clean = 'NOTA_MAT'
            elif col_clean in ['POR', 'PORT', 'PORTUG', 'PORTUGUES']:
                col_clean = 'NOTA_PORT'
            elif col_clean in ['ING', 'INGL', 'INGLES']:
                col_clean = 'NOTA_ING'
            elif col_clean in ['DEFAS', 'DEFASAGEM']:
                col_clean = 'DEFASAGEM'
            elif "ANO" in col_clean and "INGRESSO" in col_clean:
                col_clean = "ANO_INGRESSO"

            if "INST" in col_clean and "ENSINO" in col_clean: col_clean = "INSTITUICAO_ENSINO"
            if "PONTO" in col_clean and "VIRADA" in col_clean: col_clean = "PONTO_VIRADA"
            if "PSICOLOGIA" in col_clean and "REC" in col_clean: col_clean = "REC_PSICOLOGIA"

            new_cols.append(col_clean)

        df.columns = new_cols

        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        if 'RA' in df.columns:
            df['RA'] = df['RA'].astype(str).str.strip()

        return df
