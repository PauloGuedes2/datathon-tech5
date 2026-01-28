import pandas as pd
import unicodedata
import re
import os

from pathlib import Path

# --- CONFIGURAÇÃO AUTOMÁTICA DE CAMINHOS ---
BASE_DIR = Path(__file__).resolve().parent.parent
FILE_NAME = "PEDE_PASSOS_DATASET_FIAP.xlsx"

caminhos_possiveis = [
    BASE_DIR / FILE_NAME,
    BASE_DIR / "data" / FILE_NAME,
    BASE_DIR / "app" / "data" / FILE_NAME
]

OUTPUT_FILE = BASE_DIR / "app"/ "monitoring" / "reference_data.csv"


def get_input_file():
    for path in caminhos_possiveis:
        if path.exists():
            return path
    return None


def normalize_column_name(col_name, year):
    # 1. Limpeza básica
    col_clean = str(col_name).upper().strip()
    col_clean = unicodedata.normalize('NFKD', col_clean).encode('ASCII', 'ignore').decode('utf-8')

    # 2. Remove o ano
    col_clean = re.sub(f'[ _-]{year}', '', col_clean)
    col_clean = col_clean.strip()

    # 3. MAPEAMENTOS ESPECÍFICOS

    if "NOME" in col_clean and "ANONIMIZADO" in col_clean:
        return 'NOME'
    if col_clean == "NOME":
        return 'NOME'
    if col_clean in ['NOME ALUNO', 'ALUNO', 'NM_ALUNO']:
        return 'NOME'

    # Variações de Matérias
    if col_clean in ['MAT', 'MATEM', 'MATEMATICA']: return 'NOTA_MAT'
    if col_clean in ['POR', 'PORT', 'PORTUG', 'PORTUGUES', 'LINGUA PORTUGUESA']: return 'NOTA_PORT'
    if col_clean in ['ING', 'INGL', 'INGLES']: return 'NOTA_ING'

    # Variações de Indicadores
    if col_clean in ['IND', 'INDE_CONCEITO']: return 'INDE'

    return col_clean


def preparar_dados():
    print("--- INICIANDO PREPARAÇÃO DA FEATURE STORE ---")

    input_path = get_input_file()
    if not input_path:
        print(f"❌ ERRO: Arquivo '{FILE_NAME}' não encontrado.")
        return

    try:
        print(f"📂 Lendo Excel: {input_path}")
        excel_file = pd.ExcelFile(input_path, engine='openpyxl')

        abas_2023 = [sheet for sheet in excel_file.sheet_names if '2023' in sheet]
        if not abas_2023:
            print("❌ Erro: Nenhuma aba de 2023 encontrada.")
            return

        aba_alvo = abas_2023[0]
        print(f"✅ Processando aba: '{aba_alvo}'")
        df = pd.read_excel(input_path, sheet_name=aba_alvo, engine='openpyxl')

    except Exception as e:
        print(f"❌ Erro crítico ao ler o Excel: {e}")
        return

    print("🔄 Normalizando colunas...")

    # Normaliza
    mapa_colunas = {col: normalize_column_name(col, 2023) for col in df.columns}
    df.rename(columns=mapa_colunas, inplace=True)
    df['ANO_REFERENCIA'] = 2023

    # --- DEBUG FINAL ---
    if 'NOME' not in df.columns:
        print("\n❌ AINDA NÃO ENCONTROU A COLUNA 'NOME'.")
        print("🔍 Colunas atuais:", df.columns.tolist())
        return

    required_cols = ['NOME', 'INDE', 'IAA', 'IEG', 'IPS', 'IDA', 'IPP', 'IPV', 'IAN']

    # Salva CSV
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    cols_to_save = [c for c in df.columns if c in required_cols or c == 'ANO_REFERENCIA']
    df_final = df[cols_to_save].dropna(subset=['NOME'])

    df_final.to_csv(OUTPUT_FILE, index=False)

    print(f"💾 Feature Store salva com sucesso em:\n   -> {OUTPUT_FILE}")
    print(f"📊 Total de registros válidos: {len(df_final)}")


if __name__ == "__main__":
    preparar_dados()