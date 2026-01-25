import pandas as pd
import requests
import time
import random
import os
import glob
import warnings

from src.config.settings import Settings

# Ignora avisos do Excel
warnings.simplefilter(action='ignore', category=UserWarning)

API_URL = "http://localhost:8000/api/v1/predict"


def load_real_data():
    print(f"📂 Buscando arquivo Excel em {Settings.DATA_DIR}...")
    files = glob.glob(os.path.join(Settings.DATA_DIR, "*.xlsx"))

    if not files:
        raise FileNotFoundError(f"Nenhum arquivo .xlsx encontrado em {Settings.DATA_DIR}")

    file_path = files[0]
    print(f"📊 Carregando: {file_path}")

    # Lê todas as abas
    xls = pd.ExcelFile(file_path)

    # Tenta achar a aba de 2024 (que provavelmente foi seu set de teste)
    sheet_2024 = next((s for s in xls.sheet_names if "2024" in s), None)

    if sheet_2024:
        print(f"📅 Usando aba específica: {sheet_2024}")
        df = pd.read_excel(xls, sheet_name=sheet_2024)
    else:
        print("⚠️ Aba 2024 não encontrada. Usando a primeira aba disponível.")
        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

    return df


def map_columns(df):
    # Mapeamento: Nome no Excel -> Nome na API
    # Ajuste os nomes da esquerda conforme estão no seu Excel
    mapping = {
        'Idade': 'IDADE',
        'Ano ingresso': 'ANO_INGRESSO',
        'Gênero': 'GENERO',
        'Turma': 'TURMA',
        'Instituição de ensino': 'INSTITUICAO_ENSINO',
        'Fase': 'FASE'
    }

    # Normaliza nomes das colunas do Excel (remove espaços extras)
    df.columns = [c.strip() for c in df.columns]

    # Filtra apenas as colunas necessárias e renomeia
    available_cols = [c for c in mapping.keys() if c in df.columns]
    df_clean = df[available_cols].rename(columns=mapping)

    # Limpeza de dados para JSON
    df_clean = df_clean.dropna()  # Remove linhas com nulos

    if 'IDADE' in df_clean.columns:
        df_clean['IDADE'] = df_clean['IDADE'].astype(int)

    if 'ANO_INGRESSO' in df_clean.columns:
        df_clean['ANO_INGRESSO'] = df_clean['ANO_INGRESSO'].astype(int)

    return df_clean


def simulate_traffic(df, n_requests=100):
    print(f"\n🚀 Iniciando simulação de tráfego 'Zero Drift' com {n_requests} requisições...")

    records = df.to_dict(orient='records')
    random.shuffle(records)

    # Seleciona amostra
    sample = records[:n_requests]

    sucessos = 0

    for i, student in enumerate(sample):
        try:
            # Envia para a API
            response = requests.post(API_URL, json=student, timeout=5)

            if response.status_code == 200:
                data = response.json()
                risk = data.get("risk_label")
                print(f"[{i + 1}/{n_requests}] ✅ {risk} | {student['TURMA']}")
                sucessos += 1
            else:
                print(f"[{i + 1}/{n_requests}] ❌ Erro {response.status_code}: {response.text}")

            time.sleep(0.1)  # Pequeno delay

        except Exception as e:
            print(f"⚠️ Erro: {e}")

    print(f"\n🏁 Simulação finalizada. Sucessos: {sucessos}/{n_requests}")
    print("👉 Atualize o dashboard do Evidently. As colunas devem voltar a ficar VERDES.")


if __name__ == "__main__":
    try:
        df_raw = load_real_data()
        df_ready = map_columns(df_raw)

        # Verifica se temos colunas suficientes
        expected = ['IDADE', 'ANO_INGRESSO', 'GENERO', 'TURMA', 'INSTITUICAO_ENSINO', 'FASE']
        missing = [col for col in expected if col not in df_ready.columns]

        if missing:
            print(f"❌ Erro: Colunas não encontradas no Excel: {missing}")
            print("Colunas disponíveis:", df_raw.columns.tolist())
        else:
            simulate_traffic(df_ready, n_requests=100)

    except Exception as e:
        print(f"❌ Erro crítico: {e}")