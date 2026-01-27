import pandas as pd
import requests
import time
import random
import os
import glob
import warnings
import re

from src.config.settings import Settings

# Ignora avisos do Excel e Pandas
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

API_URL = "http://localhost:8000/api/v1/predict"


def load_real_data():
    print(f"📂 Buscando arquivo Excel em {Settings.DATA_DIR}...")
    files = glob.glob(os.path.join(Settings.DATA_DIR, "*.xlsx"))

    if not files:
        raise FileNotFoundError(f"Nenhum arquivo .xlsx encontrado em {Settings.DATA_DIR}")

    file_path = files[0]
    print(f"📊 Carregando: {file_path}")

    xls = pd.ExcelFile(file_path)

    # Prioriza aba de 2024 ou 2023 para simulação
    target_sheet = next((s for s in xls.sheet_names if "2024" in s),
                        next((s for s in xls.sheet_names if "2023" in s), xls.sheet_names[0]))

    print(f"📅 Usando aba: {target_sheet}")
    df = pd.read_excel(xls, sheet_name=target_sheet)
    return df


def find_column(df, keywords):
    """Tenta encontrar uma coluna no DF que contenha uma das keywords."""
    if isinstance(keywords, str): keywords = [keywords]

    for col in df.columns:
        col_norm = str(col).upper().strip()
        for k in keywords:
            # Regex simples para achar ex: "INDE_2023" ou "INDE"
            if re.search(rf"\b{k}\b", col_norm):
                return col
    return None


def map_columns(df):
    print("🛠️  Mapeando e normalizando colunas para o novo Schema...")

    # 1. Normaliza colunas básicas
    column_mapping = {
        'Idade': 'IDADE',
        'Ano ingresso': 'ANO_INGRESSO',
        'Gênero': 'GENERO',
        'Turma': 'TURMA',
        'Instituição de ensino': 'INSTITUICAO_ENSINO',
        'Fase': 'FASE',
        'Pedra': 'PEDRA'
    }

    df_ready = pd.DataFrame()

    # Mapeia colunas existentes no Excel
    for excel_name, api_name in column_mapping.items():
        found = find_column(df, excel_name.upper())
        if found:
            df_ready[api_name] = df[found]
        else:
            # Fallbacks seguros
            if api_name == 'PEDRA':
                df_ready[api_name] = "N/A"
            else:
                print(f"⚠️ Aviso: Coluna '{excel_name}' não encontrada.")

    # 2. Mapeia Indicadores (Lag Features)
    # Procura por colunas como INDE_2023, INDE, etc.
    indicators = [
        ('INDE', 'INDE_ANTERIOR'),
        ('IAA', 'IAA_ANTERIOR'),
        ('IEG', 'IEG_ANTERIOR'),
        ('IPS', 'IPS_ANTERIOR'),
        ('IDA', 'IDA_ANTERIOR'),
        ('IPP', 'IPP_ANTERIOR'),
        ('IPV', 'IPV_ANTERIOR'),
        ('IAN', 'IAN_ANTERIOR')
    ]

    for kw, api_col in indicators:
        found = find_column(df, kw)
        if found:
            # Converte para numérico forçado
            df_ready[api_col] = pd.to_numeric(df[found], errors='coerce').fillna(0.0)
        else:
            # Se não tiver no Excel, preenche com 0.0 (simula aluno novo)
            df_ready[api_col] = 0.0

    # 3. Tratamentos Finais
    df_ready['ALUNO_NOVO'] = (df_ready['INDE_ANTERIOR'] == 0).astype(int)

    if 'IDADE' in df_ready.columns:
        df_ready['IDADE'] = pd.to_numeric(df_ready['IDADE'], errors='coerce').fillna(10).astype(int)

    if 'ANO_INGRESSO' in df_ready.columns:
        df_ready['ANO_INGRESSO'] = pd.to_numeric(df_ready['ANO_INGRESSO'], errors='coerce').fillna(2023).astype(int)

    # Preenche vazios obrigatórios com padrão para não quebrar a API
    defaults = {
        'GENERO': 'Outro',
        'TURMA': 'N/A',
        'INSTITUICAO_ENSINO': 'N/A',
        'FASE': '0',
        'PEDRA': 'Quartzo'
    }
    for col, val in defaults.items():
        if col not in df_ready.columns:
            df_ready[col] = val
        else:
            df_ready[col] = df_ready[col].fillna(val).astype(str)

    return df_ready


def simulate_traffic(df, n_requests=100):
    print(f"\n🚀 Iniciando simulação com {n_requests} requisições (Data Schema V2)...")

    records = df.to_dict(orient='records')
    random.shuffle(records)
    sample = records[:n_requests]
    sucessos = 0

    for i, student in enumerate(sample):
        try:
            response = requests.post(API_URL, json=student, timeout=5)

            if response.status_code == 200:
                data = response.json()
                risk = data.get("risk_label")
                prob = data.get("risk_probability", 0)
                print(f"[{i + 1}/{n_requests}] ✅ {risk} ({prob:.1%}) | INDE_ANT: {student.get('INDE_ANTERIOR', 0):.2f}")
                sucessos += 1
            else:
                print(f"[{i + 1}/{n_requests}] ❌ Erro {response.status_code}: {response.text}")

            time.sleep(0.05)

        except Exception as e:
            print(f"⚠️ Erro de conexão: {e}")

    print(f"\n🏁 Simulação finalizada. Sucessos: {sucessos}/{n_requests}")


if __name__ == "__main__":
    try:
        df_raw = load_real_data()
        df_ready = map_columns(df_raw)

        # Validação simples
        print(f"📋 Colunas prontas para envio: {list(df_ready.columns)}")

        simulate_traffic(df_ready, n_requests=50)

    except Exception as e:
        print(f"❌ Erro crítico: {e}")