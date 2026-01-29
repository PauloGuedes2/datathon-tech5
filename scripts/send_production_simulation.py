import pandas as pd
import requests
import time
import os
import glob
import sys
import numpy as np

# --- 1. Configuração de Path (Para rodar de qualquer lugar) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from src.config.settings import Settings
except ImportError:
    sys.path.append(os.getcwd())
    from src.config.settings import Settings

# --- 2. Configurações da API ---
PORT = int(os.getenv("PORT", 8000))
API_URL = f"http://localhost:{PORT}/api/v1/predict/smart"
DELAY = 0.1


def load_real_data():
    """
    Carrega arquivos CSV ou XLSX direto da pasta definida no Settings.DATA_DIR
    """
    data_dir = Settings.DATA_DIR
    print(f"📂 Buscando arquivos de dados em: {data_dir}")

    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    xlsx_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    all_files = csv_files + xlsx_files

    if not all_files:
        print(f"❌ Nenhum arquivo de dados encontrado em {data_dir}")
        return None

    print(f"✅ Encontrados {len(all_files)} arquivos: {[os.path.basename(f) for f in all_files]}")

    dataframes = []
    for file in all_files:
        try:
            filename = os.path.basename(file)
            if file.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                try:
                    df = pd.read_csv(file, sep=';')
                    if len(df.columns) <= 1:
                        df = pd.read_csv(file, sep=',')
                except:
                    df = pd.read_csv(file, sep=',')

            df['_ORIGEM'] = filename
            dataframes.append(df)
        except Exception as e:
            print(f"⚠️ Erro ao ler {file}: {e}")

    if not dataframes:
        return None

    try:
        full_df = pd.concat(dataframes, ignore_index=True)
        return full_df
    except ValueError:
        return None


def normalize_columns(df):
    """
    Padroniza colunas para o formato da API
    """
    df.columns = [str(c).upper().strip() for c in df.columns]

    rename_map = {
        'ID_ALUNO': 'RA', 'CODIGO_ALUNO': 'RA', 'MATRICULA': 'RA',
        'ALUNO': 'NOME', 'NOME_ALUNO': 'NOME',
        'ANO': 'ANO_INGRESSO',
        'INSTITUICAO': 'INSTITUICAO_ENSINO', 'ESCOLA': 'INSTITUICAO_ENSINO',
    }
    df = df.rename(columns=rename_map)

    if 'RA' in df.columns:
        df['RA'] = df['RA'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

    return df


def get_infinite_stream(df):
    while True:
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
        for index, row in df_shuffled.iterrows():
            yield row


def simulate_production_traffic():
    print("--- 🚀 Iniciando Simulação com DADOS REAIS (Correção IDADE) ---")

    raw_df = load_real_data()
    if raw_df is None or raw_df.empty:
        print("❌ Abortando: Dataframe vazio.")
        return

    df = normalize_columns(raw_df)

    if 'RA' not in df.columns:
        print("❌ Erro: Coluna 'RA' não encontrada.")
        print(f"Colunas disponíveis: {df.columns.tolist()}")
        return

    print(f"✅ Carregado: {len(df)} linhas.")
    print(f"📡 Target: {API_URL}")

    stream = get_infinite_stream(df)
    counter = 0

    for row in stream:
        counter += 1
        try:
            # --- LÓGICA DE SANITIZAÇÃO CORRIGIDA ---

            # 1. Tratamento de IDADE (API exige >= 4)
            # Se for nulo ou menor que 4, usa 10 como padrão
            idade_raw = row.get('IDADE')
            idade_final = 10  # Default seguro

            if pd.notnull(idade_raw):
                try:
                    val = int(idade_raw)
                    if val >= 4:
                        idade_final = val
                except:
                    pass

            # 2. Tratamento de ANO_INGRESSO (API exige >= 2010)
            ano_raw = row.get('ANO_INGRESSO')
            ano_final = 2022  # Default seguro

            if pd.notnull(ano_raw):
                try:
                    val = int(ano_raw)
                    if 2010 <= val <= 2026:
                        ano_final = val
                except:
                    pass

            payload = {
                "RA": str(row['RA']),
                "NOME": str(row.get('NOME', f"Aluno {row['RA']}")),
                "IDADE": idade_final,
                "ANO_INGRESSO": ano_final,
                "GENERO": str(row.get('GENERO', 'Outro')),
                "TURMA": str(row.get('TURMA', 'N/A')),
                "INSTITUICAO_ENSINO": str(row.get('INSTITUICAO_ENSINO', 'N/A')),
                "FASE": str(row.get('FASE', '0'))
            }

            # Limpeza final para nulos nas strings
            for k, v in payload.items():
                if str(v).lower() == 'nan':
                    payload[k] = "N/A"

            start_time = time.time()
            response = requests.post(API_URL, json=payload)
            elapsed = time.time() - start_time

            origin = row.get('_ORIGEM', 'BD')

            if response.status_code == 200:
                d = response.json()
                print(
                    f"#{counter} | 📂 {origin} | RA: {payload['RA']} | Idade: {payload['IDADE']} | ⏱ {elapsed:.2f}s | {d.get('risk_label')} ({d.get('risk_probability')})")
            else:
                print(f"#{counter} | ❌ Erro {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            print(f"⚠️ API Offline em {API_URL}. Reconectando...")
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ Erro: {e}")

        time.sleep(DELAY)


if __name__ == "__main__":
    try:
        simulate_production_traffic()
    except KeyboardInterrupt:
        print("\n🛑 Encerrado.")