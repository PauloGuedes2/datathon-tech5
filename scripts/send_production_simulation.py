import glob
import os
import re
import sys
import time
import warnings

import pandas as pd
import requests

from src.config.settings import Configuracoes

# Suprime avisos de pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 1. Configuração de Path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# --- 2. Configurações da API ---
PORT = int(os.getenv("PORT", 8000))
API_URL = f"http://localhost:{PORT}/api/v1/predict/smart"
DELAY = 0.05  # Acelerado para teste


# --- FUNÇÕES DE LIMPEZA (SANITIZERS) ---

def clean_gender(val):
    """Converte Menino/Menina/Garota para o padrão da API"""
    if pd.isna(val): return "Outro"
    s = str(val).lower().strip()

    if any(x in s for x in ['fem', 'menina', 'mulher', 'garota']):
        return "Feminino"
    if any(x in s for x in ['masc', 'menino', 'homem', 'garoto']):
        return "Masculino"
    return "Outro"


def clean_phase(val):
    """Remove espaços e caracteres especiais da FASE (Ex: 'FASE 5' -> 'FASE5')"""
    if pd.isna(val): return "0"
    # Remove tudo que NÃO for letra ou número (remove espaços, parênteses, traços)
    cleaned = re.sub(r'[^A-Z0-9]', '', str(val).upper())
    return cleaned if cleaned else "0"


def get_any_col(row, possible_names):
    for name in possible_names:
        name_upper = name.upper().strip()
        if name_upper in row and pd.notnull(row[name_upper]):
            return row[name_upper]
    return None


# --- CARREGAMENTO DE DADOS ---

def load_real_data():
    data_dir = Configuracoes.DATA_DIR
    print(f"📂 Buscando arquivos em: {data_dir}")

    extensions = ['*.xlsx', '*.csv']
    all_files = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(data_dir, ext)))

    if not all_files:
        print(f"❌ Nenhum arquivo encontrado em {data_dir}")
        return None

    dataframes = []
    for file in all_files:
        try:
            filename = os.path.basename(file)
            if file.endswith('.xlsx'):
                xls = pd.ExcelFile(file)
                for sheet_name in xls.sheet_names:
                    # Tenta ler ignorando linhas de cabeçalho ruins se necessário
                    df = pd.read_excel(file, sheet_name=sheet_name)
                    df['_ORIGEM'] = f"{filename} ({sheet_name})"
                    dataframes.append(df)
            else:
                try:
                    df = pd.read_csv(file, sep=';')
                    if len(df.columns) <= 1: df = pd.read_csv(file, sep=',')
                except:
                    df = pd.read_csv(file, sep=',')
                df['_ORIGEM'] = filename
                dataframes.append(df)
        except Exception as e:
            print(f"⚠️ Ignorando {file}: {e}")

    if not dataframes: return None
    return pd.concat(dataframes, ignore_index=True)


def normalize_columns(df):
    df.columns = [str(c).upper().strip() for c in df.columns]
    rename_map = {
        'ID_ALUNO': 'RA', 'CODIGO_ALUNO': 'RA', 'MATRICULA': 'RA',
        'ALUNO': 'NOME', 'NOME_ALUNO': 'NOME'
    }
    df = df.rename(columns=rename_map)
    if 'RA' in df.columns:
        df['RA'] = df['RA'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    return df


def get_infinite_stream(df):
    while True:
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
        for _, row in df_shuffled.iterrows():
            yield row


# --- EXECUÇÃO PRINCIPAL ---

def simulate_production_traffic():
    print("--- 🚀 Iniciando Simulação BLINDADA (Sanitização Ativa) ---")

    raw_df = load_real_data()
    if raw_df is None or raw_df.empty: return

    df = normalize_columns(raw_df)

    if 'RA' not in df.columns:
        print("❌ Erro: Coluna RA não encontrada.")
        return

    print(f"✅ Dados Carregados: {len(df)} linhas.")

    # Listas de Sinônimos
    keys_idade = ['IDADE', 'IDADE 2024', 'IDADE_ALUNO', 'ANO_NASC']
    keys_ano_ing = ['ANO_INGRESSO', 'ANO INGRESSO']
    keys_genero = ['GENERO', 'GÊNERO', 'SEXO']
    keys_turma = ['TURMA', 'TURMA 2024']
    keys_inst = ['INSTITUICAO_ENSINO', 'ESCOLA', 'INSTITUICAO']
    keys_fase = ['FASE', 'FASE 2024', 'FASE_TURMA']

    stream = get_infinite_stream(df)
    counter = 0

    for row in stream:
        counter += 1
        try:
            # 1. Tratamento Idade
            idade_raw = get_any_col(row, keys_idade)
            idade_final = 10
            if idade_raw:
                try:
                    val = float(idade_raw)
                    if val > 1900: val = 2024 - val  # Correção se for ano nasc
                    if 4 <= val <= 25: idade_final = int(val)
                except:
                    pass

            # 2. Tratamento Ano Ingresso
            ano_raw = get_any_col(row, keys_ano_ing)
            ano_final = 2022
            if ano_raw:
                try:
                    val = int(float(ano_raw))
                    if 2000 <= val <= 2026: ano_final = val
                except:
                    pass

            # 3. Tratamento Gênero (CORREÇÃO DE "MENINA")
            genero_raw = get_any_col(row, keys_genero)
            genero_final = clean_gender(genero_raw)

            # 4. Tratamento Fase (CORREÇÃO DE "FASE 5")
            fase_raw = get_any_col(row, keys_fase)
            fase_final = clean_phase(fase_raw)

            payload = {
                "RA": str(row['RA']),
                "NOME": str(row.get('NOME', f"Aluno {row['RA']}")),
                "IDADE": idade_final,
                "ANO_INGRESSO": ano_final,
                "GENERO": genero_final,
                "TURMA": str(get_any_col(row, keys_turma) or "N/A"),
                "INSTITUICAO_ENSINO": str(get_any_col(row, keys_inst) or "N/A"),
                "FASE": fase_final
            }

            # Limpeza final N/A
            for k, v in payload.items():
                if str(v).lower() in ['nan', 'nat', 'none']: payload[k] = "N/A"

            # Envio
            start_time = time.time()
            response = requests.post(API_URL, json=payload)
            elapsed = time.time() - start_time

            origin = str(row.get('_ORIGEM', 'BD'))[:15]

            if response.status_code == 200:
                d = response.json()
                print(
                    f"#{counter} | ✅ {origin} | {payload['RA']} | {payload['GENERO']} | {payload['FASE']} | {d.get('risk_label')}")
            else:
                print(f"#{counter} | ❌ {response.status_code} | {response.text}")

        except requests.exceptions.ConnectionError:
            print("⚠️ API Offline...")
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ Erro script: {e}")

        time.sleep(DELAY)


if __name__ == "__main__":
    try:
        simulate_production_traffic()
    except KeyboardInterrupt:
        print("\n🛑 Encerrado.")
