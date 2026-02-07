"""
Simulador de tráfego de produção para a API de predição.

Responsabilidades:
- Carregar dados reais do diretório de dados
- Sanitizar e normalizar campos para o payload
- Enviar requisições contínuas para a API
"""

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
warnings.simplefilter(action="ignore", category=FutureWarning)

# --- 1. Configuração de Path ---
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROJETO = os.path.dirname(DIRETORIO_ATUAL)
sys.path.append(RAIZ_PROJETO)

# --- 2. Configurações da API ---
PORTA = int(os.getenv("PORT", 8000))
URL_API = f"http://localhost:{PORTA}/api/v1/predict/smart"
DELAY = 0.05  # Acelerado para teste


def limpar_genero(valor):
    """
    Converte Menino/Menina/Garota para o padrão da API.

    Parâmetros:
    - valor (Any): valor original do gênero

    Retorno:
    - str: gênero normalizado
    """
    if pd.isna(valor):
        return "Outro"
    texto = str(valor).lower().strip()

    if any(item in texto for item in ["fem", "menina", "mulher", "garota"]):
        return "Feminino"
    if any(item in texto for item in ["masc", "menino", "homem", "garoto"]):
        return "Masculino"
    return "Outro"


def limpar_fase(valor):
    """
    Remove espaços e caracteres especiais da FASE (Ex: 'FASE 5' -> 'FASE5').

    Parâmetros:
    - valor (Any): valor original da fase

    Retorno:
    - str: fase sanitizada
    """
    if pd.isna(valor):
        return "0"
    limpo = re.sub(r"[^A-Z0-9]", "", str(valor).upper())
    return limpo if limpo else "0"


def obter_coluna(row, nomes_possiveis):
    """
    Retorna o primeiro valor encontrado em possíveis nomes de coluna.

    Parâmetros:
    - row (pd.Series): linha do DataFrame
    - nomes_possiveis (list[str]): nomes de colunas possíveis

    Retorno:
    - Any: valor encontrado ou None
    """
    for nome in nomes_possiveis:
        nome_upper = nome.upper().strip()
        if nome_upper in row and pd.notnull(row[nome_upper]):
            return row[nome_upper]
    return None


def carregar_dados_reais():
    """
    Carrega dados reais do diretório de dados.

    Retorno:
    - pd.DataFrame | None: dados consolidados ou None
    """
    diretorio_dados = Configuracoes.DATA_DIR
    print(f"📂 Buscando arquivos em: {diretorio_dados}")

    extensoes = ["*.xlsx", "*.csv"]
    arquivos = []
    for extensao in extensoes:
        arquivos.extend(glob.glob(os.path.join(diretorio_dados, extensao)))

    if not arquivos:
        print(f"❌ Nenhum arquivo encontrado em {diretorio_dados}")
        return None

    dataframes = []
    for arquivo in arquivos:
        try:
            nome_arquivo = os.path.basename(arquivo)
            if arquivo.endswith(".xlsx"):
                excel = pd.ExcelFile(arquivo)
                for nome_aba in excel.sheet_names:
                    df = pd.read_excel(arquivo, sheet_name=nome_aba)
                    df["_ORIGEM"] = f"{nome_arquivo} ({nome_aba})"
                    dataframes.append(df)
            else:
                try:
                    df = pd.read_csv(arquivo, sep=";")
                    if len(df.columns) <= 1:
                        df = pd.read_csv(arquivo, sep=",")
                except Exception:
                    df = pd.read_csv(arquivo, sep=",")
                df["_ORIGEM"] = nome_arquivo
                dataframes.append(df)
        except Exception as erro:
            print(f"⚠️ Ignorando {arquivo}: {erro}")

    if not dataframes:
        return None
    return pd.concat(dataframes, ignore_index=True)


def normalizar_colunas(df):
    """
    Normaliza nomes de colunas e ajustes de RA.

    Parâmetros:
    - df (pd.DataFrame): dados originais

    Retorno:
    - pd.DataFrame: dados normalizados
    """
    df.columns = [str(c).upper().strip() for c in df.columns]
    mapa_renomear = {
        "ID_ALUNO": "RA",
        "CODIGO_ALUNO": "RA",
        "MATRICULA": "RA",
        "ALUNO": "NOME",
        "NOME_ALUNO": "NOME",
    }
    df = df.rename(columns=mapa_renomear)
    if "RA" in df.columns:
        df["RA"] = df["RA"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    return df


def obter_stream_infinito(df):
    """
    Gera um stream infinito de linhas embaralhadas.

    Parâmetros:
    - df (pd.DataFrame): dados de origem

    Retorno:
    - generator: stream infinito de linhas
    """
    while True:
        df_embaralhado = df.sample(frac=1).reset_index(drop=True)
        for _, row in df_embaralhado.iterrows():
            yield row


def _normalizar_idade(idade_raw):
    """
    Normaliza a idade a partir de valores brutos.

    Parâmetros:
    - idade_raw (Any): valor original

    Retorno:
    - int: idade normalizada
    """
    idade_final = 10
    if idade_raw:
        try:
            valor = float(idade_raw)
            if valor > 1900:
                valor = 2024 - valor
            if 4 <= valor <= 25:
                idade_final = int(valor)
        except Exception:
            pass
    return idade_final


def _normalizar_ano_ingresso(ano_raw):
    """
    Normaliza o ano de ingresso.

    Parâmetros:
    - ano_raw (Any): valor original

    Retorno:
    - int: ano de ingresso normalizado
    """
    ano_final = 2022
    if ano_raw:
        try:
            valor = int(float(ano_raw))
            if 2000 <= valor <= 2026:
                ano_final = valor
        except Exception:
            pass
    return ano_final


def _montar_payload(row, chaves):
    """
    Monta o payload de requisição a partir da linha.

    Parâmetros:
    - row (pd.Series): linha de dados
    - chaves (dict): dicionário de chaves por campo

    Retorno:
    - dict: payload pronto para envio
    """
    idade_raw = obter_coluna(row, chaves["idade"])
    ano_raw = obter_coluna(row, chaves["ano_ingresso"])
    genero_raw = obter_coluna(row, chaves["genero"])
    fase_raw = obter_coluna(row, chaves["fase"])

    idade_final = _normalizar_idade(idade_raw)
    ano_final = _normalizar_ano_ingresso(ano_raw)
    genero_final = limpar_genero(genero_raw)
    fase_final = limpar_fase(fase_raw)

    payload = {
        "RA": str(row["RA"]),
        "NOME": str(row.get("NOME", f"Aluno {row['RA']}")),
        "IDADE": idade_final,
        "ANO_INGRESSO": ano_final,
        "GENERO": genero_final,
        "TURMA": str(obter_coluna(row, chaves["turma"]) or "N/A"),
        "INSTITUICAO_ENSINO": str(obter_coluna(row, chaves["instituicao"]) or "N/A"),
        "FASE": fase_final,
    }

    for chave, valor in payload.items():
        if str(valor).lower() in ["nan", "nat", "none"]:
            payload[chave] = "N/A"

    return payload


def _enviar_payload(payload):
    """
    Envia o payload para a API.

    Parâmetros:
    - payload (dict): dados da requisição

    Retorno:
    - requests.Response: resposta da API
    """
    inicio = time.time()
    resposta = requests.post(URL_API, json=payload)
    _ = time.time() - inicio
    return resposta


def simular_trafego_producao():
    """
    Inicia a simulação de tráfego de produção.

    Retorno:
    - None: não retorna valor
    """
    print("--- 🚀 Iniciando Simulação BLINDADA (Sanitização Ativa) ---")

    dados_brutos = carregar_dados_reais()
    if dados_brutos is None or dados_brutos.empty:
        return

    dados = normalizar_colunas(dados_brutos)

    if "RA" not in dados.columns:
        print("❌ Erro: Coluna RA não encontrada.")
        return

    print(f"✅ Dados Carregados: {len(dados)} linhas.")

    chaves = {
        "idade": ["IDADE", "IDADE 2024", "IDADE_ALUNO", "ANO_NASC"],
        "ano_ingresso": ["ANO_INGRESSO", "ANO INGRESSO"],
        "genero": ["GENERO", "GÊNERO", "SEXO"],
        "turma": ["TURMA", "TURMA 2024"],
        "instituicao": ["INSTITUICAO_ENSINO", "ESCOLA", "INSTITUICAO"],
        "fase": ["FASE", "FASE 2024", "FASE_TURMA"],
    }

    stream = obter_stream_infinito(dados)
    contador = 0

    for row in stream:
        contador += 1
        try:
            payload = _montar_payload(row, chaves)
            resposta = _enviar_payload(payload)

            origem = str(row.get("_ORIGEM", "BD"))[:15]

            if resposta.status_code == 200:
                dados_resposta = resposta.json()
                print(
                    f"#{contador} | ✅ {origem} | {payload['RA']} | {payload['GENERO']} | {payload['FASE']} | "
                    f"{dados_resposta.get('risk_label')}"
                )
            else:
                print(f"#{contador} | ❌ {resposta.status_code} | {resposta.text}")

        except requests.exceptions.ConnectionError:
            print("⚠️ API Offline...")
            time.sleep(2)
        except Exception as erro:
            print(f"⚠️ Erro script: {erro}")

        time.sleep(DELAY)


if __name__ == "__main__":
    try:
        simular_trafego_producao()
    except KeyboardInterrupt:
        print("\n🛑 Encerrado.")