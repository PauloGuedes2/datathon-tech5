import json
from pathlib import Path

import pandas as pd
import requests

# Configurações
API_URL = "http://localhost:8000/api/v1/predict/smart"
BASE_DIR = Path(__file__).resolve().parent.parent
REFERENCE_DATA_PATH = BASE_DIR / "app"/ "monitoring" / "reference_data.csv"


def com_aluno_real():
    print("\n🚀 INICIANDO TESTE FUNCIONAL COM DADOS REAIS")

    # 1. Carrega um aluno aleatório da base histórica
    try:
        df = pd.read_csv(REFERENCE_DATA_PATH)
        # Filtra quem tem INDE válido (não nulo) para garantir que o teste seja rico
        df = df.dropna(subset=['INDE'])

        if df.empty:
            print("❌ Erro: O arquivo de referência está vazio.")
            return

        # Sorteia um aluno
        aluno_real = df.sample(1).iloc[0]
        nome_aluno = aluno_real['NOME']
        inde_esperado = aluno_real['INDE']

        print(f"👤 Aluno Sorteado: {nome_aluno}")
        print(f"📊 INDE no Histórico (2023): {inde_esperado}")

    except FileNotFoundError:
        print("❌ Erro: Execute o 'setup_dados_reais.py' primeiro!")
        return

    # 2. Monta o Payload (Dados 'Atuais' de 2024 simulados)
    # Note que NÃO enviamos o INDE_ANTERIOR. A API deve buscar sozinha.
    payload = {
        "NOME": nome_aluno,
        "IDADE": 12,  # Simulação
        "ANO_INGRESSO": 2020,  # Simulação
        "GENERO": "Feminino",  # Simulação (Genérico para passar na validação)
        "TURMA": "5A",
        "INSTITUICAO_ENSINO": "ESCOLA TESTE",
        "FASE": "2"
    }

    print("\n📡 Enviando requisição para API...")

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            print("\n✅ SUCESSO! A API retornou:")
            print(json.dumps(result, indent=2))

            print(f"\n📝 Análise:")
            print(f"   O modelo calculou: {result['risk_label']}")
            print(f"   Com probabilidade: {result['risk_probability']:.2%}")
            print("   (Verifique no log da API se apareceu 'Histórico encontrado')")
        else:
            print(f"\n❌ Erro na API ({response.status_code}):")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("\n❌ Erro: A API não está rodando. Execute 'python main.py' em outro terminal.")


if __name__ == "__main__":
    com_aluno_real()
