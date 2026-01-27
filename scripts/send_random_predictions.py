import random
import time
import requests

# URL da API
API_URL = "http://localhost:8000/api/v1/predict"

# Opções para gerar dados categóricos
OPCOES = {
    "GENERO": ["Masculino", "Feminino"],
    "INSTITUICAO_ENSINO": [
        "Escola Pública",
        "Privada",
        "Privada com Bolsa",
        "SESI"
    ],
    "TURMA": [
        "Alfa A", "Alfa B",
        "Fase 1A", "Fase 1B",
        "Fase 2A", "Fase 3C",
        "Fase 6", "Fase 7 (Universitários)"
    ],
    "FASE": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
    "PEDRA": ["Quartzo", "Ágata", "Ametista", "Topázio"]
}


def gerar_aluno_aleatorio():
    """
    Gera um payload JSON compatível com o schema Student atualizado.
    Inclui as Lag Features (Histórico) e Indicadores.
    """
    # Simula um aluno com histórico ou novato (10% chance de ser novato)
    is_novo = 1 if random.random() < 0.1 else 0

    student = {
        # --- Dados Demográficos ---
        "IDADE": random.randint(7, 24),
        "ANO_INGRESSO": random.randint(2018, 2025),
        "GENERO": random.choice(OPCOES["GENERO"]),
        "TURMA": random.choice(OPCOES["TURMA"]),
        "INSTITUICAO_ENSINO": random.choice(OPCOES["INSTITUICAO_ENSINO"]),
        "FASE": random.choice(OPCOES["FASE"]),
        "NOME": f"Aluno Teste {random.randint(1000, 9999)}",

        # --- Flag de Controle ---
        "ALUNO_NOVO": is_novo,
        "PEDRA": random.choice(OPCOES["PEDRA"])
    }

    # --- Indicadores Históricos (Lag Features) ---
    # Se for aluno novo, indicadores são 0. Se veterano, gera notas aleatórias.
    if is_novo:
        indicadores = {
            "INDE_ANTERIOR": 0.0,
            "IAA_ANTERIOR": 0.0,
            "IEG_ANTERIOR": 0.0,
            "IPS_ANTERIOR": 0.0,
            "IDA_ANTERIOR": 0.0,
            "IPP_ANTERIOR": 0.0,
            "IPV_ANTERIOR": 0.0,
            "IAN_ANTERIOR": 0.0
        }
    else:
        # Gera notas realistas (entre 2.0 e 9.5)
        indicadores = {
            "INDE_ANTERIOR": round(random.uniform(4.0, 9.5), 3),
            "IAA_ANTERIOR": round(random.uniform(5.0, 10.0), 3),
            "IEG_ANTERIOR": round(random.uniform(4.0, 9.5), 3),
            "IPS_ANTERIOR": round(random.uniform(5.0, 9.0), 3),
            "IDA_ANTERIOR": round(random.uniform(3.0, 9.5), 3),
            "IPP_ANTERIOR": round(random.uniform(5.0, 9.0), 3),
            "IPV_ANTERIOR": round(random.uniform(4.0, 9.0), 3),
            "IAN_ANTERIOR": round(random.uniform(5.0, 10.0), 3)
        }

    student.update(indicadores)
    return student


def testar_api(qtd_requisicoes=50, delay=0.2):
    print(f"🚀 Iniciando teste de carga atualizado em: {API_URL}")
    print(f"📦 Enviando {qtd_requisicoes} requisições com Lag Features...\n")

    sucessos = 0
    erros = 0

    for i in range(1, qtd_requisicoes + 1):
        payload = gerar_aluno_aleatorio()

        try:
            start_time = time.time()
            response = requests.post(API_URL, json=payload, timeout=5)
            duration = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                probabilidade = data.get('risk_probability', 0)
                risco = data.get('risk_label', 'N/A')

                # Formatação visual
                icon = "🔴" if risco == "ALTO RISCO" else "🟢"
                print(
                    f"[{i:02d}/{qtd_requisicoes}] {icon} {risco} (Prob: {probabilidade:.1%}) | INDE_ANT: {payload['INDE_ANTERIOR']} | {duration:.2f}s")
                sucessos += 1
            else:
                print(f"[{i:02d}/{qtd_requisicoes}] ❌ Erro {response.status_code}: {response.text}")
                erros += 1

        except requests.exceptions.ConnectionError:
            print(f"[{i:02d}/{qtd_requisicoes}] ☠️  Erro de Conexão. A API está rodando?")
            break
        except Exception as e:
            print(f"[{i:02d}/{qtd_requisicoes}] ⚠️  Exceção: {e}")
            erros += 1

        time.sleep(delay)

    print("\n" + "=" * 40)
    print(f"🏁 Teste finalizado!")
    print(f"✅ Sucessos: {sucessos}")
    print(f"❌ Falhas: {erros}")
    print("=" * 40)


if __name__ == "__main__":
    testar_api(qtd_requisicoes=20, delay=0.1)