import random
import time

import requests

# URL da API (ajuste a porta se necessário)
API_URL = "http://localhost:8000/api/v1/predict"

# Opções para gerar dados realistas baseados no seu domínio
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
    "FASE": ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
}


def gerar_aluno_aleatorio():
    """Gera um payload JSON compatível com o schema Student."""
    return {
        "IDADE": random.randint(7, 24),  # Idade varia de crianças a universitários
        "ANO_INGRESSO": random.randint(2018, 2025),
        "GENERO": random.choice(OPCOES["GENERO"]),
        "TURMA": random.choice(OPCOES["TURMA"]),
        "INSTITUICAO_ENSINO": random.choice(OPCOES["INSTITUICAO_ENSINO"]),
        "FASE": random.choice(OPCOES["FASE"])
    }


def testar_api(qtd_requisicoes=50, delay=0.5):
    print(f"🚀 Iniciando teste de carga em: {API_URL}")
    print(f"📦 Enviando {qtd_requisicoes} requisições...\n")

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

                # Formatação visual do output
                icon = "🔴" if risco == "ALTO RISCO" else "🟢"
                print(f"[{i:02d}/{qtd_requisicoes}] {icon} {risco} (Prob: {probabilidade:.1%}) | {duration:.2f}s")
                sucessos += 1
            else:
                print(f"[{i:02d}/{qtd_requisicoes}] ❌ Erro {response.status_code}: {response.text}")
                erros += 1

        except requests.exceptions.ConnectionError:
            print(f"[{i:02d}/{qtd_requisicoes}] ☠️  Erro de Conexão. A API está rodando?")
            erros += 1
            break
        except Exception as e:
            print(f"[{i:02d}/{qtd_requisicoes}] ⚠️  Exceção: {e}")
            erros += 1

        # Pequeno delay para simular tráfego real e não sobrecarregar instantaneamente
        time.sleep(delay)

    print("\n" + "=" * 40)
    print(f"🏁 Teste finalizado!")
    print(f"✅ Sucessos: {sucessos}")
    print(f"❌ Falhas: {erros}")
    print("=" * 40)


if __name__ == "__main__":
    # Certifique-se de instalar a biblioteca requests: pip install requests
    testar_api(qtd_requisicoes=20, delay=0.2)
