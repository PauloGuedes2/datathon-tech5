import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta
from src.application.risk_service import RiskService
from src.config.settings import Settings


def carregar_referencia():
    """Carrega os dados usados no treino para imitar o padrão real."""
    if not os.path.exists(Settings.REFERENCE_PATH):
        raise FileNotFoundError(f"Arquivo de referência não encontrado: {Settings.REFERENCE_PATH}")
    return pd.read_csv(Settings.REFERENCE_PATH)


def gerar_dados_normais(n=500):
    """
    CENÁRIO 1: SEM DRIFT (Normal)
    Pega dados reais do treino e adiciona um leve 'ruído' para não serem idênticos.
    Isso simula um dia normal de operação.
    """
    print(f"🟢 Gerando {n} registros NORMAIS (Baseados no Histórico)...")
    ref_df = carregar_referencia()

    # Amostra aleatória do histórico (respeita a distribuição original)
    samples = ref_df.sample(n=n, replace=True).copy()

    # Adiciona leve ruído/variação para parecer dados novos
    # Ex: Varia a idade em +/- 1 ano as vezes
    noise = np.random.choice([-1, 0, 1], size=n, p=[0.1, 0.8, 0.1])
    samples["IDADE"] = samples["IDADE"] + noise

    # Garante que não ficou negativo
    samples["IDADE"] = samples["IDADE"].clip(lower=6, upper=25)

    return samples


def gerar_dados_com_drift(n=500):
    """
    CENÁRIO 2: COM DRIFT (Anomalia)
    Simula uma mudança brusca no perfil.
    Ex: A ONG começou a atender um público muito mais velho ou de outra região.
    """
    print(f"🔴 Gerando {n} registros com DRIFT (Mudança de Perfil)...")

    data = []
    # Gera dados sintéticos que sabemos que são diferentes do treino
    for _ in range(n):
        row = {
            # Drift de Idade: Média muito mais alta (20 anos)
            "IDADE": int(np.random.normal(20, 2)),
            "GENERO": random.choice(["MASCULINO", "FEMININO"]),
            # Drift de Instituição: Inverte a lógica (80% Particular)
            "INSTITUICAO_ENSINO": random.choice(["ESCOLA PARTICULAR"] * 80 + ["ESCOLA PUBLICA"] * 20),
            "TURMA": "TURMA EXTENA",
            "FASE": "8",  # Fase que nem existe no treino
            "ANO_INGRESSO": datetime.now().year - 1
        }
        data.append(row)

    return pd.DataFrame(data)


def processar_e_salvar(df_input):
    """Passa os dados pelo modelo e salva no CSV de logs."""
    service = RiskService()

    # 1. Prepara Features (calcula TEMPO_NA_ONG se necessário)
    # Importante: O _prepare_features espera as colunas originais.
    # Se viemos do Reference Data, já temos TEMPO_NA_ONG calculado.
    # Vamos recalcular para garantir consistência.

    if "ANO_INGRESSO" not in df_input.columns and "TEMPO_NA_ONG" in df_input.columns:
        # Se veio da referência, simulamos o ano de ingresso reverso
        df_input["ANO_INGRESSO"] = datetime.now().year - df_input["TEMPO_NA_ONG"]

    df_input["ANO_REFERENCIA"] = datetime.now().year

    # Usa o serviço para limpar e preparar
    df_processed = service._prepare_features(df_input)

    # 2. Recalcula Predição (O modelo julga os dados novos)
    probs = service.model.predict_proba(df_processed)[:, 1]

    # 3. Monta Log
    log_df = df_processed.copy()
    log_df["prediction"] = probs

    # Timestamps recentes
    base_time = datetime.now()
    timestamps = [base_time - timedelta(seconds=random.randint(0, 3600)) for _ in range(len(df_input))]
    timestamps.sort()
    log_df["timestamp"] = timestamps

    # 4. Salva
    file_path = Settings.LOG_PATH
    header = not os.path.exists(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    log_df.to_csv(file_path, mode='a', header=header, index=False)

    print(f"✅ Salvo em: {file_path}")


if __name__ == "__main__":
    print("Escolha o cenário de teste:")
    print("1 - Simular Operação NORMAL (Sem Drift)")
    print("2 - Simular Mudança de Perfil (Com Drift)")
    print("3 - Limpar logs antigos (Reset)")

    choice = input("Opção: ")

    if choice == "1":
        df = gerar_dados_normais(500)
        processar_e_salvar(df)
        print("👉 Confira o Dashboard. As colunas devem estar VERDES (pouco ou nenhum drift).")

    elif choice == "2":
        df = gerar_dados_com_drift(500)
        processar_e_salvar(df)
        print("👉 Confira o Dashboard. As colunas devem estar VERMELHAS (Drift detectado).")

    elif choice == "3":
        if os.path.exists(Settings.LOG_PATH):
            os.remove(Settings.LOG_PATH)
            print("🗑️ Logs apagados.")
        else:
            print("Nada para apagar.")