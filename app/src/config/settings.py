"""
Configurações centrais do projeto.

Responsabilidades:
- Definir caminhos de arquivos
- Definir hiperparâmetros do modelo
- Definir colunas de features
"""

import os
from pathlib import Path


class Configuracoes:
    """
    Centraliza configurações da aplicação.

    Responsabilidades:
    - Fornecer caminhos de diretórios
    - Declarar constantes de treinamento
    - Listar colunas permitidas
    """

    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    MONITORING_DIR = os.path.join(BASE_DIR, "monitoring")

    MODEL_PATH = os.path.join(MODEL_DIR, "model_passos_magicos.joblib")
    LOG_PATH = os.path.join(LOG_DIR, "predictions.jsonl")
    REFERENCE_PATH = os.path.join(MONITORING_DIR, "reference_data.csv")
    METRICS_FILE = os.path.join(MONITORING_DIR, "train_metrics.json")

    RISK_THRESHOLD = 0.5
    TARGET_COL = "RISCO_DEFASAGEM"
    RANDOM_STATE = 42

    FEATURES_NUMERICAS = [
        "IDADE",
        "TEMPO_NA_ONG",
        "INDE_ANTERIOR",
        "IAA_ANTERIOR",
        "IEG_ANTERIOR",
        "IPS_ANTERIOR",
        "IDA_ANTERIOR",
        "IPP_ANTERIOR",
        "IPV_ANTERIOR",
        "IAN_ANTERIOR",
        "ALUNO_NOVO",
    ]

    FEATURES_CATEGORICAS = [
        "GENERO",
        "TURMA",
        "INSTITUICAO_ENSINO",
        "FASE",
    ]

    COLUNAS_PROIBIDAS_NO_TREINO = [
        "INDE",
        "PEDRA",
        "DEFASAGEM",
        "NOTA_PORT",
        "NOTA_MAT",
        "NOTA_ING",
    ]

# Aliases para compatibilidade com nomes anteriores
Settings = Configuracoes
