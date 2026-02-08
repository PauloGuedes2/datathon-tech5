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
    FEATURE_STATS_PATH = os.path.join(MONITORING_DIR, "feature_stats.json")
    MODEL_SHA256 = os.getenv("MODEL_SHA256")
    MODEL_SHA256_REQUIRED = os.getenv("MODEL_SHA256_REQUIRED", "false").lower() in ("1", "true", "yes")

    RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.5"))
    TARGET_COL = "RISCO_DEFASAGEM"
    RANDOM_STATE = 42
    MIN_RECALL = float(os.getenv("MIN_RECALL", "0.6"))
    N_JOBS = int(os.getenv("MODEL_N_JOBS", "1"))

    HISTORICAL_PATH = os.getenv("HISTORICAL_PATH")
    LOG_SAMPLE_LIMIT = int(os.getenv("LOG_SAMPLE_LIMIT", "1000"))
    LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))

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

    # Features sensíveis que não devem ser usadas pelo modelo.
    FEATURES_SENSIVEIS = ["GENERO"]

    FEATURES_CATEGORICAS = [
        "GENERO",
        "TURMA",
        "INSTITUICAO_ENSINO",
        "FASE",
    ]

    FEATURES_MODELO_NUMERICAS = FEATURES_NUMERICAS
    FEATURES_MODELO_CATEGORICAS = []

    FAIRNESS_GROUP_COL = "GENERO"

    COLUNAS_PROIBIDAS_NO_TREINO = [
        "INDE",
        "PEDRA",
        "DEFASAGEM",
        "NOTA_PORT",
        "NOTA_MAT",
        "NOTA_ING",
    ]


Configuracoes.FEATURES_MODELO_CATEGORICAS = [
    c for c in Configuracoes.FEATURES_CATEGORICAS if c not in Configuracoes.FEATURES_SENSIVEIS
]
