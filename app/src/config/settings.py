import os
from pathlib import Path


class Settings:
    # --- Caminhos Base ---
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    MONITORING_DIR = os.path.join(BASE_DIR, "monitoring")

    # --- Arquivos Finais ---
    MODEL_PATH = os.path.join(MODEL_DIR, "model_passos_magicos.joblib")
    LOG_PATH = os.path.join(LOG_DIR, "predictions.jsonl")
    REFERENCE_PATH = os.path.join(MONITORING_DIR, "reference_data.csv")
    METRICS_FILE = os.path.join(MONITORING_DIR, "train_metrics.json")

    # --- Configuração do Modelo ---
    RISK_THRESHOLD = 0.5
    TARGET_COL = "RISCO_DEFASAGEM"
    RANDOM_STATE = 42

    # --- FEATURES QUE O MODELO PODE VER ---
    FEATURES_NUMERICAS = [
        "IDADE",
        "TEMPO_NA_ONG",
        # Features Históricas (Passado)
        "INDE_ANTERIOR",
        "IAA_ANTERIOR",
        "IEG_ANTERIOR",
        "IPS_ANTERIOR",
        "IDA_ANTERIOR",
        "IPP_ANTERIOR",
        "IPV_ANTERIOR",
        "IAN_ANTERIOR",
        # Flag de controle
        "ALUNO_NOVO"
    ]

    FEATURES_CATEGORICAS = [
        "GENERO",
        "TURMA",
        "INSTITUICAO_ENSINO",
        "FASE"
    ]

    # Lista de colunas usadas APENAS para criar o target, mas proibidas no X (input)
    # Isso previne o "Somador de Notas"
    COLUNAS_PROIBIDAS_NO_TREINO = [
        "INDE", "PEDRA", "DEFASAGEM",
        "NOTA_PORT", "NOTA_MAT", "NOTA_ING"
    ]