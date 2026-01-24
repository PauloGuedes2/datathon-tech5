import os
from pathlib import Path


class Settings:
    # --- Caminhos ---
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    MODEL_PATH = os.path.join(MODEL_DIR, "model_passos_magicos.joblib")
    LOG_PATH = os.path.join(BASE_DIR, "logs", "predictions.csv")

    # Monitoramento
    MONITORING_PATH = os.path.join(BASE_DIR, "monitoring")
    REFERENCE_PATH = os.path.join(MONITORING_PATH, "reference_data.csv")

    # --- Configuração do Modelo ---
    RISK_THRESHOLD = 0.5
    TARGET_COL = "RISCO_DEFASAGEM"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    # --- FEATURES EXPLICITAS (WHITELIST) ---
    FEATURES_NUMERICAS = [
        "IDADE",
        "TEMPO_NA_ONG"
    ]

    FEATURES_CATEGORICAS = [
        "GENERO",
        "TURMA",
        "INSTITUICAO_ENSINO",
        "FASE"
    ]