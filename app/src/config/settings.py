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
    TARGET_COL = "RISCO_DEFASAGEM"  # 1 = Risco, 0 = Ok
    RANDOM_STATE = 42

    # --- FEATURES (WHITELIST) ---
    # Nenhuma nota aqui, garantindo que não é "soma de médias"
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