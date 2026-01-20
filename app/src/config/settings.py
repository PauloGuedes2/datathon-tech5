import os
from pathlib import Path


class Settings:
    # --- Caminhos ---
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_PATH = os.path.join(BASE_DIR, "data", "PEDE_PASSOS_DATASET_FIAP.xlsx")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    MODEL_PATH = os.path.join(MODEL_DIR, "model_passos_magicos.joblib")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- Regras de Negócio ---
    RISK_THRESHOLD = 0.5  # Acima disso é ALTO RISCO

    # --- ML Configs ---
    TARGET_COL = "RISCO_DEFASAGEM"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    FEATURES_NUMERICAS = [
        "IDADE_22",
        "CG",
        "CF",
        "CT",
        "IAA",
        "IEG",
        "IPS",
        "IDA",
        "MATEM",
        "PORTUG",
        "INGLES"
    ]

    FEATURES_CATEGORICAS = [
        "GENERO",
        "TURMA",
        "INSTITUICAO_DE_ENSINO",
    ]


