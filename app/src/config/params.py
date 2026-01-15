import os
from pathlib import Path


class Params:
    # --- Caminhos ---
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = os.path.join(BASE_DIR, "data", "PEDE_PASSOS_DATASET_FIAP.xlsx")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    MODEL_PATH = os.path.join(MODEL_DIR, "model_passos_magicos.joblib")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- Configurações do Modelo ---
    TARGET_COL = "RISCO_DEFASAGEM"  # Coluna alvo que criaremos
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    # Features numéricas baseadas no Dicionário de Dados (INDE e seus componentes)
    FEATURES_NUMERICAS = [
        "IAN", "IDA", "IEG", "IAA",
        "IPS", "IPV", "INDE_22"
    ]

    # Features categóricas/texto
    FEATURES_CATEGORICAS = [
        "PEDRA_22", "ATINGIU_PV"
    ]

    """Configurações globais"""

    # --- Configurações de Logging ---
    LOG_LEVEL: str = "INFO"  # Nível de log (DEBUG, INFO, WARNING, ERROR)
    LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'