import json
import os
import threading
import uuid
from datetime import datetime
from src.config.settings import Settings
from src.util.logger import logger

class PredictionLogger:
    """
    Logger thread-safe para persistir predições em formato JSONL.
    Atende ao Requisito 3: Concorrência Segura e JSON Estruturado.
    """
    _instance = None
    _lock = threading.Lock() # Lock global para controlar escrita no arquivo

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(PredictionLogger, cls).__new__(cls)
        return cls._instance

    def log_prediction(self, features: dict, prediction_data: dict, model_version: str = "2.1.0"):
        """
        Escreve um registro de predição de forma atômica.
        """
        # Cria o payload estruturado (Requisito 3)
        log_entry = {
            "prediction_id": str(uuid.uuid4()),
            "correlation_id": prediction_data.get("correlation_id", str(uuid.uuid4())),
            "timestamp": datetime.now().isoformat(),
            "model_version": model_version,
            "input_features": features, # Salva o dicionário de features usado
            "prediction_result": {
                "class": prediction_data.get("prediction"),
                "probability": prediction_data.get("risk_probability"),
                "label": prediction_data.get("risk_label")
            }
        }

        # Serializa para JSON
        try:
            json_line = json.dumps(log_entry, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Falha ao serializar log: {e}")
            return

        # Escrita Thread-Safe
        # O lock garante que apenas uma thread escreva no arquivo por vez
        with self._lock:
            try:
                os.makedirs(os.path.dirname(Settings.LOG_PATH), exist_ok=True)
                with open(Settings.LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json_line + "\n")
            except Exception as e:
                logger.error(f"Falha Crítica ao escrever no log de predição: {e}")