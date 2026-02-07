import os
from joblib import load
from threading import Lock
from typing import Any, Optional
from src.config.settings import Settings
from src.util.logger import logger


class ModelManager:
    """
    Singleton thread-safe para gerenciamento do modelo de ML.
    """
    _instance = None
    _lock = Lock()
    _model: Optional[Any] = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def load_model(self) -> None:
        """Carrega o modelo do disco para a memória."""
        if self._model is not None:
            logger.info("Modelo já carregado em memória. Reutilizando.")
            return

        if not os.path.exists(Settings.MODEL_PATH):
            logger.critical(f"Arquivo de modelo não encontrado em: {Settings.MODEL_PATH}")
            return

        try:
            logger.info(f"Carregando modelo do disco: {Settings.MODEL_PATH}...")
            self._model = load(Settings.MODEL_PATH)
            logger.info("Modelo carregado com sucesso!")
        except Exception as e:
            logger.critical(f"Falha fatal ao carregar o modelo: {e}")
            raise e

    def get_model(self) -> Any:
        """Retorna a instância do modelo carregado."""
        if self._model is None:
            self.load_model()

        if self._model is None:
            raise RuntimeError("Modelo indisponível para inferência.")

        return self._model