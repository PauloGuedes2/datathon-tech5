"""
Gerenciador singleton para o modelo de ML.

Responsabilidades:
- Carregar o modelo do disco
- Expor o modelo carregado
- Garantir thread-safety
"""

import os
from joblib import load
from threading import Lock
from typing import Any, Optional

from src.config.settings import Configuracoes
from src.util.logger import logger


class GerenciadorModelo:
    """
    Singleton thread-safe para gerenciamento do modelo.

    Responsabilidades:
    - Controlar a instância única
    - Manter o modelo em memória
    - Evitar recargas desnecessárias
    """

    _instancia = None
    _lock = Lock()
    _modelo: Optional[Any] = None

    def __new__(cls):
        """
        Cria ou reutiliza a instância única.

        Retorno:
        - GerenciadorModelo: instância singleton
        """
        if cls._instancia is None:
            with cls._lock:
                if cls._instancia is None:
                    cls._instancia = super(GerenciadorModelo, cls).__new__(cls)
        return cls._instancia

    def carregar_modelo(self) -> None:
        """
        Carrega o modelo do disco para a memória.

        Retorno:
        - None: não retorna valor
        """
        if self._modelo is not None:
            logger.info("Modelo já carregado em memória. Reutilizando.")
            return

        if not os.path.exists(Configuracoes.MODEL_PATH):
            logger.critical(f"Arquivo de modelo não encontrado em: {Configuracoes.MODEL_PATH}")
            return

        try:
            logger.info(f"Carregando modelo do disco: {Configuracoes.MODEL_PATH}...")
            self._modelo = load(Configuracoes.MODEL_PATH)
            logger.info("Modelo carregado com sucesso!")
        except Exception as erro:
            logger.critical(f"Falha fatal ao carregar o modelo: {erro}")
            raise erro

    def obter_modelo(self) -> Any:
        """
        Retorna o modelo carregado.

        Retorno:
        - Any: modelo em memória

        Exceções:
        - RuntimeError: quando o modelo não está disponível
        """
        if self._modelo is None:
            self.carregar_modelo()

        if self._modelo is None:
            raise RuntimeError("Modelo indisponível para inferência.")

        return self._modelo


# Aliases para compatibilidade com nomes anteriores
ModelManager = GerenciadorModelo
