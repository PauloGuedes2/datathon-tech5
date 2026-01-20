import logging
import sys

from src.config.settings import Settings


class LoggerFactory:
    """
    Responsável por configurar e fornecer instâncias de Logger
    padronizadas para toda a aplicação.
    """

    _configured = False

    @classmethod
    def setup(cls, name: str = "PASSOS_MAGICOS_APP"):
        """
        Configura o logger se ainda não estiver configurado.
        Garante que não haja duplicação de handlers.
        """
        logger = logging.getLogger(name)

        if not logger.handlers:
            # Define o nível de log (padrão INFO se não estiver nas settings)
            log_level = getattr(Settings, "LOG_LEVEL", "INFO")
            logger.setLevel(log_level)

            # Formatação Padronizada
            formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            # Handler para Console (Stdout) - Essencial para Docker logs
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # Evita propagação para loggers raiz (evita logs duplicados do Uvicorn/FastAPI)
            logger.propagate = False

        return logger

logger = LoggerFactory.setup()
