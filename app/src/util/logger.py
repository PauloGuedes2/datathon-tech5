import logging
import sys

from src.config.settings import Settings


class LoggerFactory:
    """
    Responsável por configurar e fornecer instâncias de Logger
    padronizadas para toda a aplicação.
    
    Funcionalidades:
        - Configuração única de logger
        - Formatação padronizada
        - Handler para console/Docker
        - Prevenção de logs duplicados
    """

    _configured = False

    @classmethod
    def setup(cls, name: str = "PASSOS_MAGICOS_APP"):
        """
        Configura o logger se ainda não estiver configurado.
        
        Args:
            name: Nome do logger (padrão: PASSOS_MAGICOS_APP)
            
        Returns:
            Logger configurado e pronto para uso
            
        Features:
            - Garante que não haja duplicação de handlers
            - Formatação com timestamp e nível
            - Output para stdout (compatível com Docker)
        """
        logger = logging.getLogger(name)

        if not logger.handlers:
            log_level = getattr(Settings, "LOG_LEVEL", "INFO")
            logger.setLevel(log_level)

            formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            logger.propagate = False

        return logger

logger = LoggerFactory.setup()
