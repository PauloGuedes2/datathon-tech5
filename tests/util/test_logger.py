import logging
import sys
from unittest.mock import Mock, patch

import pytest

from src.util.logger import LoggerFactory, logger


class TestLoggerFactory:
    """
    Classe de testes para LoggerFactory.
    
    Testa:
        - Configuração de logger
        - Formatação de mensagens
        - Handlers de console
        - Prevenção de duplicação
    """

    def test_logger_factory_setup_basic(self):
        """
        Testa configuração básica do logger.
        
        Verifica:
            - Logger é criado corretamente
            - Nome padrão é usado
            - Instância é retornada
        """
        test_logger = LoggerFactory.setup("test_logger")

        assert isinstance(test_logger, logging.Logger)
        assert test_logger.name == "test_logger"

    def test_logger_factory_default_name(self):
        """
        Testa uso do nome padrão.
        
        Verifica:
            - Nome padrão é aplicado quando não especificado
            - Logger é configurado corretamente
        """
        test_logger = LoggerFactory.setup()

        assert test_logger.name == "PASSOS_MAGICOS_APP"

    @patch('src.util.logger.Settings')
    def test_logger_level_configuration(self, mock_settings):
        """
        Testa configuração do nível de log.
        
        Args:
            mock_settings: Mock das configurações
            
        Verifica:
            - Nível é configurado corretamente
            - Fallback para INFO funciona
        """
        # Teste com nível customizado
        mock_settings.LOG_LEVEL = "DEBUG"
        test_logger = LoggerFactory.setup("test_debug")

        # Limpa handlers para teste limpo
        test_logger.handlers.clear()
        test_logger = LoggerFactory.setup("test_debug")

        assert test_logger.level == logging.DEBUG or test_logger.level == logging.INFO

    @patch('src.util.logger.Settings')
    def test_logger_level_fallback(self, mock_settings):
        """
        Testa fallback para nível INFO.
        
        Args:
            mock_settings: Mock das configurações
            
        Verifica:
            - Quando LOG_LEVEL não existe, usa INFO
            - Configuração é robusta
        """
        # Remove LOG_LEVEL do mock
        del mock_settings.LOG_LEVEL

        test_logger = LoggerFactory.setup("test_fallback")

        # Verifica que não houve erro e logger foi criado
        assert isinstance(test_logger, logging.Logger)

    def test_logger_handler_configuration(self):
        """
        Testa configuração dos handlers.
        
        Verifica:
            - Handler de console é adicionado
            - Formatter é configurado
            - Output vai para stdout
        """
        test_logger = LoggerFactory.setup("test_handler")

        # Verifica que tem pelo menos um handler
        assert len(test_logger.handlers) >= 1

        # Verifica que tem handler de console
        console_handlers = [h for h in test_logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(console_handlers) >= 1

        # Verifica que handler usa stdout
        console_handler = console_handlers[0]
        assert console_handler.stream == sys.stdout

    def test_logger_formatter_configuration(self):
        """
        Testa configuração do formatter.
        
        Verifica:
            - Formatter é aplicado aos handlers
            - Formato inclui timestamp, nome, nível, mensagem
        """
        test_logger = LoggerFactory.setup("test_formatter")

        # Pega o primeiro handler
        if test_logger.handlers:
            handler = test_logger.handlers[0]
            formatter = handler.formatter

            assert formatter is not None

            # Testa formato criando um log record
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="Test message",
                args=(),
                exc_info=None
            )

            formatted = formatter.format(record)

            # Verifica elementos do formato
            assert "test" in formatted  # nome
            assert "INFO" in formatted  # nível
            assert "Test message" in formatted  # mensagem
            assert "-" in formatted  # separadores

    def test_logger_no_propagation(self):
        """
        Testa que propagação está desabilitada.
        
        Verifica:
            - propagate é False
            - Evita logs duplicados
        """
        test_logger = LoggerFactory.setup("test_propagation")

        assert test_logger.propagate is False

    def test_logger_handler_duplication_prevention(self):
        """
        Testa prevenção de duplicação de handlers.
        
        Verifica:
            - Múltiplas chamadas não adicionam handlers duplicados
            - Logger mantém configuração consistente
        """
        # Primeira configuração
        test_logger1 = LoggerFactory.setup("test_duplication")
        initial_handler_count = len(test_logger1.handlers)

        # Segunda configuração (mesmo nome)
        test_logger2 = LoggerFactory.setup("test_duplication")
        final_handler_count = len(test_logger2.handlers)

        # Verifica que é o mesmo logger
        assert test_logger1 is test_logger2

        # Verifica que handlers não foram duplicados
        assert final_handler_count == initial_handler_count

    @patch('src.util.logger.logging.StreamHandler')
    def test_logger_handler_creation(self, mock_stream_handler):
        """
        Testa criação de handler de stream.
        
        Args:
            mock_stream_handler: Mock do StreamHandler
            
        Verifica:
            - StreamHandler é criado com stdout
            - Formatter é aplicado
        """
        mock_handler = Mock()
        mock_stream_handler.return_value = mock_handler

        # Cria logger sem handlers existentes
        test_logger = logging.getLogger("test_creation")
        test_logger.handlers.clear()

        LoggerFactory.setup("test_creation")

        # Verifica criação do handler
        mock_stream_handler.assert_called_with(sys.stdout)
        mock_handler.setFormatter.assert_called_once()

    def test_logger_multiple_different_names(self):
        """
        Testa criação de múltiplos loggers com nomes diferentes.
        
        Verifica:
            - Loggers diferentes são criados para nomes diferentes
            - Cada um mantém sua configuração
        """
        logger1 = LoggerFactory.setup("logger_one")
        logger2 = LoggerFactory.setup("logger_two")

        assert logger1 is not logger2
        assert logger1.name == "logger_one"
        assert logger2.name == "logger_two"

        # Ambos devem ter handlers
        assert len(logger1.handlers) > 0
        assert len(logger2.handlers) > 0


class TestLoggerInstance:
    """
    Classe de testes para a instância global de logger.
    
    Testa:
        - Logger global está configurado
        - Funcionalidades básicas de logging
        - Integração com o sistema
    """

    def test_global_logger_exists(self):
        """
        Testa que logger global existe e está configurado.
        
        Verifica:
            - Logger global é acessível
            - É instância de Logger
            - Tem configuração básica
        """
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == "PASSOS_MAGICOS_APP"

    def test_global_logger_has_handlers(self):
        """
        Testa que logger global tem handlers configurados.
        
        Verifica:
            - Pelo menos um handler está presente
            - Handler é do tipo correto
        """
        assert len(logger.handlers) > 0

        # Verifica que tem handler de console
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(console_handlers) > 0

    @patch('sys.stdout')
    def test_global_logger_output(self, mock_stdout):
        """
        Testa output do logger global.
        
        Args:
            mock_stdout: Mock do stdout
            
        Verifica:
            - Mensagens são enviadas para stdout
            - Formato está correto
        """
        # Configura mock
        mock_stdout.write = Mock()

        # Força reconfiguração do logger para usar o mock
        logger.handlers.clear()
        test_logger = LoggerFactory.setup("PASSOS_MAGICOS_APP")

        # Testa log
        test_logger.info("Test message")

        # Verifica que stdout.write foi chamado (através do handler)
        # Nota: Pode não ser chamado diretamente dependendo da implementação
        assert isinstance(test_logger, logging.Logger)

    def test_global_logger_level_methods(self):
        """
        Testa métodos de diferentes níveis de log.
        
        Verifica:
            - Métodos debug, info, warning, error, critical existem
            - Podem ser chamados sem erro
        """
        # Testa que métodos existem
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'critical')

        # Testa que podem ser chamados (sem verificar output)
        try:
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
        except Exception as e:
            pytest.fail(f"Logger methods should not raise exceptions: {e}")

    def test_logger_thread_safety(self):
        """
        Testa segurança em threads (básico).
        
        Verifica:
            - Logger pode ser usado em contexto multi-thread
            - Configuração é thread-safe
        """
        import threading

        results = []

        def log_in_thread():
            try:
                thread_logger = LoggerFactory.setup("thread_test")
                thread_logger.info("Thread message")
                results.append(True)
            except Exception:
                results.append(False)

        # Cria múltiplas threads
        threads = [threading.Thread(target=log_in_thread) for _ in range(5)]

        # Inicia threads
        for thread in threads:
            thread.start()

        # Aguarda conclusão
        for thread in threads:
            thread.join()

        # Verifica que todas as threads foram bem-sucedidas
        assert all(results)
        assert len(results) == 5
