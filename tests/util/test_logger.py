import logging

from src.util.logger import LoggerFactory


def test_logger_factory_single_setup(monkeypatch):
    logger_name = "TEST_LOGGER"
    logger = logging.getLogger(logger_name)
    logger.handlers = []

    first = LoggerFactory.setup(logger_name)
    handler_count = len(first.handlers)
    second = LoggerFactory.setup(logger_name)

    assert first is second
    assert len(second.handlers) == handler_count
