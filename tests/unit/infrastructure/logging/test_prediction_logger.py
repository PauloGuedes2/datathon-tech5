from unittest.mock import Mock, mock_open

from src.infrastructure.logging.prediction_logger import PredictionLogger


def reset_logger():
    PredictionLogger._instance = None


def test_prediction_logger_singleton():
    reset_logger()
    first = PredictionLogger()
    second = PredictionLogger()
    assert first is second


def test_log_prediction_success(monkeypatch):
    reset_logger()

    class DummyLock:
        def __init__(self):
            self.entered = False

        def __enter__(self):
            self.entered = True

        def __exit__(self, exc_type, exc, tb):
            return False

    logger = PredictionLogger()
    logger._lock = DummyLock()

    monkeypatch.setattr("src.infrastructure.logging.prediction_logger.uuid.uuid4", lambda: "uuid")

    class FixedDate:
        @classmethod
        def now(cls):
            class _Now:
                def isoformat(self):
                    return "2024-01-01T00:00:00"

            return _Now()

    monkeypatch.setattr("src.infrastructure.logging.prediction_logger.datetime", FixedDate)
    monkeypatch.setattr("src.infrastructure.logging.prediction_logger.os.makedirs", Mock())

    mock_file = mock_open()
    monkeypatch.setattr("builtins.open", mock_file)

    logger.log_prediction({"IDADE": 10}, {"prediction": 1, "risk_probability": 0.5, "risk_label": "ALTO"})

    assert logger._lock.entered is True


def test_log_prediction_serialization_failure(monkeypatch):
    reset_logger()
    logger = PredictionLogger()

    monkeypatch.setattr("src.infrastructure.logging.prediction_logger.json.dumps",
                        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad")))

    error_mock = Mock()
    monkeypatch.setattr("src.infrastructure.logging.prediction_logger.logger", error_mock)

    logger.log_prediction({"bad": object()}, {"prediction": 1})

    error_mock.error.assert_called_once()


def test_log_prediction_write_failure(monkeypatch):
    reset_logger()
    logger = PredictionLogger()

    monkeypatch.setattr("src.infrastructure.logging.prediction_logger.uuid.uuid4", lambda: "uuid")

    class FixedDate:
        @classmethod
        def now(cls):
            class _Now:
                def isoformat(self):
                    return "2024-01-01T00:00:00"

            return _Now()

    monkeypatch.setattr("src.infrastructure.logging.prediction_logger.datetime", FixedDate)

    monkeypatch.setattr("src.infrastructure.logging.prediction_logger.os.makedirs", Mock())

    def raise_open(*args, **kwargs):
        raise OSError("nope")

    monkeypatch.setattr("builtins.open", raise_open)

    error_mock = Mock()
    monkeypatch.setattr("src.infrastructure.logging.prediction_logger.logger", error_mock)

    logger.log_prediction({"IDADE": 10}, {"prediction": 1})

    error_mock.error.assert_called_once()
