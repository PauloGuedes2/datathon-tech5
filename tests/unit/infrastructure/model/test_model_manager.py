from unittest.mock import Mock

import pytest

from src.infrastructure.model.model_manager import ModelManager


def reset_manager():
    ModelManager._instance = None
    ModelManager._model = None


def test_model_manager_singleton():
    reset_manager()
    first = ModelManager()
    second = ModelManager()
    assert first is second


def test_load_model_missing_file(monkeypatch):
    reset_manager()
    monkeypatch.setattr("src.infrastructure.model.model_manager.os.path.exists", lambda path: False)

    manager = ModelManager()
    manager.load_model()

    assert manager._model is None


def test_load_model_success(monkeypatch):
    reset_manager()
    monkeypatch.setattr("src.infrastructure.model.model_manager.os.path.exists", lambda path: True)
    model_obj = Mock()
    monkeypatch.setattr("src.infrastructure.model.model_manager.load", lambda path: model_obj)

    manager = ModelManager()
    manager.load_model()

    assert manager.get_model() is model_obj


def test_load_model_failure(monkeypatch):
    reset_manager()
    monkeypatch.setattr("src.infrastructure.model.model_manager.os.path.exists", lambda path: True)

    def raise_error(path):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.infrastructure.model.model_manager.load", raise_error)

    manager = ModelManager()
    with pytest.raises(RuntimeError):
        manager.load_model()


def test_get_model_raises_when_unavailable(monkeypatch):
    reset_manager()
    monkeypatch.setattr("src.infrastructure.model.model_manager.os.path.exists", lambda path: False)

    manager = ModelManager()

    with pytest.raises(RuntimeError):
        manager.get_model()


def test_load_model_no_reload(monkeypatch):
    reset_manager()
    monkeypatch.setattr("src.infrastructure.model.model_manager.os.path.exists", lambda path: True)
    model_obj = Mock()
    monkeypatch.setattr("src.infrastructure.model.model_manager.load", lambda path: model_obj)

    manager = ModelManager()
    manager.load_model()
    manager.load_model()

    assert manager.get_model() is model_obj
