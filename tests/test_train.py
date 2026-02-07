import runpy
from unittest.mock import Mock


def test_train_success(monkeypatch):
    loader_mock = Mock()
    loader_mock.load_data.return_value = Mock()

    trainer_mock = Mock()
    trainer_mock.create_target.return_value = "targeted"

    monkeypatch.setattr("src.infrastructure.data.data_loader.DataLoader", lambda: loader_mock)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.trainer", trainer_mock)

    runpy.run_module("train", run_name="__main__")

    loader_mock.load_data.assert_called_once()
    trainer_mock.create_target.assert_called_once_with(loader_mock.load_data.return_value)
    trainer_mock.train.assert_called_once_with("targeted")


def test_train_failure(monkeypatch):
    loader_mock = Mock()
    loader_mock.load_data.side_effect = RuntimeError("boom")

    trainer_mock = Mock()

    monkeypatch.setattr("src.infrastructure.data.data_loader.DataLoader", lambda: loader_mock)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.trainer", trainer_mock)

    exit_mock = Mock()
    monkeypatch.setattr("builtins.exit", exit_mock)

    runpy.run_module("train", run_name="__main__")

    exit_mock.assert_called_once_with(1)
