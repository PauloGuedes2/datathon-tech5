from unittest.mock import Mock

import pandas as pd

from src.infrastructure.data.historical_repository import HistoricalRepository


def reset_repository():
    HistoricalRepository._instance = None
    HistoricalRepository._data = None


def test_repository_loads_from_reference_with_ra(monkeypatch):
    reset_repository()

    monkeypatch.setattr("src.infrastructure.data.historical_repository.os.path.exists", lambda path: True)
    monkeypatch.setattr(
        "src.infrastructure.data.historical_repository.pd.read_csv",
        lambda path: pd.DataFrame({"RA": ["1"], "ANO_REFERENCIA": [2023], "INDE": [5.0]}),
    )

    repo = HistoricalRepository()

    history = repo.get_student_history("1")

    assert history["INDE_ANTERIOR"] == 5.0


def test_repository_reload_when_reference_missing_ra(monkeypatch):
    reset_repository()

    monkeypatch.setattr("src.infrastructure.data.historical_repository.os.path.exists", lambda path: True)
    monkeypatch.setattr(
        "src.infrastructure.data.historical_repository.pd.read_csv",
        lambda path: pd.DataFrame({"ANO_REFERENCIA": [2023]}),
    )

    loader_mock = Mock()
    loader_mock.load_data.return_value = pd.DataFrame({"RA": ["2"], "ANO_REFERENCIA": [2023], "INDE": [3.0]})
    monkeypatch.setattr("src.infrastructure.data.data_loader.DataLoader", lambda: loader_mock)

    repo = HistoricalRepository()

    history = repo.get_student_history("2")

    assert history["INDE_ANTERIOR"] == 3.0


def test_repository_returns_empty_when_no_data(monkeypatch):
    reset_repository()

    monkeypatch.setattr("src.infrastructure.data.historical_repository.os.path.exists", lambda path: False)

    loader_mock = Mock()
    loader_mock.load_data.side_effect = RuntimeError("boom")
    monkeypatch.setattr("src.infrastructure.data.data_loader.DataLoader", lambda: loader_mock)

    repo = HistoricalRepository()

    assert repo.get_student_history("1") == {}


def test_repository_returns_none_when_student_missing(monkeypatch):
    reset_repository()

    monkeypatch.setattr("src.infrastructure.data.historical_repository.os.path.exists", lambda path: True)
    monkeypatch.setattr(
        "src.infrastructure.data.historical_repository.pd.read_csv",
        lambda path: pd.DataFrame({"RA": ["1"], "ANO_REFERENCIA": [2023], "INDE": [5.0]}),
    )

    repo = HistoricalRepository()

    assert repo.get_student_history("999") is None


def test_repository_safe_get_handles_invalid_values(monkeypatch):
    reset_repository()

    monkeypatch.setattr("src.infrastructure.data.historical_repository.os.path.exists", lambda path: True)
    monkeypatch.setattr(
        "src.infrastructure.data.historical_repository.pd.read_csv",
        lambda path: pd.DataFrame({"RA": ["1"], "ANO_REFERENCIA": [2023], "INDE": ["x"], "IAA": [None]}),
    )

    repo = HistoricalRepository()

    history = repo.get_student_history("1")

    assert history["INDE_ANTERIOR"] == 0.0
    assert history["IAA_ANTERIOR"] == 0.0
