from unittest.mock import Mock

import pandas as pd
import pytest

from src.infrastructure.data.data_loader import DataLoader
from src.config.settings import Settings


def test_load_data_no_files(monkeypatch):
    monkeypatch.setattr("src.infrastructure.data.data_loader.glob.glob", lambda path: [])
    monkeypatch.setattr("src.infrastructure.data.data_loader.os.listdir", lambda path: ["file.txt"])

    loader = DataLoader()

    with pytest.raises(FileNotFoundError):
        loader.load_data()


def test_load_data_invalid_excel(monkeypatch):
    monkeypatch.setattr("src.infrastructure.data.data_loader.glob.glob", lambda path: ["file.xlsx"])

    def raise_error(*args, **kwargs):
        raise RuntimeError("invalid")

    monkeypatch.setattr("src.infrastructure.data.data_loader.pd.read_excel", raise_error)

    loader = DataLoader()

    with pytest.raises(RuntimeError):
        loader.load_data()


def test_load_data_ignores_non_year_sheets(monkeypatch):
    monkeypatch.setattr("src.infrastructure.data.data_loader.glob.glob", lambda path: ["file.xlsx"])

    sheet_data = {
        "Resumo": pd.DataFrame({"RA": ["1"], "ANO_INGRESSO": [2020]}),
        "2023": pd.DataFrame({"RA": ["1"], "ANO_INGRESSO": [2020]}),
    }

    monkeypatch.setattr("src.infrastructure.data.data_loader.pd.read_excel", lambda *args, **kwargs: sheet_data)

    loader = DataLoader()
    df = loader.load_data()

    assert "ANO_REFERENCIA" in df.columns
    assert df["ANO_REFERENCIA"].iloc[0] == 2023


def test_load_data_concat_error(monkeypatch):
    monkeypatch.setattr("src.infrastructure.data.data_loader.glob.glob", lambda path: ["file.xlsx"])

    sheet_data = {
        "2023": pd.DataFrame({"RA": ["1"], "ANO_INGRESSO": [2020]}),
        "2024": pd.DataFrame({"RA": ["2"], "ANO_INGRESSO": [2021]}),
    }

    monkeypatch.setattr("src.infrastructure.data.data_loader.pd.read_excel", lambda *args, **kwargs: sheet_data)

    def raise_concat(*args, **kwargs):
        raise RuntimeError("concat")

    monkeypatch.setattr("src.infrastructure.data.data_loader.pd.concat", raise_concat)

    loader = DataLoader()

    with pytest.raises(RuntimeError):
        loader.load_data()


def test_process_dataframe_normalizes_columns():
    df = pd.DataFrame({
        "ra": ["1"],
        "matematica 2023": [10],
        "port 23": [9],
        "ing": [8],
        "defasagem": [0],
        "ano ingresso": [2020],
        "inst ensino": ["Escola"],
    })

    processed = DataLoader._process_dataframe(df, 2023)

    assert "RA" in processed.columns
    assert "NOTA_MAT" in processed.columns
    assert "NOTA_PORT" in processed.columns
    assert "NOTA_ING" in processed.columns
    assert "DEFASAGEM" in processed.columns
    assert "ANO_INGRESSO" in processed.columns
    assert "INSTITUICAO_ENSINO" in processed.columns
