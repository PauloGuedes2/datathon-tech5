import pandas as pd

from src.application.feature_processor import FeatureProcessor
from src.config.settings import Settings


def test_process_fills_missing_columns_and_uses_snapshot_date(monkeypatch):
    df = pd.DataFrame([
        {
            "ANO_INGRESSO": 2020,
            "IDADE": "11",
            "GENERO": None,
        }
    ])

    class FixedDate:
        @classmethod
        def now(cls):
            class _Now:
                year = 2024
            return _Now()

    monkeypatch.setattr("src.application.feature_processor.datetime", FixedDate)

    processed = FeatureProcessor.process(df)

    assert "TEMPO_NA_ONG" in processed.columns
    assert processed.loc[0, "TEMPO_NA_ONG"] == 4
    assert processed.loc[0, "IDADE"] == 11
    genero_value = processed.loc[0, "GENERO"]
    assert pd.isna(genero_value) or genero_value in {"N/A", "None"}
    for col in Settings.FEATURES_NUMERICAS + Settings.FEATURES_CATEGORICAS:
        assert col in processed.columns


def test_process_handles_null_ano_ingresso_with_stats():
    df = pd.DataFrame([
        {
            "ANO_INGRESSO": None,
            "IDADE": 10,
            "GENERO": "Masculino",
            "TURMA": "A",
            "INSTITUICAO_ENSINO": "Escola",
            "FASE": "1A",
        }
    ])

    processed = FeatureProcessor.process(df, snapshot_date=None, stats={"mediana_ano_ingresso": 2019})
    assert processed.loc[0, "TEMPO_NA_ONG"] >= 0


def test_process_default_temporal_when_missing_ano_ingresso():
    df = pd.DataFrame([
        {
            "IDADE": 9,
            "GENERO": "Feminino",
            "TURMA": "B",
            "INSTITUICAO_ENSINO": "Escola",
            "FASE": "2B",
        }
    ])

    processed = FeatureProcessor.process(df, snapshot_date=None)
    assert processed.loc[0, "TEMPO_NA_ONG"] == 0
