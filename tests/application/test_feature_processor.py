"""Testes do processador de features."""

import pandas as pd

from src.application.feature_processor import ProcessadorFeatures
from src.config.settings import Configuracoes


def test_processar_preenche_colunas_e_usa_snapshot(monkeypatch):
    df = pd.DataFrame([
        {
            "ANO_INGRESSO": 2020,
            "IDADE": "11",
            "GENERO": None,
        }
    ])

    class DataFixa:
        """Classe de data fixa para testes."""
        @classmethod
        def now(cls):
            """Retorna um objeto com ano fixo."""
            class _Agora:
                """Objeto simples com ano fixo."""
                year = 2024
            return _Agora()

    monkeypatch.setattr("src.application.feature_processor.datetime", DataFixa)

    processado = ProcessadorFeatures.processar(df)

    assert "TEMPO_NA_ONG" in processado.columns
    assert processado.loc[0, "TEMPO_NA_ONG"] == 4
    assert processado.loc[0, "IDADE"] == 11
    valor_genero = processado.loc[0, "GENERO"]
    assert pd.isna(valor_genero) or valor_genero in {"N/A", "None"}
    for coluna in Configuracoes.FEATURES_NUMERICAS + Configuracoes.FEATURES_CATEGORICAS:
        assert coluna in processado.columns


def test_processar_ano_ingresso_nulo_com_estatisticas():
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

    processado = ProcessadorFeatures.processar(df, data_snapshot=None, estatisticas={"mediana_ano_ingresso": 2019})
    assert processado.loc[0, "TEMPO_NA_ONG"] >= 0


def test_processar_sem_ano_ingresso():
    df = pd.DataFrame([
        {
            "IDADE": 9,
            "GENERO": "Feminino",
            "TURMA": "B",
            "INSTITUICAO_ENSINO": "Escola",
            "FASE": "2B",
        }
    ])

    processado = ProcessadorFeatures.processar(df, data_snapshot=None)
    assert processado.loc[0, "TEMPO_NA_ONG"] == 0
