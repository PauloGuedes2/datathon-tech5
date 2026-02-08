"""Testes do simulador de producao."""

from pathlib import Path
from unittest.mock import Mock
import importlib.util
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "app"))

spec = importlib.util.spec_from_file_location(
    "send_production_simulation",
    ROOT / "scripts" / "send_production_simulation.py",
)
simulador = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(simulador)


def test_normalizar_idade_valor_direto():
    assert simulador._normalizar_idade(10, 2024) == 10


def test_normalizar_idade_por_ano_nascimento():
    assert simulador._normalizar_idade(2008, 2024) == 16


def test_normalizar_idade_por_data_nascimento():
    assert simulador._normalizar_idade("2008-01-01", 2024) == 16


def test_normalizar_ano_ingresso():
    assert simulador._normalizar_ano_ingresso(2020) == 2020
    assert simulador._normalizar_ano_ingresso("x") is None


def test_montar_payload_sem_idade_retorna_none():
    row = pd.Series(
        {
            "RA": "1",
            "NOME": "Aluno",
            "ANO INGRESSO": 2022,
            "GENERO": "Feminino",
            "TURMA": "1A",
            "INSTITUICAO_ENSINO": "Publica",
            "FASE": "1A",
        }
    )
    chaves = {
        "idade": ["IDADE", "DATA DE NASC"],
        "ano_ingresso": ["ANO INGRESSO"],
        "genero": ["GENERO"],
        "turma": ["TURMA"],
        "instituicao": ["INSTITUICAO_ENSINO"],
        "fase": ["FASE"],
        "ano_referencia": ["ANO_REFERENCIA"],
    }

    assert simulador._montar_payload(row, chaves) is None


def test_montar_payload_ok():
    row = pd.Series(
        {
            "RA": "1",
            "NOME": "Aluno",
            "IDADE": 12,
            "ANO INGRESSO": 2022,
            "GENERO": "Feminino",
            "TURMA": "1A",
            "INSTITUICAO_ENSINO": "Publica",
            "FASE": "1A",
        }
    )
    chaves = {
        "idade": ["IDADE", "DATA DE NASC"],
        "ano_ingresso": ["ANO INGRESSO"],
        "genero": ["GENERO"],
        "turma": ["TURMA"],
        "instituicao": ["INSTITUICAO_ENSINO"],
        "fase": ["FASE"],
        "ano_referencia": ["ANO_REFERENCIA"],
    }

    payload = simulador._montar_payload(row, chaves)

    assert payload["IDADE"] == 12
    assert payload["ANO_INGRESSO"] == 2022
    assert payload["FASE"] == "1A"


def test_carregar_dados_reais_usa_carregador(monkeypatch):
    dummy = pd.DataFrame({"RA": ["1"]})
    carregador = Mock()
    carregador.carregar_dados.return_value = dummy

    monkeypatch.setattr(
        "src.infrastructure.data.data_loader.CarregadorDados",
        lambda: carregador,
    )

    assert simulador.carregar_dados_reais().equals(dummy)
