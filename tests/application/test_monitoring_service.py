"""Testes do serviço de monitoramento."""

from unittest.mock import Mock

import pandas as pd

from src.application.monitoring_service import ServicoMonitoramento
from src.config.settings import Configuracoes


def test_gerar_dashboard_sem_referencia(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda path: False)

    html = ServicoMonitoramento.gerar_dashboard()

    assert "Dataset de Referência" in html


def test_gerar_dashboard_sem_logs(monkeypatch):
    def exists(path):
        if path == Configuracoes.REFERENCE_PATH:
            return True
        return False

    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", exists)

    html = ServicoMonitoramento.gerar_dashboard()

    assert "Nenhum dado de produção" in html


def test_gerar_dashboard_logs_invalidos(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda path: True)
    monkeypatch.setattr("src.application.monitoring_service.pd.read_csv", lambda path: pd.DataFrame({"prediction": [1]}))

    def levantar_erro(*args, **kwargs):
        raise ValueError("invalid")

    monkeypatch.setattr("src.application.monitoring_service.pd.read_json", levantar_erro)

    html = ServicoMonitoramento.gerar_dashboard()

    assert "arquivo de logs vazio" in html.lower()


def test_gerar_dashboard_logs_vazios(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda path: True)
    monkeypatch.setattr("src.application.monitoring_service.pd.read_csv", lambda path: pd.DataFrame({"prediction": [1]}))
    monkeypatch.setattr("src.application.monitoring_service.pd.read_json", lambda *args, **kwargs: pd.DataFrame())

    html = ServicoMonitoramento.gerar_dashboard()

    assert "logs sem dados" in html.lower()


def test_gerar_dashboard_aguarda_dados(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda path: True)

    referencia = pd.DataFrame({"prediction": [0, 1, 0]})
    atual_raw = pd.DataFrame({
        "input_features": [{"IDADE": 10}, {"IDADE": 11}],
        "prediction_result": [{"class": 0}, {"class": 1}],
    })

    monkeypatch.setattr("src.application.monitoring_service.pd.read_csv", lambda path: referencia)
    monkeypatch.setattr("src.application.monitoring_service.pd.read_json", lambda *args, **kwargs: atual_raw)

    html = ServicoMonitoramento.gerar_dashboard()

    assert "Aguardando mais dados" in html


def test_gerar_dashboard_sucesso(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda path: True)

    referencia = pd.DataFrame({"prediction": [0, 1, 0, 1, 0], "IDADE": [10, 11, 12, 13, 14]})
    atual_raw = pd.DataFrame({
        "input_features": [
            {"IDADE": 10, "GENERO": "Masculino"},
            {"IDADE": 11, "GENERO": "Feminino"},
            {"IDADE": 12, "GENERO": "Masculino"},
            {"IDADE": 13, "GENERO": "Outro"},
            {"IDADE": 14, "GENERO": "Masculino"},
        ],
        "prediction_result": [
            {"class": 0}, {"class": 1}, {"class": 0}, {"class": 1}, {"class": 0}
        ],
    })

    monkeypatch.setattr("src.application.monitoring_service.pd.read_csv", lambda path: referencia)
    monkeypatch.setattr("src.application.monitoring_service.pd.read_json", lambda *args, **kwargs: atual_raw)

    relatorio = Mock()
    relatorio.get_html.return_value = "<html>ok</html>"

    def fabrica_relatorio(*args, **kwargs):
        return relatorio

    monkeypatch.setattr("src.application.monitoring_service.Report", fabrica_relatorio)

    html = ServicoMonitoramento.gerar_dashboard()

    assert html == "<html>ok</html>"
    relatorio.run.assert_called_once()


def test_gerar_dashboard_trata_excecao(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda path: True)
    monkeypatch.setattr(
        "src.application.monitoring_service.pd.read_csv",
        lambda path: pd.DataFrame({"prediction": [1, 0, 1, 0, 1], "IDADE": [10, 11, 12, 13, 14]}),
    )
    monkeypatch.setattr(
        "src.application.monitoring_service.pd.read_json",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "input_features": [
                    {"IDADE": 10},
                    {"IDADE": 11},
                    {"IDADE": 12},
                    {"IDADE": 13},
                    {"IDADE": 14},
                ],
                "prediction_result": [
                    {"class": 0},
                    {"class": 1},
                    {"class": 0},
                    {"class": 1},
                    {"class": 0},
                ],
            }
        ),
    )

    def levantar_erro(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.application.monitoring_service.Report", levantar_erro)

    html = ServicoMonitoramento.gerar_dashboard()

    assert "Erro interno" in html
