from unittest.mock import Mock

import pandas as pd

from src.application.monitoring_service import MonitoringService
from src.config.settings import Settings


def test_generate_dashboard_no_reference(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda path: False)

    html = MonitoringService.generate_dashboard()

    assert "Dataset de Referência" in html


def test_generate_dashboard_no_logs(monkeypatch):
    def exists(path):
        if path == Settings.REFERENCE_PATH:
            return True
        return False

    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", exists)

    html = MonitoringService.generate_dashboard()

    assert "Nenhum dado de produção" in html


def test_generate_dashboard_invalid_logs(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda path: True)

    monkeypatch.setattr("src.application.monitoring_service.pd.read_csv", lambda path: pd.DataFrame({"prediction": [1]}))

    def raise_value_error(*args, **kwargs):
        raise ValueError("invalid")

    monkeypatch.setattr("src.application.monitoring_service.pd.read_json", raise_value_error)

    html = MonitoringService.generate_dashboard()

    assert "arquivo de logs vazio" in html.lower()


def test_generate_dashboard_empty_logs(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda path: True)
    monkeypatch.setattr("src.application.monitoring_service.pd.read_csv", lambda path: pd.DataFrame({"prediction": [1]}))
    monkeypatch.setattr("src.application.monitoring_service.pd.read_json", lambda *args, **kwargs: pd.DataFrame())

    html = MonitoringService.generate_dashboard()

    assert "logs sem dados" in html.lower()


def test_generate_dashboard_waits_for_more(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda path: True)

    reference = pd.DataFrame({"prediction": [0, 1, 0]})
    current_raw = pd.DataFrame({
        "input_features": [{"IDADE": 10}, {"IDADE": 11}],
        "prediction_result": [{"class": 0}, {"class": 1}],
    })

    monkeypatch.setattr("src.application.monitoring_service.pd.read_csv", lambda path: reference)
    monkeypatch.setattr("src.application.monitoring_service.pd.read_json", lambda *args, **kwargs: current_raw)

    html = MonitoringService.generate_dashboard()

    assert "Aguardando mais dados" in html


def test_generate_dashboard_success(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda path: True)

    reference = pd.DataFrame({"prediction": [0, 1, 0, 1, 0], "IDADE": [10, 11, 12, 13, 14]})
    current_raw = pd.DataFrame({
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

    monkeypatch.setattr("src.application.monitoring_service.pd.read_csv", lambda path: reference)
    monkeypatch.setattr("src.application.monitoring_service.pd.read_json", lambda *args, **kwargs: current_raw)

    report_mock = Mock()
    report_mock.get_html.return_value = "<html>ok</html>"

    def report_factory(*args, **kwargs):
        return report_mock

    monkeypatch.setattr("src.application.monitoring_service.Report", report_factory)

    html = MonitoringService.generate_dashboard()

    assert html == "<html>ok</html>"
    report_mock.run.assert_called_once()


def test_generate_dashboard_handles_exception(monkeypatch):
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

    def raise_error(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.application.monitoring_service.Report", raise_error)

    html = MonitoringService.generate_dashboard()

    assert "Erro interno" in html
