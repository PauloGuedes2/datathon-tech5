"""Testes do controlador de monitoramento."""

from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.monitoring_controller import ControladorMonitoramento, obter_servico_monitoramento


def test_endpoint_painel():
    aplicacao = FastAPI()
    controlador = ControladorMonitoramento()

    servico = Mock()
    servico.gerar_dashboard.return_value = "<html>ok</html>"

    def override():
        return servico

    aplicacao.dependency_overrides[obter_servico_monitoramento] = override
    aplicacao.include_router(controlador.roteador, prefix="/api/v1/monitoring")

    cliente = TestClient(aplicacao)
    resposta = cliente.get("/api/v1/monitoring/dashboard")

    assert resposta.status_code == 200
    assert "<html>ok</html>" in resposta.text
