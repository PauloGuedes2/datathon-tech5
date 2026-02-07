"""Testes do serviço de risco."""

from unittest.mock import Mock

import numpy as np

from src.application.risk_service import ServicoRisco
from src.domain.student import EntradaEstudante
from src.config.settings import Configuracoes


def test_prever_risco_sucesso(estudante_exemplo):
    modelo = Mock()
    modelo.predict_proba.return_value = np.array([[0.2, 0.8]])

    logger_mock = Mock()
    repo_mock = Mock()

    servico = ServicoRisco(modelo=modelo)
    servico.logger = logger_mock
    servico.repositorio = repo_mock

    resultado = servico.prever_risco(estudante_exemplo)

    assert resultado["prediction"] == 1
    assert resultado["risk_label"] == "ALTO RISCO"
    logger_mock.registrar_predicao.assert_called_once()


def test_prever_risco_limite_threshold(estudante_exemplo):
    modelo = Mock()
    modelo.predict_proba.return_value = np.array([[0.6, Configuracoes.RISK_THRESHOLD]])

    servico = ServicoRisco(modelo=modelo)
    servico.logger = Mock()

    resultado = servico.prever_risco(estudante_exemplo)

    assert resultado["prediction"] == 0
    assert resultado["risk_label"] == "BAIXO RISCO"


def test_prever_risco_modelo_nulo():
    servico = ServicoRisco(modelo=None)
    servico.logger = Mock()

    try:
        servico.prever_risco({})
    except RuntimeError as erro:
        assert "Modelo não inicializado" in str(erro)
    else:
        raise AssertionError("RuntimeError esperado")


def test_prever_risco_inteligente_com_historico(entrada_estudante_exemplo):
    modelo = Mock()
    modelo.predict_proba.return_value = np.array([[0.3, 0.7]])

    servico = ServicoRisco(modelo=modelo)
    servico.logger = Mock()
    servico.repositorio = Mock()
    servico.repositorio.obter_historico_estudante.return_value = {
        "INDE_ANTERIOR": 1.0,
        "IAA_ANTERIOR": 1.0,
        "IEG_ANTERIOR": 1.0,
        "IPS_ANTERIOR": 1.0,
        "IDA_ANTERIOR": 1.0,
        "IPP_ANTERIOR": 1.0,
        "IPV_ANTERIOR": 1.0,
        "IAN_ANTERIOR": 1.0,
        "ALUNO_NOVO": 0,
    }

    resultado = servico.prever_risco_inteligente(EntradaEstudante(**entrada_estudante_exemplo))

    assert resultado["prediction"] == 1


def test_prever_risco_inteligente_sem_historico(entrada_estudante_exemplo):
    modelo = Mock()
    modelo.predict_proba.return_value = np.array([[0.7, 0.2]])

    servico = ServicoRisco(modelo=modelo)
    servico.logger = Mock()
    servico.repositorio = Mock()
    servico.repositorio.obter_historico_estudante.return_value = None

    resultado = servico.prever_risco_inteligente(EntradaEstudante(**entrada_estudante_exemplo))

    assert resultado["prediction"] == 0
