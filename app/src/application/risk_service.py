"""
Serviço de predição de risco.

Responsabilidades:
- Preparar dados de entrada
- Executar predições com o modelo
- Registrar logs de predição
"""

import json
import os
import pandas as pd

from src.application.feature_processor import ProcessadorFeatures
from src.config.settings import Configuracoes
from src.domain.student import EntradaEstudante, Estudante
from src.infrastructure.data.historical_repository import RepositorioHistorico
from src.infrastructure.logging.prediction_logger import LoggerPredicao
from src.util.logger import logger


class ServicoRisco:
    """
    Serviço para predição de risco de defasagem acadêmica.

    Responsabilidades:
    - Converter entradas em DataFrame
    - Aplicar processamento de features
    - Calcular probabilidade e classe de risco
    - Persistir logs de predição
    """

    def __init__(self, modelo):
        """
        Inicializa o serviço com o modelo.

        Parâmetros:
        - modelo (Any): modelo de ML carregado
        """
        self.modelo = modelo
        self.processador = ProcessadorFeatures()
        self.logger = LoggerPredicao()
        self.repositorio = RepositorioHistorico()

    def prever_risco(self, dados_estudante: dict) -> dict:
        """
        Realiza a predição de risco.

        Parâmetros:
        - dados_estudante (dict): dados completos do aluno

        Retorno:
        - dict: resultado da predição

        Exceções:
        - RuntimeError: quando o modelo não está inicializado
        - Exception: quando ocorre erro na inferência
        """
        if not self.modelo:
            raise RuntimeError("Serviço indisponível: Modelo não inicializado.")

        try:
            dados_brutos = pd.DataFrame([dados_estudante])
            estatisticas = self._carregar_estatisticas()
            dados_features = self.processador.processar(dados_brutos, estatisticas=estatisticas)

            prob_risco = self.modelo.predict_proba(dados_features)[:, 1][0]
            threshold = self._obter_threshold()
            classe_predicao = int(prob_risco >= threshold)
            rotulo_risco = "ALTO RISCO" if classe_predicao == 1 else "BAIXO RISCO"

            resultado = {
                "risk_probability": round(float(prob_risco), 4),
                "risk_label": rotulo_risco,
                "prediction": classe_predicao,
            }

            features = dados_features.to_dict(orient="records")[0]
            self.logger.registrar_predicao(features=features, dados_predicao=resultado)

            return resultado

        except Exception as erro:
            logger.error(f"Erro na inferência: {erro}")
            raise erro

    @staticmethod
    def _obter_threshold() -> float:
        """
        Obtém o threshold configurado ou salvo em métricas.

        Retorno:
        - float: threshold de risco
        """
        try:
            if Configuracoes.METRICS_FILE and os.path.exists(Configuracoes.METRICS_FILE):
                with open(Configuracoes.METRICS_FILE, "r") as arquivo:
                    metricas = json.load(arquivo)
                return float(metricas.get("risk_threshold", Configuracoes.RISK_THRESHOLD))
        except Exception as erro:
            logger.warning(f"Falha ao carregar threshold salvo: {erro}")
        return Configuracoes.RISK_THRESHOLD

    @staticmethod
    def _carregar_estatisticas() -> dict:
        """
        Carrega estatísticas de treino salvas.

        Retorno:
        - dict: estatísticas ou vazio se indisponível
        """
        try:
            if os.path.exists(Configuracoes.FEATURE_STATS_PATH):
                with open(Configuracoes.FEATURE_STATS_PATH, "r") as arquivo:
                    return json.load(arquivo)
        except Exception as erro:
            logger.warning(f"Falha ao carregar estatísticas de treino: {erro}")
        return {}

    def prever_risco_inteligente(self, entrada: EntradaEstudante) -> dict:
        """
        Predição inteligente que busca histórico automaticamente.

        Parâmetros:
        - entrada (EntradaEstudante): dados básicos do aluno

        Retorno:
        - dict: resultado da predição
        """
        historico = self.repositorio.obter_historico_estudante(entrada.RA)

        if historico:
            logger.info(f"Histórico encontrado para RA: {entrada.RA}")
        else:
            logger.info(f"Aluno novo ou sem histórico (RA: {entrada.RA})")
            historico = {
                "INDE_ANTERIOR": 0.0,
                "IAA_ANTERIOR": 0.0,
                "IEG_ANTERIOR": 0.0,
                "IPS_ANTERIOR": 0.0,
                "IDA_ANTERIOR": 0.0,
                "IPP_ANTERIOR": 0.0,
                "IPV_ANTERIOR": 0.0,
                "IAN_ANTERIOR": 0.0,
                "ALUNO_NOVO": 1,
            }

        dados_completos = entrada.model_dump()
        dados_completos.update(historico)

        estudante = Estudante(**dados_completos)
        return self.prever_risco(estudante.model_dump())
