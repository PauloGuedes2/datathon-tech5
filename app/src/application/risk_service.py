"""
Serviço de predição de risco.

Responsabilidades:
- Preparar dados de entrada
- Executar predições com o modelo
- Registrar logs de predição
"""

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
            dados_features = self.processador.processar(dados_brutos)

            prob_risco = self.modelo.predict_proba(dados_features)[:, 1][0]
            classe_predicao = int(prob_risco > Configuracoes.RISK_THRESHOLD)
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
