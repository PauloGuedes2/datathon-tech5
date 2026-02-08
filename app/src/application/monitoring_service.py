"""
Serviço de monitoramento de drift.

Responsabilidades:
- Ler dados de referência e produção
- Gerar relatório Evidently em HTML
- Tratar cenários de erro e dados insuficientes
"""

import os

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

from src.config.settings import Configuracoes
from src.util.logger import logger


class ServicoMonitoramento:
    """
    Serviço para monitoramento de Data Drift e Target Drift.

    Responsabilidades:
    - Validar existência de arquivos necessários
    - Preparar dados atuais e de referência
    - Executar e retornar relatório Evidently
    """

    @staticmethod
    def gerar_dashboard() -> str:
        """
        Gera o relatório HTML comparando referência vs produção.

        Retorno:
        - str: HTML do relatório ou mensagens de aviso/erro
        """
        if not os.path.exists(Configuracoes.REFERENCE_PATH):
            return "<h1>Erro: Dataset de Referência não encontrado. Treine o modelo primeiro.</h1>"

        if not os.path.exists(Configuracoes.LOG_PATH):
            return "<h1>Aviso: Nenhum dado de produção ainda. Faça algumas predições na API primeiro.</h1>"

        try:
            referencia = pd.read_csv(Configuracoes.REFERENCE_PATH)
            dados_atual_raw = ServicoMonitoramento._carregar_logs()
            if isinstance(dados_atual_raw, str):
                return dados_atual_raw

            if dados_atual_raw.empty:
                return "<h1>Aviso: Arquivo de logs sem dados.</h1>"

            dados_atual = ServicoMonitoramento._montar_dados_atual(dados_atual_raw)
            referencia, dados_atual = ServicoMonitoramento._filtrar_predicoes_validas(referencia, dados_atual)

            colunas_comuns = list(set(referencia.columns) & set(dados_atual.columns))
            if len(dados_atual) < 5:
                return "<h1>Aguardando mais dados... (Mínimo 5 requisições para gerar relatório confiável)</h1>"

            mapeamento_colunas = ServicoMonitoramento._criar_mapeamento(colunas_comuns, dados_atual)
            relatorio = ServicoMonitoramento._executar_relatorio(
                referencia, dados_atual, mapeamento_colunas
            )
            return relatorio.get_html()

        except Exception as erro:
            logger.error(f"Erro ao gerar dashboard: {erro}")
            return f"<h1>Erro interno ao gerar relatório: {str(erro)}</h1>"

    @staticmethod
    def _carregar_logs():
        """
        Carrega os logs de produção em JSONL.

        Retorno:
        - pd.DataFrame | str: DataFrame com logs ou mensagem HTML de aviso
        """
        try:
            return pd.read_json(Configuracoes.LOG_PATH, lines=True)
        except ValueError:
            return "<h1>Aviso: Arquivo de logs vazio ou inválido.</h1>"

    @staticmethod
    def _montar_dados_atual(dados_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Monta o DataFrame atual combinando features e predições.

        Parâmetros:
        - dados_raw (pd.DataFrame): logs brutos

        Retorno:
        - pd.DataFrame: dados atuais preparados
        """
        features_df = pd.json_normalize(dados_raw["input_features"])
        preds_df = pd.json_normalize(dados_raw["prediction_result"])
        dados_atual = pd.concat([features_df, preds_df], axis=1)

        if "class" in dados_atual.columns:
            dados_atual.rename(columns={"class": "prediction"}, inplace=True)

        return dados_atual

    @staticmethod
    def _filtrar_predicoes_validas(referencia: pd.DataFrame, atual: pd.DataFrame):
        """
        Remove linhas sem coluna de predição.

        Parâmetros:
        - referencia (pd.DataFrame): dados de referência
        - atual (pd.DataFrame): dados atuais

        Retorno:
        - tuple[pd.DataFrame, pd.DataFrame]: referência e atual filtrados
        """
        referencia_filtrada = referencia.dropna(subset=["prediction"])
        atual_filtrado = atual.dropna(subset=["prediction"])
        return referencia_filtrada, atual_filtrado

    @staticmethod
    def _criar_mapeamento(colunas_comuns, dados_atual: pd.DataFrame) -> ColumnMapping:
        """
        Cria o mapeamento de colunas para o Evidently.

        Parâmetros:
        - colunas_comuns (list): colunas presentes em ambos os conjuntos
        - dados_atual (pd.DataFrame): dados atuais

        Retorno:
        - ColumnMapping: configuração de colunas
        """
        mapeamento = ColumnMapping()
        mapeamento.numerical_features = [c for c in Configuracoes.FEATURES_NUMERICAS if c in colunas_comuns]
        mapeamento.categorical_features = [c for c in Configuracoes.FEATURES_CATEGORICAS if c in colunas_comuns]

        if "prediction" in dados_atual.columns:
            mapeamento.prediction = "prediction"

        return mapeamento

    @staticmethod
    def _executar_relatorio(referencia: pd.DataFrame, atual: pd.DataFrame, mapeamento: ColumnMapping) -> Report:
        """
        Executa o relatório de drift.

        Parâmetros:
        - referencia (pd.DataFrame): dados de referência
        - atual (pd.DataFrame): dados atuais
        - mapeamento (ColumnMapping): configuração de colunas

        Retorno:
        - Report: relatório Evidently executado
        """
        metricas = [DataDriftPreset()]
        if ServicoMonitoramento._tem_target_valido(referencia, atual):
            metricas.append(TargetDriftPreset())
        relatorio = Report(metrics=metricas)
        relatorio.run(
            reference_data=referencia,
            current_data=atual,
            column_mapping=mapeamento,
        )
        return relatorio

    @staticmethod
    def _tem_target_valido(referencia: pd.DataFrame, atual: pd.DataFrame) -> bool:
        """
        Verifica se há coluna de target com dados válidos.

        Retorno:
        - bool: True se o target está disponível nos dois conjuntos
        """
        target_col = Configuracoes.TARGET_COL
        if target_col not in referencia.columns or target_col not in atual.columns:
            return False
        return referencia[target_col].notna().any() and atual[target_col].notna().any()
