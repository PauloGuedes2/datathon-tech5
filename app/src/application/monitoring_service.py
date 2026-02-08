"""
Serviço de monitoramento de drift.

Responsabilidades:
- Ler dados de referência e produção
- Gerar relatório Evidently em HTML
- Tratar cenários de erro e dados insuficientes
"""

import os
from collections import deque
from io import StringIO

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
            fairness_html = ServicoMonitoramento._gerar_fairness_html(referencia, dados_atual)
            return f"{relatorio.get_html()}{fairness_html}"

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
            linhas = ServicoMonitoramento._ler_ultimas_linhas(
                Configuracoes.LOG_PATH, Configuracoes.LOG_SAMPLE_LIMIT
            )
            if not linhas:
                return "<h1>Aviso: Arquivo de logs vazio ou inválido.</h1>"
            buffer = StringIO("".join(linhas))
            return pd.read_json(buffer, lines=True)
        except ValueError:
            return "<h1>Aviso: Arquivo de logs vazio ou inválido.</h1>"
        except FileNotFoundError:
            return "<h1>Aviso: Nenhum dado de produção ainda. Faça algumas predições na API primeiro.</h1>"

    @staticmethod
    def _ler_ultimas_linhas(caminho: str, limite: int):
        """
        Lê as últimas linhas de um arquivo de log.

        Parâmetros:
        - caminho (str): caminho do arquivo
        - limite (int): quantidade máxima de linhas

        Retorno:
        - list[str]: linhas lidas
        """
        if limite <= 0:
            return []
        with open(caminho, "r", encoding="utf-8") as arquivo:
            return list(deque(arquivo, maxlen=limite))

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
    def _gerar_fairness_html(referencia: pd.DataFrame, atual: pd.DataFrame) -> str:
        """
        Gera um bloco HTML com métricas de fairness por grupo.

        Parâmetros:
        - referencia (pd.DataFrame): dados de referência com target/prediction
        - atual (pd.DataFrame): dados atuais

        Retorno:
        - str: HTML com métricas de fairness ou mensagem de aviso
        """
        grupos = []

        def adicionar_metricas(dataset_nome: str, dados: pd.DataFrame):
            metricas = ServicoMonitoramento._calcular_metricas_fairness(dados)
            if isinstance(metricas, str):
                return f"<h3>{dataset_nome}</h3><p>{metricas}</p>"
            tabela_html = metricas.to_html(index=False, classes="fairness-table")
            return f"<h3>{dataset_nome}</h3>{tabela_html}"

        grupos.append(adicionar_metricas("Referência (Treino/Teste)", referencia))
        if Configuracoes.TARGET_COL in atual.columns:
            grupos.append(adicionar_metricas("Produção (com target)", atual))

        conteudo = "".join(grupos)
        return f"<section><h2>Fairness por Grupo</h2>{conteudo}</section>"

    @staticmethod
    def _calcular_metricas_fairness(dados: pd.DataFrame):
        """
        Calcula FPR/FNR por grupo para análise de fairness.

        Parâmetros:
        - dados (pd.DataFrame): dados com target e prediction

        Retorno:
        - pd.DataFrame | str: métricas por grupo ou mensagem de aviso
        """
        grupo_coluna = Configuracoes.FAIRNESS_GROUP_COL
        target_col = Configuracoes.TARGET_COL

        colunas_necessarias = {grupo_coluna, target_col, "prediction"}
        if not colunas_necessarias.issubset(dados.columns):
            return "Dados insuficientes para calcular fairness (grupo, target ou prediction ausentes)."

        metricas = []
        for grupo, subset in dados.groupby(grupo_coluna):
            y_true = subset[target_col]
            y_pred = subset["prediction"]

            falso_positivo = int(((y_pred == 1) & (y_true == 0)).sum())
            falso_negativo = int(((y_pred == 0) & (y_true == 1)).sum())
            verdadeiro_positivo = int(((y_pred == 1) & (y_true == 1)).sum())
            verdadeiro_negativo = int(((y_pred == 0) & (y_true == 0)).sum())

            fpr_denom = falso_positivo + verdadeiro_negativo
            fnr_denom = falso_negativo + verdadeiro_positivo

            metricas.append(
                {
                    grupo_coluna: grupo,
                    "false_positive_rate": round(falso_positivo / fpr_denom, 4) if fpr_denom else 0.0,
                    "false_negative_rate": round(falso_negativo / fnr_denom, 4) if fnr_denom else 0.0,
                    "support": int(len(subset)),
                }
            )

        return pd.DataFrame(metricas)

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
