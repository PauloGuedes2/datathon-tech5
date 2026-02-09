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
                return f"<div class='fairness-section'><h3>{dataset_nome}</h3><p>{metricas}</p></div>"
            tabela_html = metricas.to_html(index=False, classes="fairness-table")
            resumo = ServicoMonitoramento._resumir_gaps_fairness(metricas)
            barras = ServicoMonitoramento._renderizar_barras_fairness(metricas)
            return f"<div class='fairness-section'><h3>{dataset_nome}</h3>{resumo}{barras}{tabela_html}</div>"

        grupos.append(adicionar_metricas("Referência (Treino/Teste)", referencia))
        if Configuracoes.TARGET_COL in atual.columns:
            grupos.append(adicionar_metricas("Produção (com target)", atual))

        conteudo = "".join(grupos)
        estilos = (
            "<style>"
            ".fairness-wrapper{"
            "--fair-bg:var(--euiColorEmptyShade,#ffffff);"
            "--fair-bg-soft:var(--euiColorLightestShade,#f7f7f9);"
            "--fair-card:var(--euiColorLightestShade,#f9fafc);"
            "--fair-border:var(--euiColorLightShade,#e6e6ef);"
            "--fair-text:var(--euiTextColor,#1d1f2a);"
            "--fair-muted:var(--euiTextSubduedColor,#5b6070);"
            "--fair-chip:var(--euiColorPrimaryTint,#eef2ff);"
            "--fair-chip-border:var(--euiColorPrimaryLightShade,#d8e1ff);"
            "--fair-table-head:var(--euiColorLightestShade,#f3f4f6);"
            "--fair-hover:var(--euiColorLightestShade,#fafafa);"
            "--fair-shadow:rgba(20,20,40,0.08);"
            "--fair-success:var(--euiColorSuccess,#2f855a);"
            "--fair-warning:var(--euiColorWarning,#b7791f);"
            "--fair-danger:var(--euiColorDanger,#c53030);"
            "--fair-primary:var(--euiColorPrimary,#3b82f6);"
            "}"
            ".fairness-wrapper{font-family:Arial,Helvetica,sans-serif;margin:24px 0;padding:20px;"
            "background:linear-gradient(135deg,var(--fair-bg-soft) 0%,var(--fair-bg) 70%);"
            "border:1px solid var(--fair-border);border-radius:14px;box-shadow:0 8px 20px var(--fair-shadow)}"
            ".fairness-header{display:flex;align-items:center;justify-content:space-between;gap:16px}"
            ".fairness-title{font-size:22px;margin:0;color:var(--fair-text)}"
            ".fairness-subtitle{margin:6px 0 0;color:var(--fair-muted);font-size:13px}"
            ".fairness-chip{font-size:12px;font-weight:600;color:var(--fair-text);background:var(--fair-chip);"
            "border:1px solid var(--fair-chip-border);padding:6px 10px;border-radius:999px}"
            ".fairness-section{margin-top:18px;padding:14px 16px;background:var(--fair-bg);"
            "border:1px solid var(--fair-border);border-radius:12px}"
            ".fairness-section h3{margin:0 0 8px;font-size:16px;color:var(--fair-text)}"
            ".fairness-summary{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0 4px}"
            ".fairness-card{flex:1 1 180px;background:var(--fair-card);border:1px solid var(--fair-border);"
            "border-radius:10px;padding:10px 12px}"
            ".fairness-card .label{font-size:12px;color:var(--fair-muted);margin-bottom:4px}"
            ".fairness-card .value{font-size:18px;font-weight:700;color:var(--fair-text)}"
            ".fairness-badges{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 2px}"
            ".fairness-badge{font-size:11px;font-weight:700;letter-spacing:0.3px;padding:4px 8px;"
            "border-radius:999px;border:1px solid var(--fair-border);text-transform:uppercase}"
            ".fairness-badge.low{background:color-mix(in srgb,var(--fair-success) 20%, transparent);"
            "color:var(--fair-success);border-color:color-mix(in srgb,var(--fair-success) 35%, transparent)}"
            ".fairness-badge.med{background:color-mix(in srgb,var(--fair-warning) 20%, transparent);"
            "color:var(--fair-warning);border-color:color-mix(in srgb,var(--fair-warning) 35%, transparent)}"
            ".fairness-badge.high{background:color-mix(in srgb,var(--fair-danger) 20%, transparent);"
            "color:var(--fair-danger);border-color:color-mix(in srgb,var(--fair-danger) 35%, transparent)}"
            ".fairness-bars{margin:10px 0 6px;display:flex;flex-direction:column;gap:8px}"
            ".fairness-bar-row{display:grid;grid-template-columns:140px 1fr 60px;gap:10px;align-items:center}"
            ".fairness-bar-label{font-size:12px;color:var(--fair-muted)}"
            ".fairness-bar-track{height:8px;border-radius:999px;background:var(--fair-table-head);overflow:hidden}"
            ".fairness-bar-fill{height:100%;border-radius:999px}"
            ".fairness-bar-fill.fpr{background:var(--fair-warning)}"
            ".fairness-bar-fill.fnr{background:var(--fair-primary)}"
            ".fairness-bar-value{font-size:12px;color:var(--fair-text);text-align:right}"
            ".fairness-table{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px;color:var(--fair-text)}"
            ".fairness-table th,.fairness-table td{padding:8px 10px;border-bottom:1px solid var(--fair-border)}"
            ".fairness-table th{text-align:left;color:var(--fair-text);background:var(--fair-table-head);"
            "position:sticky;top:0}"
            ".fairness-table tr:hover{background:var(--fair-hover)}"
            "</style>"
        )
        return (
            f"{estilos}"
            "<section class='fairness-wrapper'>"
            "<div class='fairness-header'>"
            "<div>"
            "<h2 class='fairness-title'>Fairness por Grupo</h2>"
            "<p class='fairness-subtitle'>Taxas em %, diferencas altas indicam risco de vies.</p>"
            "</div>"
            "<div class='fairness-chip'>Auditoria de Equidade</div>"
            "</div>"
            f"{conteudo}"
            "</section>"
        )

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

            fpr = round(falso_positivo / fpr_denom, 4) if fpr_denom else 0.0
            fnr = round(falso_negativo / fnr_denom, 4) if fnr_denom else 0.0
            metricas.append(
                {
                    grupo_coluna: grupo,
                    "false_positive_rate_pct": round(fpr * 100, 2),
                    "false_negative_rate_pct": round(fnr * 100, 2),
                    "support": int(len(subset)),
                }
            )

        return pd.DataFrame(metricas)

    @staticmethod
    def _resumir_gaps_fairness(metricas: pd.DataFrame) -> str:
        """
        Resume os gaps de fairness com base nas taxas por grupo.

        Parâmetros:
        - metricas (pd.DataFrame): métricas por grupo

        Retorno:
        - str: HTML com resumo de gaps
        """
        if metricas.empty:
            return "<p>Sem dados suficientes para resumir gaps de fairness.</p>"

        gap_fpr = metricas["false_positive_rate_pct"].max() - metricas["false_positive_rate_pct"].min()
        gap_fnr = metricas["false_negative_rate_pct"].max() - metricas["false_negative_rate_pct"].min()
        def _nivel_gap(valor: float) -> str:
            if valor <= 5:
                return "low"
            if valor <= 10:
                return "med"
            return "high"

        nivel_fpr = _nivel_gap(gap_fpr)
        nivel_fnr = _nivel_gap(gap_fnr)
        return (
            "<div class='fairness-summary'>"
            "<div class='fairness-card'>"
            "<div class='label'>Gap de FPR</div>"
            f"<div class='value'>{gap_fpr:.2f} pp</div>"
            "<div class='fairness-badges'>"
            f"<span class='fairness-badge {nivel_fpr}'>FPR {nivel_fpr}</span>"
            "</div>"
            "</div>"
            "<div class='fairness-card'>"
            "<div class='label'>Gap de FNR</div>"
            f"<div class='value'>{gap_fnr:.2f} pp</div>"
            "<div class='fairness-badges'>"
            f"<span class='fairness-badge {nivel_fnr}'>FNR {nivel_fnr}</span>"
            "</div>"
            "</div>"
            "</div>"
        )

    @staticmethod
    def _renderizar_barras_fairness(metricas: pd.DataFrame) -> str:
        """
        Renderiza barras simples de FPR/FNR por grupo.

        Parâmetros:
        - metricas (pd.DataFrame): métricas por grupo

        Retorno:
        - str: HTML das barras
        """
        if metricas.empty:
            return ""

        max_valor = max(
            metricas["false_positive_rate_pct"].max(),
            metricas["false_negative_rate_pct"].max(),
        )
        if not max_valor or pd.isna(max_valor):
            max_valor = 1.0

        linhas = []
        for _, row in metricas.iterrows():
            grupo = row.iloc[0]
            fpr = float(row["false_positive_rate_pct"])
            fnr = float(row["false_negative_rate_pct"])
            fpr_pct = max(0.0, min(100.0, (fpr / max_valor) * 100.0))
            fnr_pct = max(0.0, min(100.0, (fnr / max_valor) * 100.0))
            linhas.append(
                "<div class='fairness-bar-row'>"
                f"<div class='fairness-bar-label'>{grupo} · FPR</div>"
                "<div class='fairness-bar-track'>"
                f"<div class='fairness-bar-fill fpr' style='width:{fpr_pct:.1f}%'></div>"
                "</div>"
                f"<div class='fairness-bar-value'>{fpr:.2f}%</div>"
                "</div>"
            )
            linhas.append(
                "<div class='fairness-bar-row'>"
                f"<div class='fairness-bar-label'>{grupo} · FNR</div>"
                "<div class='fairness-bar-track'>"
                f"<div class='fairness-bar-fill fnr' style='width:{fnr_pct:.1f}%'></div>"
                "</div>"
                f"<div class='fairness-bar-value'>{fnr:.2f}%</div>"
                "</div>"
            )

        return "<div class='fairness-bars'>" + "".join(linhas) + "</div>"

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
