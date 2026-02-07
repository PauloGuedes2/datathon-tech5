"""
Processamento de features de entrada.

Responsabilidades:
- Calcular tempo na ONG
- Garantir colunas obrigatórias
- Normalizar tipos numéricos e categóricos
"""

from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd

from src.config.settings import Configuracoes


class ProcessadorFeatures:
    """
    Processa DataFrames de entrada para o formato esperado pelo modelo.

    Responsabilidades:
    - Aplicar regras de cálculo de tempo na ONG
    - Preencher colunas ausentes
    - Normalizar tipos de dados
    """

    @staticmethod
    def processar(
        dados: pd.DataFrame,
        data_snapshot: Optional[datetime] = None,
        estatisticas: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Processa o DataFrame de entrada.

        Parâmetros:
        - dados (pd.DataFrame): dados de entrada
        - data_snapshot (datetime | None): data de referência para cálculos
        - estatisticas (dict | None): estatísticas para preenchimento de nulos

        Retorno:
        - pd.DataFrame: DataFrame com features normalizadas
        """
        dados_copia = dados.copy()

        referencia = ProcessadorFeatures._obter_ano_referencia(dados_copia, data_snapshot)
        dados_copia = ProcessadorFeatures._calcular_tempo_ong(dados_copia, referencia, estatisticas)
        dados_copia = ProcessadorFeatures._garantir_colunas_obrigatorias(dados_copia)

        colunas = Configuracoes.FEATURES_NUMERICAS + Configuracoes.FEATURES_CATEGORICAS
        dados_processados = dados_copia[colunas].copy()
        dados_processados = ProcessadorFeatures._normalizar_numericos(dados_processados)
        dados_processados = ProcessadorFeatures._normalizar_categoricos(dados_processados)

        return dados_processados

    @staticmethod
    def _obter_ano_referencia(dados: pd.DataFrame, data_snapshot: Optional[datetime]):
        """
        Obtém o ano de referência para cálculos.

        Parâmetros:
        - dados (pd.DataFrame): dados de entrada
        - data_snapshot (datetime | None): data de referência

        Retorno:
        - int | pd.Series: ano de referência
        """
        if "ANO_REFERENCIA" in dados.columns:
            return pd.to_numeric(dados["ANO_REFERENCIA"], errors="coerce")

        data_atual = data_snapshot or datetime.now()
        return data_atual.year

    @staticmethod
    def _calcular_tempo_ong(
        dados: pd.DataFrame, referencia, estatisticas: Optional[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Calcula a coluna TEMPO_NA_ONG.

        Parâmetros:
        - dados (pd.DataFrame): dados de entrada
        - referencia (int | pd.Series): ano de referência
        - estatisticas (dict | None): estatísticas para preenchimento

        Retorno:
        - pd.DataFrame: DataFrame com TEMPO_NA_ONG atualizado
        """
        if "ANO_INGRESSO" in dados.columns:
            ano_ingresso = pd.to_numeric(dados["ANO_INGRESSO"], errors="coerce")
            ano_ingresso = ProcessadorFeatures._preencher_ano_ingresso(ano_ingresso, estatisticas)
            dados["TEMPO_NA_ONG"] = referencia - ano_ingresso
            dados["TEMPO_NA_ONG"] = dados["TEMPO_NA_ONG"].clip(lower=0)
        else:
            dados["TEMPO_NA_ONG"] = 0

        return dados

    @staticmethod
    def _preencher_ano_ingresso(serie: pd.Series, estatisticas: Optional[Dict[str, Any]]) -> pd.Series:
        """
        Preenche valores nulos de ANO_INGRESSO.

        Parâmetros:
        - serie (pd.Series): série de ano de ingresso
        - estatisticas (dict | None): estatísticas para preenchimento

        Retorno:
        - pd.Series: série com nulos preenchidos
        """
        if not serie.isnull().any():
            return serie

        if estatisticas and "mediana_ano_ingresso" in estatisticas:
            mediana = estatisticas["mediana_ano_ingresso"]
        else:
            mediana = serie.median() if not serie.isnull().all() else 2020

        return serie.fillna(mediana)

    @staticmethod
    def _garantir_colunas_obrigatorias(dados: pd.DataFrame) -> pd.DataFrame:
        """
        Garante a presença de colunas obrigatórias.

        Parâmetros:
        - dados (pd.DataFrame): dados de entrada

        Retorno:
        - pd.DataFrame: DataFrame com colunas garantidas
        """
        colunas_obrigatorias = Configuracoes.FEATURES_NUMERICAS + Configuracoes.FEATURES_CATEGORICAS
        for coluna in colunas_obrigatorias:
            if coluna not in dados.columns:
                dados[coluna] = 0 if coluna in Configuracoes.FEATURES_NUMERICAS else "N/A"
        return dados

    @staticmethod
    def _normalizar_numericos(dados: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza colunas numéricas.

        Parâmetros:
        - dados (pd.DataFrame): DataFrame com features

        Retorno:
        - pd.DataFrame: DataFrame com numéricos normalizados
        """
        for coluna in Configuracoes.FEATURES_NUMERICAS:
            dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce").fillna(0)
        return dados

    @staticmethod
    def _normalizar_categoricos(dados: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza colunas categóricas.

        Parâmetros:
        - dados (pd.DataFrame): DataFrame com features

        Retorno:
        - pd.DataFrame: DataFrame com categóricos normalizados
        """
        for coluna in Configuracoes.FEATURES_CATEGORICAS:
            dados[coluna] = dados[coluna].astype(str).replace("nan", "N/A")
        return dados
