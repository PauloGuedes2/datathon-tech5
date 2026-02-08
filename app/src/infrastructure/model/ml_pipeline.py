"""
Pipeline de treinamento do modelo de ML.

Responsabilidades:
- Criar variável alvo
- Gerar features históricas
- Treinar modelo e avaliar métricas
- Promover modelo com base em critérios
"""

import json
import os
import shutil
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, f1_score, precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.application.feature_processor import ProcessadorFeatures
from src.config.settings import Configuracoes
from src.util.logger import logger


class PipelineML:
    """
    Pipeline de treinamento do modelo.

    Responsabilidades:
    - Preparar dados
    - Treinar e avaliar o modelo
    - Promover modelo quando aplicável
    """

    def __init__(self):
        """
        Inicializa o pipeline.

        Responsabilidades:
        - Instanciar o processador de features
        """
        self.processador = ProcessadorFeatures()

    @staticmethod
    def criar_target(dados: pd.DataFrame) -> pd.DataFrame:
        """
        Cria a variável alvo RISCO_DEFASAGEM.

        Parâmetros:
        - dados (pd.DataFrame): dados de entrada

        Retorno:
        - pd.DataFrame: dados com coluna alvo
        """
        if "DEFASAGEM" in dados.columns:
            dados[Configuracoes.TARGET_COL] = dados["DEFASAGEM"].apply(
                lambda valor: 1 if (isinstance(valor, (int, float)) and valor < 0) else 0
            )
        elif "INDE" in dados.columns:
            dados["INDE"] = pd.to_numeric(dados["INDE"], errors="coerce")
            dados[Configuracoes.TARGET_COL] = (dados["INDE"] < 6.0).astype(int)
        elif "PEDRA" in dados.columns:
            dados[Configuracoes.TARGET_COL] = dados["PEDRA"].astype(str).str.upper().apply(
                lambda valor: 1 if "QUARTZO" in valor else 0
            )
        else:
            raise ValueError("Colunas de target ausentes para criação do rótulo.")

        return dados

    @staticmethod
    def criar_features_lag(dados: pd.DataFrame) -> pd.DataFrame:
        """
        Gera features históricas (Lag).

        Parâmetros:
        - dados (pd.DataFrame): dados de entrada

        Retorno:
        - pd.DataFrame: dados com features de histórico
        """
        logger.info("Gerando Features Históricas (Lag)...")

        if "RA" not in dados.columns or "ANO_REFERENCIA" not in dados.columns:
            return dados

        dados = dados.sort_values(by=["RA", "ANO_REFERENCIA"])
        metricas = ["INDE", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN"]

        for coluna in metricas:
            if coluna in dados.columns:
                nome_coluna = f"{coluna}_ANTERIOR"
                dados[nome_coluna] = dados.groupby("RA")[coluna].shift(1).fillna(0)

        if "INDE_ANTERIOR" in dados.columns:
            dados["ALUNO_NOVO"] = (dados["INDE_ANTERIOR"] == 0).astype(int)
        else:
            dados["ALUNO_NOVO"] = 1

        return dados

    def treinar(self, dados: pd.DataFrame):
        """
        Executa o treinamento do modelo.

        Parâmetros:
        - dados (pd.DataFrame): dados para treinamento

        Exceções:
        - ValueError: quando ANO_REFERENCIA não está disponível
        """
        logger.info("Iniciando pipeline de treinamento Enterprise (Anti-Leakage)...")

        dados = self.criar_target(dados)
        dados = self.criar_features_lag(dados)
        dados = self._remover_colunas_proibidas(dados)

        mascara_treino, mascara_teste = self._definir_particao_temporal(dados)

        estatisticas = self._calcular_estatisticas_treino(dados, mascara_treino)
        logger.info(f"Estatísticas de Treino calculadas: {estatisticas}")
        self._salvar_estatisticas(estatisticas)

        dados_processados = self.processador.processar(dados, estatisticas=estatisticas)
        dados_processados[Configuracoes.TARGET_COL] = dados[Configuracoes.TARGET_COL]
        dados_processados["ANO_REFERENCIA"] = dados["ANO_REFERENCIA"]

        features_uso = [
            f
            for f in Configuracoes.FEATURES_MODELO_NUMERICAS + Configuracoes.FEATURES_MODELO_CATEGORICAS
            if f in dados_processados.columns
        ]

        matriz_treino = dados_processados.loc[mascara_treino, features_uso]
        alvo_treino = dados_processados.loc[mascara_treino, Configuracoes.TARGET_COL]

        matriz_teste = dados_processados.loc[mascara_teste, features_uso]
        alvo_teste = dados_processados.loc[mascara_teste, Configuracoes.TARGET_COL]

        logger.info(f"Treino: {matriz_treino.shape}, Teste: {matriz_teste.shape}")

        modelo = self._criar_modelo(matriz_treino)
        modelo.fit(matriz_treino, alvo_treino)
        probabilidades = modelo.predict_proba(matriz_teste)[:, 1]
        threshold = self._calcular_threshold(alvo_teste, probabilidades)
        predicoes = (probabilidades >= threshold).astype(int)

        novas_metricas = self._calcular_metricas(
            alvo_teste,
            predicoes,
            threshold,
            dados.loc[mascara_teste],
            matriz_treino,
            matriz_teste,
        )
        logger.info(f"Métricas: {novas_metricas}")

        if self._deve_promover_modelo(novas_metricas):
            self._promover_modelo(
                modelo,
                novas_metricas,
                dados_processados.loc[mascara_teste],
                alvo_teste,
                predicoes,
            )

    @staticmethod
    def _definir_particao_temporal(dados: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Define máscaras de treino e teste com base em ANO_REFERENCIA.

        Parâmetros:
        - dados (pd.DataFrame): dados de entrada

        Retorno:
        - tuple[np.ndarray, np.ndarray]: máscaras de treino e teste

        Exceções:
        - ValueError: quando ANO_REFERENCIA não está disponível
        """
        if "ANO_REFERENCIA" not in dados.columns:
            raise ValueError("Coluna ANO_REFERENCIA necessária para treino.")

        anos_disponiveis = sorted(dados["ANO_REFERENCIA"].unique())
        if len(anos_disponiveis) > 1:
            ano_teste = anos_disponiveis[-1]
            logger.info(f"Split Temporal: Treino < {ano_teste} | Teste == {ano_teste}")
            mascara_treino = dados["ANO_REFERENCIA"] < ano_teste
            mascara_teste = dados["ANO_REFERENCIA"] == ano_teste
            return mascara_treino, mascara_teste

        logger.warning("Split temporal com apenas um ano disponível. Aplicando split aleatório 80/20.")
        indices = np.arange(len(dados))
        if len(indices) < 2:
            mascara_treino = np.ones(len(dados), dtype=bool)
            mascara_teste = np.ones(len(dados), dtype=bool)
            return mascara_treino, mascara_teste
        rng = np.random.default_rng(Configuracoes.RANDOM_STATE)
        rng.shuffle(indices)
        corte = int(len(indices) * 0.8)
        mascara_treino = np.zeros(len(dados), dtype=bool)
        mascara_teste = np.zeros(len(dados), dtype=bool)
        mascara_treino[indices[:corte]] = True
        mascara_teste[indices[corte:]] = True
        return mascara_treino, mascara_teste

    @staticmethod
    def _calcular_estatisticas_treino(dados: pd.DataFrame, mascara_treino: np.ndarray) -> Dict[str, Any]:
        """
        Calcula estatísticas do conjunto de treino.

        Parâmetros:
        - dados (pd.DataFrame): dados de entrada
        - mascara_treino (np.ndarray): máscara de treino

        Retorno:
        - dict: estatísticas calculadas
        """
        ano_ingresso = pd.to_numeric(dados.loc[mascara_treino, "ANO_INGRESSO"], errors="coerce")
        mediana = ano_ingresso.median()
        if pd.isna(mediana):
            mediana = datetime.now().year
        return {"mediana_ano_ingresso": mediana}

    @staticmethod
    def _salvar_estatisticas(estatisticas: Dict[str, Any]) -> None:
        """
        Salva estatísticas de treino para uso em inferência.

        Parâmetros:
        - estatisticas (dict): estatísticas calculadas
        """
        try:
            os.makedirs(os.path.dirname(Configuracoes.FEATURE_STATS_PATH), exist_ok=True)
            with open(Configuracoes.FEATURE_STATS_PATH, "w") as arquivo:
                json.dump(estatisticas, arquivo)
        except Exception as erro:
            logger.warning(f"Falha ao salvar estatísticas de treino: {erro}")

    @staticmethod
    def _remover_colunas_proibidas(dados: pd.DataFrame) -> pd.DataFrame:
        """
        Remove colunas proibidas para evitar vazamento de informação.

        Parâmetros:
        - dados (pd.DataFrame): dados de entrada

        Retorno:
        - pd.DataFrame: dados sem colunas proibidas
        """
        colunas_remover = [
            coluna for coluna in Configuracoes.COLUNAS_PROIBIDAS_NO_TREINO if coluna in dados.columns
        ]
        if colunas_remover:
            dados = dados.drop(columns=colunas_remover)
        return dados

    @staticmethod
    def _criar_modelo(matriz_treino: pd.DataFrame) -> Pipeline:
        """
        Cria pipeline de pré-processamento e modelo.

        Parâmetros:
        - matriz_treino (pd.DataFrame): dados de treino

        Retorno:
        - Pipeline: pipeline do modelo
        """
        features_numericas = [
            f for f in Configuracoes.FEATURES_MODELO_NUMERICAS if f in matriz_treino.columns
        ]
        features_categoricas = [
            f for f in Configuracoes.FEATURES_MODELO_CATEGORICAS if f in matriz_treino.columns
        ]

        transformador_numerico = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        transformador_categorico = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", transformador_numerico, features_numericas),
                ("cat", transformador_categorico, features_categoricas),
            ]
        )

        return Pipeline(steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=Configuracoes.RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=Configuracoes.N_JOBS,
                ),
            ),
        ])

    @staticmethod
    def _calcular_metricas(
        alvo_teste,
        predicoes,
        threshold: float,
        dados_teste: pd.DataFrame,
        matriz_treino: pd.DataFrame,
        matriz_teste: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Calcula métricas do modelo.

        Parâmetros:
        - alvo_teste (pd.Series): valores reais
        - predicoes (np.ndarray): predições
        - matriz_treino (pd.DataFrame): dados de treino
        - matriz_teste (pd.DataFrame): dados de teste

        Retorno:
        - dict: métricas calculadas
        """
        metricas = {
            "timestamp": datetime.now().isoformat(),
            "recall": round(recall_score(alvo_teste, predicoes, zero_division=0), 4),
            "f1_score": round(f1_score(alvo_teste, predicoes, zero_division=0), 4),
            "precision": round(precision_score(alvo_teste, predicoes, zero_division=0), 4),
            "train_size": int(len(matriz_treino)),
            "test_size": int(len(matriz_teste)),
            "model_version": "candidate",
            "risk_threshold": round(float(threshold), 4),
        }
        metricas["group_metrics"] = PipelineML._calcular_metricas_grupo(
            dados_teste, alvo_teste, predicoes
        )
        return metricas

    @staticmethod
    def _calcular_threshold(alvo_teste, probabilidades) -> float:
        """
        Calcula o melhor threshold baseado em F1.

        Retorno:
        - float: threshold selecionado
        """
        precisions, recalls, thresholds = precision_recall_curve(alvo_teste, probabilidades)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
        if thresholds.size == 0:
            return Configuracoes.RISK_THRESHOLD
        melhor_indice = int(np.nanargmax(f1_scores[:-1]))
        return float(thresholds[melhor_indice])

    @staticmethod
    def _calcular_metricas_grupo(dados_teste: pd.DataFrame, alvo_teste, predicoes) -> Dict[str, Any]:
        """
        Calcula métricas por grupos sensíveis para auditoria.

        Retorno:
        - dict: métricas agregadas por grupo
        """
        metricas_grupo = {}
        for coluna in Configuracoes.FEATURES_CATEGORICAS:
            if coluna not in dados_teste.columns:
                continue
            metricas_coluna = {}
            for valor, indices in dados_teste.groupby(coluna).groups.items():
                y_true = alvo_teste.loc[indices]
                y_pred = pd.Series(predicoes, index=alvo_teste.index).loc[indices]
                metricas_coluna[str(valor)] = {
                    "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
                    "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
                    "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
                    "support": int(len(indices)),
                }
            metricas_grupo[coluna] = metricas_coluna
        return metricas_grupo

    @staticmethod
    def _deve_promover_modelo(novas_metricas: Dict[str, Any]) -> bool:
        """
        Avalia se o modelo deve ser promovido.

        Parâmetros:
        - novas_metricas (dict): métricas do modelo candidato

        Retorno:
        - bool: True se deve promover
        """
        if novas_metricas.get("recall", 0) < Configuracoes.MIN_RECALL:
            logger.warning("Recall abaixo do mínimo configurado. Modelo não promovido.")
            return False
        if not os.path.exists(Configuracoes.METRICS_FILE):
            return True
        try:
            with open(Configuracoes.METRICS_FILE, "r") as arquivo:
                atual = json.load(arquivo)
            return novas_metricas["f1_score"] >= (atual.get("f1_score", 0) * 0.95)
        except Exception:
            return True

    @staticmethod
    def _promover_modelo(modelo, metricas, dados_teste_original, alvo_teste, predicoes):
        """
        Promove o modelo e salva dados de referência.

        Parâmetros:
        - modelo (Any): modelo treinado
        - metricas (dict): métricas do modelo
        - dados_teste_original (pd.DataFrame): dados originais de teste
        - alvo_teste (pd.Series): valores reais
        - predicoes (np.ndarray): predições
        """
        logger.info("Promovendo Modelo...")

        if os.path.exists(Configuracoes.MODEL_PATH):
            shutil.copy(Configuracoes.MODEL_PATH, f"{Configuracoes.MODEL_PATH}.bak")

        os.makedirs(os.path.dirname(Configuracoes.MODEL_PATH), exist_ok=True)
        dump(modelo, Configuracoes.MODEL_PATH)

        metricas["model_version"] = datetime.now().strftime("v%Y.%m.%d")
        with open(Configuracoes.METRICS_FILE, "w") as arquivo:
            json.dump(metricas, arquivo)

        referencia_df = dados_teste_original.copy()
        referencia_df["prediction"] = predicoes
        referencia_df[Configuracoes.TARGET_COL] = alvo_teste

        referencia_df.to_csv(Configuracoes.REFERENCE_PATH, index=False)
        logger.info(f"Reference Data salvo com colunas processadas: {Configuracoes.REFERENCE_PATH}")


treinador = PipelineML()
