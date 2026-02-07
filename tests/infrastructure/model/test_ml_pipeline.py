"""Testes do pipeline de treinamento."""

from unittest.mock import Mock, mock_open

import numpy as np
import pandas as pd
import pytest

from src.infrastructure.model.ml_pipeline import PipelineML
from src.config.settings import Configuracoes


class PipelineFalso:
    """Pipeline falso para simular comportamento do sklearn."""

    def __init__(self, *args, **kwargs):
        """Inicializa o pipeline falso."""
        self.args = args
        self.kwargs = kwargs
        self.treinado = False

    def fit(self, dados, alvo):
        """Simula o ajuste do modelo."""
        self.treinado = True
        return self

    def predict(self, dados):
        """Simula a predição do modelo."""
        return np.zeros(len(dados), dtype=int)


def test_criar_target_defasagem():
    dados = pd.DataFrame({"DEFASAGEM": [-1, 2]})
    resultado = PipelineML.criar_target(dados)
    assert resultado[Configuracoes.TARGET_COL].tolist() == [1, 0]


def test_criar_target_inde():
    dados = pd.DataFrame({"INDE": [5.0, 7.0, "bad"]})
    resultado = PipelineML.criar_target(dados)
    assert resultado[Configuracoes.TARGET_COL].tolist() == [1, 0, 0]


def test_criar_target_pedra():
    dados = pd.DataFrame({"PEDRA": ["Quartzo", "onix"]})
    resultado = PipelineML.criar_target(dados)
    assert resultado[Configuracoes.TARGET_COL].tolist() == [1, 0]


def test_criar_target_padrao():
    dados = pd.DataFrame({"OTHER": [1]})
    resultado = PipelineML.criar_target(dados)
    assert resultado[Configuracoes.TARGET_COL].tolist() == [0]


def test_criar_features_lag_com_colunas_faltantes():
    dados = pd.DataFrame({"RA": ["1"], "INDE": [5]})
    resultado = PipelineML.criar_features_lag(dados)
    assert "INDE_ANTERIOR" not in resultado.columns


def test_criar_features_lag_gera_flags():
    dados = pd.DataFrame({
        "RA": ["1", "1"],
        "ANO_REFERENCIA": [2022, 2023],
        "INDE": [5.0, 6.0],
    })
    resultado = PipelineML.criar_features_lag(dados)
    assert "INDE_ANTERIOR" in resultado.columns
    assert resultado.loc[resultado["ANO_REFERENCIA"] == 2023, "ALUNO_NOVO"].iloc[0] == 0


def test_deve_promover_modelo_sem_metricas(monkeypatch):
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.os.path.exists", lambda path: False)
    assert PipelineML._deve_promover_modelo({"f1_score": 0.5}) is True


def test_deve_promover_modelo_com_metricas(monkeypatch):
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.os.path.exists", lambda path: True)

    arquivo_mock = mock_open(read_data='{"f1_score": 0.8}')
    monkeypatch.setattr("builtins.open", arquivo_mock)

    assert PipelineML._deve_promover_modelo({"f1_score": 0.76}) is True
    assert PipelineML._deve_promover_modelo({"f1_score": 0.7}) is False


def test_deve_promover_modelo_em_erro(monkeypatch):
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.os.path.exists", lambda path: True)
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    assert PipelineML._deve_promover_modelo({"f1_score": 0.1}) is True


def test_promover_modelo_cria_backup_e_arquivos(monkeypatch):
    modelo = Mock()
    metricas = {"f1_score": 0.9}
    dados_teste = pd.DataFrame({"RA": ["1"]})
    alvo_teste = pd.Series([1])
    predicoes = np.array([1])

    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.os.path.exists", lambda path: True)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.shutil.copy", Mock())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.os.makedirs", Mock())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.dump", Mock())

    arquivo_mock = mock_open()
    monkeypatch.setattr("builtins.open", arquivo_mock)

    class DataFixa:
        """Classe de data fixa para testes."""
        @classmethod
        def now(cls):
            """Retorna um objeto com data fixa."""
            class _Agora:
                """Objeto simples com data fixa."""
                def strftime(self, fmt):
                    """Retorna versão fixa do modelo."""
                    return "v2024.01.01"
            return _Agora()

    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.datetime", DataFixa)
    monkeypatch.setattr(pd.DataFrame, "to_csv", Mock())

    PipelineML._promover_modelo(modelo, metricas, dados_teste, alvo_teste, predicoes)

    assert metricas["model_version"] == "v2024.01.01"


def test_treinar_exige_ano_referencia(dataframe_base):
    pipeline = PipelineML()
    dados = dataframe_base.drop(columns=["ANO_REFERENCIA"])

    with pytest.raises(ValueError):
        pipeline.treinar(dados)


def test_treinar_com_ano_unico(monkeypatch, dataframe_base):
    pipeline = PipelineML()

    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.Pipeline", PipelineFalso)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.ColumnTransformer", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.RandomForestClassifier", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.SimpleImputer", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.StandardScaler", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.OneHotEncoder", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.recall_score", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.f1_score", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.precision_score", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.train_test_split", lambda idx, test_size, random_state: (idx[:1], idx[1:]))
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.PipelineML._deve_promover_modelo", lambda *args, **kwargs: False)

    pipeline.treinar(dataframe_base)


def test_treinar_com_varios_anos_promove(monkeypatch, dataframe_base):
    pipeline = PipelineML()

    dados = pd.concat([
        dataframe_base,
        dataframe_base.assign(RA="2", ANO_REFERENCIA=2024),
    ], ignore_index=True)

    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.Pipeline", PipelineFalso)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.ColumnTransformer", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.RandomForestClassifier", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.SimpleImputer", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.StandardScaler", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.OneHotEncoder", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.recall_score", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.f1_score", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.precision_score", lambda *args, **kwargs: 0.5)

    promovido = Mock()
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.PipelineML._promover_modelo", promovido)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.PipelineML._deve_promover_modelo", lambda *args, **kwargs: True)

    pipeline.treinar(dados)

    promovido.assert_called_once()
