from unittest.mock import Mock, mock_open

import numpy as np
import pandas as pd
import pytest

from src.config.settings import Settings
from src.infrastructure.model.ml_pipeline import MLPipeline


class DummyPipeline:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.fitted = False

    def fit(self, X, y):
        self.fitted = True
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def test_create_target_defasagem():
    df = pd.DataFrame({"DEFASAGEM": [-1, 2]})
    result = MLPipeline.create_target(df)
    assert result[Settings.TARGET_COL].tolist() == [1, 0]


def test_create_target_inde():
    df = pd.DataFrame({"INDE": [5.0, 7.0, "bad"]})
    result = MLPipeline.create_target(df)
    assert result[Settings.TARGET_COL].tolist() == [1, 0, 0]


def test_create_target_pedra():
    df = pd.DataFrame({"PEDRA": ["Quartzo", "onix"]})
    result = MLPipeline.create_target(df)
    assert result[Settings.TARGET_COL].tolist() == [1, 0]


def test_create_target_default():
    df = pd.DataFrame({"OTHER": [1]})
    result = MLPipeline.create_target(df)
    assert result[Settings.TARGET_COL].tolist() == [0]


def test_create_lag_features_with_missing_columns():
    df = pd.DataFrame({"RA": ["1"], "INDE": [5]})
    result = MLPipeline.create_lag_features(df)
    assert "INDE_ANTERIOR" not in result.columns


def test_create_lag_features_generates_flags():
    df = pd.DataFrame({
        "RA": ["1", "1"],
        "ANO_REFERENCIA": [2022, 2023],
        "INDE": [5.0, 6.0],
    })
    result = MLPipeline.create_lag_features(df)
    assert "INDE_ANTERIOR" in result.columns
    assert result.loc[result["ANO_REFERENCIA"] == 2023, "ALUNO_NOVO"].iloc[0] == 0


def test_should_promote_model_when_no_metrics(monkeypatch):
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.os.path.exists", lambda path: False)
    assert MLPipeline._should_promote_model({"f1_score": 0.5}) is True


def test_should_promote_model_with_metrics(monkeypatch):
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.os.path.exists", lambda path: True)

    mock_file = mock_open(read_data='{"f1_score": 0.8}')
    monkeypatch.setattr("builtins.open", mock_file)

    assert MLPipeline._should_promote_model({"f1_score": 0.76}) is True
    assert MLPipeline._should_promote_model({"f1_score": 0.7}) is False


def test_should_promote_model_on_error(monkeypatch):
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.os.path.exists", lambda path: True)
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    assert MLPipeline._should_promote_model({"f1_score": 0.1}) is True


def test_promote_model_creates_backup_and_files(monkeypatch, tmp_path):
    model = Mock()
    metrics = {"f1_score": 0.9}
    df_test = pd.DataFrame({"RA": ["1"]})
    y_test = pd.Series([1])
    y_pred = np.array([1])

    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.os.path.exists", lambda path: True)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.shutil.copy", Mock())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.os.makedirs", Mock())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.dump", Mock())

    mock_file = mock_open()
    monkeypatch.setattr("builtins.open", mock_file)

    class FixedDate:
        @classmethod
        def now(cls):
            class _Now:
                def strftime(self, fmt):
                    return "v2024.01.01"
            return _Now()

    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.datetime", FixedDate)

    monkeypatch.setattr(pd.DataFrame, "to_csv", Mock())

    MLPipeline._promote_model(model, metrics, df_test, y_test, y_pred)

    assert metrics["model_version"] == "v2024.01.01"


def test_train_requires_ano_referencia(monkeypatch, base_dataframe):
    pipeline = MLPipeline()
    df = base_dataframe.drop(columns=["ANO_REFERENCIA"])

    with pytest.raises(ValueError):
        pipeline.train(df)


def test_train_with_single_year(monkeypatch, base_dataframe):
    pipeline = MLPipeline()

    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.Pipeline", DummyPipeline)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.ColumnTransformer", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.RandomForestClassifier", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.SimpleImputer", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.StandardScaler", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.OneHotEncoder", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.recall_score", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.f1_score", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.precision_score", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.train_test_split", lambda idx, test_size, random_state: (idx[:1], idx[1:]))
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.MLPipeline._should_promote_model", lambda *args, **kwargs: False)

    pipeline.train(base_dataframe)


def test_train_with_multiple_years_promotes(monkeypatch, base_dataframe):
    pipeline = MLPipeline()

    df = pd.concat([
        base_dataframe,
        base_dataframe.assign(RA="2", ANO_REFERENCIA=2024),
    ], ignore_index=True)

    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.Pipeline", DummyPipeline)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.ColumnTransformer", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.RandomForestClassifier", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.SimpleImputer", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.StandardScaler", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.OneHotEncoder", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.recall_score", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.f1_score", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.precision_score", lambda *args, **kwargs: 0.5)

    promoted = Mock()
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.MLPipeline._promote_model", promoted)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.MLPipeline._should_promote_model", lambda *args, **kwargs: True)

    pipeline.train(df)

    promoted.assert_called_once()
