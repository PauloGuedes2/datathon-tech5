import pandas as pd
from src.infrastructure.model.ml_pipeline import MLPipeline


def test_create_target():
    df = pd.DataFrame({
        "DEFAS": [-1, 0, 2, -3]
    })

    df_out = MLPipeline.create_target(df)

    assert "RISCO_DEFASAGEM" in df_out.columns
    assert list(df_out["RISCO_DEFASAGEM"]) == [1, 0, 0, 1]
