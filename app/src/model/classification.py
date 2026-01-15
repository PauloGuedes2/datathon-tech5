import logging

import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config.params import Params
from src.model.feature_engineer import FeatureEngineer

logger = logging.getLogger("ModelPipeline")


class StudentRiskModel:
    def __init__(self):
        self.model = None
        self.pipeline = None

    def build_pipeline(self):
        self.pipeline = Pipeline([
            ('feature_engineer', FeatureEngineer()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=Params.RANDOM_STATE,
                class_weight='balanced'
            ))
        ])

    def train(self, df: pd.DataFrame):
        logger.info("Iniciando treinamento...")

        # 1. Definir Features e Target
        features = Params.FEATURES_NUMERICAS + Params.FEATURES_CATEGORICAS

        # Garantir que as colunas existem
        missing_cols = [c for c in features if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Colunas faltando no dataset: {missing_cols}")

        X = df[features]
        y = df[Params.TARGET_COL]

        # 2. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Params.TEST_SIZE, random_state=Params.RANDOM_STATE, stratify=y
        )

        # 3. Treinar
        self.build_pipeline()
        self.pipeline.fit(X_train, y_train)

        # 4. Avaliar
        y_pred = self.pipeline.predict(X_test)
        metrics = classification_report(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logger.info(f"Modelo treinado. F1-Score: {f1:.4f}")
        logger.info(f"\n{metrics}")

        # 5. Salvar
        dump(self.pipeline, Params.MODEL_PATH)
        logger.info(f"Modelo salvo em {Params.MODEL_PATH}")

    def load(self):
        self.pipeline = load(Params.MODEL_PATH)

    def predict(self, input_data: pd.DataFrame) -> list:
        if not self.pipeline:
            self.load()

        # Probabilidade da classe 1 (Risco)
        probas = self.pipeline.predict_proba(input_data)[:, 1]
        return probas
