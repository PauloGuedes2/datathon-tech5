import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config.settings import Settings
from src.infrastructure.model.feature_engineer import FeatureEngineer
from src.util.logger import logger


class MLPipeline:
    def __init__(self):
        self.model = None

    @staticmethod
    def create_target(df: pd.DataFrame) -> pd.DataFrame:
        if "DEFAS" not in df.columns:
            raise ValueError("Coluna DEFAS não encontrada no dataset.")

        # 1 = RISCO (defasagem), 0 = OK
        df[Settings.TARGET_COL] = (df["DEFAS"] < 0).astype(int)

        return df

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.model:
            self.load()
        if not self.model:
            raise RuntimeError("Modelo indisponível.")

        clf = self.model.named_steps["clf"]
        features = Settings.FEATURES_NUMERICAS + Settings.FEATURES_CATEGORICAS

        importances = clf.feature_importances_

        df_importance = (
            pd.DataFrame({
                "feature": features,
                "importance": importances
            })
            .sort_values(by="importance", ascending=False)
            .reset_index(drop=True)
        )

        return df_importance

    @staticmethod
    def train(df: pd.DataFrame):
        X = df[Settings.FEATURES_NUMERICAS + Settings.FEATURES_CATEGORICAS]
        y = df[Settings.TARGET_COL]

        # Log do desbalanceamento (IMPORTANTE PARA BANCA)
        class_dist = y.value_counts()
        logger.info(f"Distribuição do target:\n{class_dist}")

        # Fallback inteligente para stratify
        stratify = y if class_dist.min() >= 2 else None
        if stratify is None:
            logger.warning(
                "Classe minoritária com menos de 2 amostras. "
                "Treinando sem stratify."
            )

        logger.info(f"FEATURES USADAS NO TREINO: {list(X.columns)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=Settings.TEST_SIZE,
            random_state=Settings.RANDOM_STATE,
            stratify=stratify
        )

        pipeline = Pipeline([
            ("fe", FeatureEngineer()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=Settings.RANDOM_STATE,
                    class_weight="balanced"  # 🔥 FUNDAMENTAL
                )
            )
        ])

        pipeline.fit(X_train, y_train)

        # Avaliação focada em risco
        y_pred = pipeline.predict(X_test)
        logger.info(
            "Relatório de Classificação:\n"
            f"{classification_report(y_test, y_pred, zero_division=0)}"
        )

        dump(pipeline, Settings.MODEL_PATH)
        logger.info(f"Modelo salvo em {Settings.MODEL_PATH}")

    def load(self):
        try:
            self.model = load(Settings.MODEL_PATH)
        except FileNotFoundError:
            logger.error("Modelo não encontrado. Execute 'train.py' primeiro.")
            self.model = None

    def predict_proba(self, df: pd.DataFrame) -> float:
        if not self.model:
            self.load()
        if not self.model:
            raise RuntimeError("Modelo indisponível.")

        return self.model.predict_proba(df)[:, 1][0]
