import pandas as pd
import numpy as np
import os
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, recall_score, f1_score

from src.config.settings import Settings
from src.util.logger import logger


class MLPipeline:
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria a variável alvo RISCO_DEFASAGEM."""
        if "DEFASAGEM" in df.columns:
            df[Settings.TARGET_COL] = df["DEFASAGEM"].apply(
                lambda x: 1 if (isinstance(x, (int, float)) and x < 0) else 0
            )
        elif "INDE" in df.columns:
            df["INDE"] = pd.to_numeric(df["INDE"], errors='coerce')
            df[Settings.TARGET_COL] = (df["INDE"] < 6.0).astype(int)
        elif "PEDRA" in df.columns:
            df[Settings.TARGET_COL] = df["PEDRA"].astype(str).str.upper().apply(
                lambda x: 1 if "QUARTZO" in x else 0
            )
        else:
            raise ValueError("Colunas DEFASAGEM, INDE ou PEDRA não encontradas.")
        return df

    def _sanitize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_existentes = [c for c in Settings.FEATURES_NUMERICAS if c in df.columns]
        for col in cols_existentes:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in Settings.FEATURES_CATEGORICAS:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', 'N/A')
        if cols_existentes:
            df[cols_existentes] = df[cols_existentes].fillna(0)
        return df

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        if "ANO_INGRESSO" in df.columns and "ANO_REFERENCIA" in df.columns:
            df["ANO_INGRESSO"] = pd.to_numeric(df["ANO_INGRESSO"], errors='coerce')
            mediana_ingresso = df["ANO_INGRESSO"].median()
            df["ANO_INGRESSO"] = df["ANO_INGRESSO"].fillna(mediana_ingresso)
            df["TEMPO_NA_ONG"] = df["ANO_REFERENCIA"] - df["ANO_INGRESSO"]
            df["TEMPO_NA_ONG"] = df["TEMPO_NA_ONG"].apply(lambda x: x if x >= 0 else 0)
        else:
            df["TEMPO_NA_ONG"] = 0
        return df

    def train(self, df: pd.DataFrame):
        logger.info("Iniciando preparação e sanitização dos dados...")

        # 1. Pipeline de Dados
        df = self._sanitize_data(df)
        df = self._feature_engineering(df)

        features_to_use = Settings.FEATURES_NUMERICAS + Settings.FEATURES_CATEGORICAS
        missing_cols = [col for col in features_to_use if col not in df.columns]
        for col in missing_cols: df[col] = 0

        X = df[features_to_use]
        y = df[Settings.TARGET_COL]

        logger.info(f"Distribuição do target:\n{y.value_counts()}")

        # 2. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Settings.TEST_SIZE, random_state=Settings.RANDOM_STATE, stratify=y
        )

        # 3. Pipeline de Modelo
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, Settings.FEATURES_NUMERICAS),
                ('cat', categorical_transformer, Settings.FEATURES_CATEGORICAS)
            ])

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                random_state=Settings.RANDOM_STATE,
                class_weight='balanced',
                n_jobs=-1
            ))
        ])

        logger.info("Treinando modelo...")
        pipeline.fit(X_train, y_train)

        # 4. Avaliação
        y_pred = pipeline.predict(X_test)

        # --- CORREÇÃO AQUI: Calculando as métricas antes de logar ---
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logger.info("=== RESULTADOS ===")
        logger.info(f"Recall: {recall:.2%}")
        logger.info(f"F1-Score: {f1:.2%}")
        # ------------------------------------------------------------

        # 5. Salvar Modelo
        os.makedirs(os.path.dirname(Settings.MODEL_PATH), exist_ok=True)
        dump(pipeline, Settings.MODEL_PATH)
        logger.info(f"Modelo salvo em: {Settings.MODEL_PATH}")

        # 6. Salvar Reference Data (COM PREDICTION COLUMN)
        logger.info("Gerando dataset de referência para monitoramento...")

        # Predição de probabilidade no dataset de treino (X)
        ref_predictions = pipeline.predict_proba(X)[:, 1]

        reference_df = X.copy()
        reference_df[Settings.TARGET_COL] = y
        reference_df["prediction"] = ref_predictions  # Coluna essencial para o Evidently

        os.makedirs(os.path.dirname(Settings.REFERENCE_PATH), exist_ok=True)
        reference_df.to_csv(Settings.REFERENCE_PATH, index=False)
        logger.info(f"Dataset de referência salvo com sucesso em: {Settings.REFERENCE_PATH}")


trainer = MLPipeline()