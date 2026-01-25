import json
import os
from datetime import datetime

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.settings import Settings
from src.util.logger import logger


class MLPipeline:
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria a variável alvo RISCO_DEFASAGEM (1 = Alto Risco, 0 = Baixo Risco).
        Prioriza a coluna 'DEFASAGEM' (negativo indica defasagem) e fallback para INDE/PEDRA.
        """
        # Lógica 1: Se tem a métrica de defasagem calculada
        if "DEFASAGEM" in df.columns:
            # Defasagem negativa indica que o aluno está atrasado -> Risco 1
            df[Settings.TARGET_COL] = df["DEFASAGEM"].apply(
                lambda x: 1 if (isinstance(x, (int, float)) and x < 0) else 0
            )
        # Lógica 2: Fallback para INDE (Nota geral < 6 indica risco)
        elif "INDE" in df.columns:
            df["INDE"] = pd.to_numeric(df["INDE"], errors='coerce')
            df[Settings.TARGET_COL] = (df["INDE"] < 6.0).astype(int)
        # Lógica 3: Fallback para PEDRA (Quartzo é a pedra de menor desempenho)
        elif "PEDRA" in df.columns:
            df[Settings.TARGET_COL] = df["PEDRA"].astype(str).str.upper().apply(
                lambda x: 1 if "QUARTZO" in x else 0
            )
        else:
            raise ValueError("Não foi possível criar o target: Colunas DEFASAGEM, INDE ou PEDRA ausentes.")

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
        # Garante criação de TEMPO_NA_ONG
        if "ANO_INGRESSO" in df.columns and "ANO_REFERENCIA" in df.columns:
            df["ANO_INGRESSO"] = pd.to_numeric(df["ANO_INGRESSO"], errors='coerce')
            mediana_ingresso = df["ANO_INGRESSO"].median()
            df["ANO_INGRESSO"] = df["ANO_INGRESSO"].fillna(mediana_ingresso)
            df["TEMPO_NA_ONG"] = df["ANO_REFERENCIA"] - df["ANO_INGRESSO"]
            df["TEMPO_NA_ONG"] = df["TEMPO_NA_ONG"].clip(lower=0)
        else:
            df["TEMPO_NA_ONG"] = 0
        return df

    def train(self, df: pd.DataFrame):
        logger.info("Iniciando pipeline de treinamento blindado contra Data Leakage...")

        # 1. Pipeline de Dados
        df = self._sanitize_data(df)
        df = self._feature_engineering(df)

        features_to_use = Settings.FEATURES_NUMERICAS + Settings.FEATURES_CATEGORICAS

        # Garante que todas features existem
        for col in features_to_use:
            if col not in df.columns:
                df[col] = 0 if col in Settings.FEATURES_NUMERICAS else "N/A"

        # 2. Split Temporal (CRÍTICO para evitar vazamento)
        # Se tivermos dados de 2024, usamos como teste. Se não, usamos o último ano disponível.
        anos_disponiveis = sorted(df["ANO_REFERENCIA"].unique())
        if len(anos_disponiveis) > 1:
            ano_teste = anos_disponiveis[-1]  # Último ano (ex: 2024)
            logger.info(f"Split Temporal: Treino {anos_disponiveis[:-1]} | Teste {ano_teste}")

            mask_test = df["ANO_REFERENCIA"] == ano_teste
            mask_train = ~mask_test
        else:
            # Fallback se tiver só 1 ano (o que não deve ocorrer no seu dataset)
            logger.warning("Apenas 1 ano de dados detectado. Usando Split Aleatório (Cuidado com Leakage).")
            from sklearn.model_selection import train_test_split
            mask_train, mask_test = train_test_split(df.index, test_size=0.2, random_state=42)

        X_train = df.loc[mask_train, features_to_use]
        y_train = df.loc[mask_train, Settings.TARGET_COL]

        X_test = df.loc[mask_test, features_to_use]
        y_test = df.loc[mask_test, Settings.TARGET_COL]

        logger.info(f"Tamanho Treino: {X_train.shape} | Tamanho Teste: {X_test.shape}")
        logger.info(f"Distribuição Target Treino:\n{y_train.value_counts(normalize=True)}")

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

        # Random Forest robusto
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,  # Evita overfitting
                random_state=Settings.RANDOM_STATE,
                class_weight='balanced',
                n_jobs=-1
            ))
        ])

        pipeline.fit(X_train, y_train)

        # 4. Avaliação
        y_pred = pipeline.predict(X_test)

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test))
        }

        logger.info("=== RESULTADOS (Teste) ===")
        logger.info(json.dumps(metrics, indent=2))

        # Salvar métricas
        os.makedirs(os.path.dirname(Settings.METRICS_FILE), exist_ok=True)
        with open(Settings.METRICS_FILE, "w") as f:
            json.dump(metrics, f)

        # 5. Salvar Modelo
        os.makedirs(os.path.dirname(Settings.MODEL_PATH), exist_ok=True)
        dump(pipeline, Settings.MODEL_PATH)
        logger.info(f"Modelo salvo: {Settings.MODEL_PATH}")

        # 6. Salvar Reference Data para o Evidently (Observabilidade)
        # IMPORTANTE: O dataset de referência deve ser o de TESTE (ou uma validação holdout)
        # para que o monitoramento compare "Mundo Real Atual" vs "O que o modelo viu no teste".
        # Se usarmos o treino, o drift será enviesado.

        logger.info("Gerando Reference Data para Monitoramento...")
        reference_df = X_test.copy()
        reference_df[Settings.TARGET_COL] = y_test

        # O Evidently precisa da coluna 'prediction' para monitorar Performance Drift
        reference_df["prediction"] = y_pred

        os.makedirs(os.path.dirname(Settings.REFERENCE_PATH), exist_ok=True)
        reference_df.to_csv(Settings.REFERENCE_PATH, index=False)
        logger.info(f"Reference Data salvo: {Settings.REFERENCE_PATH}")


trainer = MLPipeline()
