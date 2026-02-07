import json
import os
import shutil
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import numpy as np
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.application.feature_processor import FeatureProcessor
from src.config.settings import Settings
from src.util.logger import logger


class MLPipeline:
    def __init__(self):
        self.processor = FeatureProcessor()

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
            # Se não tiver target, assume 0 (apenas para evitar crash em inferência simulada)
            # Mas em treino deve falhar.
            logger.warning("Colunas de Target ausentes. Criando target dummy.")
            df[Settings.TARGET_COL] = 0

        return df

    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera features históricas (Lag)."""
        logger.info("Gerando Features Históricas (Lag)...")

        if "RA" not in df.columns or "ANO_REFERENCIA" not in df.columns:
            return df

        df = df.sort_values(by=["RA", "ANO_REFERENCIA"])
        metricas = ["INDE", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN"]

        for col in metricas:
            if col in df.columns:
                col_name = f"{col}_ANTERIOR"
                df[col_name] = df.groupby("RA")[col].shift(1).fillna(0)

        if "INDE_ANTERIOR" in df.columns:
            df["ALUNO_NOVO"] = (df["INDE_ANTERIOR"] == 0).astype(int)
        else:
            df["ALUNO_NOVO"] = 1

        return df

    def train(self, df: pd.DataFrame):
        logger.info("Iniciando pipeline de treinamento Enterprise (Anti-Leakage)...")

        # 1. Preparação Básica (Target e Lags)
        # Lags dependem apenas do passado, então podem ser feitos antes do split
        df = self.create_target(df)
        df = self.create_lag_features(df)

        # 2. Definição do Split Temporal (ANTES do processamento)
        # Necessário para calcular estatísticas apenas no treino
        if "ANO_REFERENCIA" in df.columns:
            anos_disponiveis = sorted(df["ANO_REFERENCIA"].unique())
            if len(anos_disponiveis) > 1:
                ano_teste = anos_disponiveis[-1]
                logger.info(f"Split Temporal: Treino < {ano_teste} | Teste == {ano_teste}")
                mask_train = df["ANO_REFERENCIA"] < ano_teste
                mask_test = df["ANO_REFERENCIA"] == ano_teste
            else:
                logger.warning("Apenas 1 ano disponível. Split Aleatório.")
                # Cria índices aleatórios mantendo consistência
                indices = np.arange(len(df))
                train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
                mask_train = np.zeros(len(df), dtype=bool)
                mask_test = np.zeros(len(df), dtype=bool)
                mask_train[train_idx] = True
                mask_test[test_idx] = True
        else:
            raise ValueError("Coluna ANO_REFERENCIA necessária para treino.")

        # 3. Cálculo de Estatísticas no Treino (Evita Leakage)
        ano_ingresso_train = pd.to_numeric(df.loc[mask_train, "ANO_INGRESSO"], errors='coerce')
        mediana_treino = ano_ingresso_train.median()
        if pd.isna(mediana_treino): mediana_treino = 2020

        stats = {"mediana_ano_ingresso": mediana_treino}
        logger.info(f"Estatísticas de Treino calculadas: {stats}")

        # 4. Processamento (Aplicando stats do treino em tudo)
        X_processed = self.processor.process(df, stats=stats)

        # 5. Recolocamos Target e Ano para separação
        X_processed[Settings.TARGET_COL] = df[Settings.TARGET_COL]
        X_processed["ANO_REFERENCIA"] = df["ANO_REFERENCIA"]

        # 6. Aplicação do Split
        features_to_use = [f for f in Settings.FEATURES_NUMERICAS + Settings.FEATURES_CATEGORICAS
                           if f in X_processed.columns]

        X_train = X_processed.loc[mask_train, features_to_use]
        y_train = X_processed.loc[mask_train, Settings.TARGET_COL]

        X_test = X_processed.loc[mask_test, features_to_use]
        y_test = X_processed.loc[mask_test, Settings.TARGET_COL]

        logger.info(f"Treino: {X_train.shape}, Teste: {X_test.shape}")

        # 7. Pipeline e Modelo
        numeric_features = [f for f in Settings.FEATURES_NUMERICAS if f in X_train.columns]
        categorical_features = [f for f in Settings.FEATURES_CATEGORICAS if f in X_train.columns]

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
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=Settings.RANDOM_STATE,
                class_weight='balanced', n_jobs=-1
            ))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 8. Métricas
        new_metrics = {
            "timestamp": datetime.now().isoformat(),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "model_version": "candidate"
        }

        logger.info(f"Métricas: {new_metrics}")

        if self._should_promote_model(new_metrics):
            # Passamos o 'df' original (com colunas brutas) para salvar o Reference Data correto
            self._promote_model(model, new_metrics, df.loc[mask_test], y_test, y_pred)

    def _should_promote_model(self, new_metrics: Dict[str, Any]) -> bool:
        if not os.path.exists(Settings.METRICS_FILE):
            return True
        try:
            with open(Settings.METRICS_FILE, "r") as f:
                current = json.load(f)
            # Aceita se não cair mais que 5%
            return new_metrics["f1_score"] >= (current.get("f1_score", 0) * 0.95)
        except:
            return True

    def _promote_model(self, model, metrics, df_test_original, y_test, y_pred):
        """
        Promove o modelo e salva dados de referência.
        Args:
            df_test_original: DataFrame ORIGINAL do conjunto de teste (contém INDE, PEDRA, etc.)
                              Isso corrige o bug de perder histórico para o próximo ano.
        """
        logger.info("Promovendo Modelo...")

        # Backup e Save Model (Igual original)
        if os.path.exists(Settings.MODEL_PATH):
            shutil.copy(Settings.MODEL_PATH, f"{Settings.MODEL_PATH}.bak")

        os.makedirs(os.path.dirname(Settings.MODEL_PATH), exist_ok=True)
        dump(model, Settings.MODEL_PATH)

        # Save Metrics
        metrics["model_version"] = datetime.now().strftime("v%Y.%m.%d")
        with open(Settings.METRICS_FILE, "w") as f:
            json.dump(metrics, f)

        # Save Reference Data (CORRIGIDO)
        # Salva as colunas originais + predição
        reference_df = df_test_original.copy()
        reference_df["prediction"] = y_pred

        # Opcional: Garantir que target está lá
        reference_df[Settings.TARGET_COL] = y_test

        reference_df.to_csv(Settings.REFERENCE_PATH, index=False)
        logger.info(f"Reference Data salvo com colunas originais: {Settings.REFERENCE_PATH}")


trainer = MLPipeline()