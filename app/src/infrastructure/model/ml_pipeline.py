import json
import os
import shutil
from datetime import datetime
from typing import Dict, Any

import pandas as pd
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
        """
        Cria a variável alvo RISCO_DEFASAGEM.
        Prioridade: DEFASAGEM > INDE > PEDRA.
        """
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
            raise ValueError("Não foi possível criar o target: Colunas DEFASAGEM, INDE ou PEDRA ausentes.")

        return df

    def train(self, df: pd.DataFrame):
        """
        Executa o pipeline completo de treinamento com verificação de qualidade (Quality Gate).
        """
        logger.info("Iniciando pipeline de treinamento Enterprise...")

        # ---------------------------------------------------------
        # 1. Preparação dos Dados (Target + Features)
        # ---------------------------------------------------------
        df = self.create_target(df)

        # Uso do FeatureProcessor para garantir consistência com a API (Requisito 1)
        # O processador lida com nulos e cálculos temporais (TEMPO_NA_ONG)
        X_processed = self.processor.process(df)

        # Recolocamos colunas auxiliares necessárias para o split e treino
        X_processed[Settings.TARGET_COL] = df[Settings.TARGET_COL]
        # Precisamos do ANO_REFERENCIA original para fazer o split temporal correto
        if "ANO_REFERENCIA" in df.columns:
            X_processed["ANO_REFERENCIA"] = df["ANO_REFERENCIA"]
        else:
            # Fallback se não existir (não deveria acontecer no treino)
            X_processed["ANO_REFERENCIA"] = datetime.now().year

        # ---------------------------------------------------------
        # 2. Split Temporal (Evita Data Leakage do Futuro)
        # ---------------------------------------------------------
        anos_disponiveis = sorted(X_processed["ANO_REFERENCIA"].unique())

        if len(anos_disponiveis) > 1:
            # Se temos múltiplos anos, o último ano é SEMPRE teste
            ano_teste = anos_disponiveis[-1]
            logger.info(f"Split Temporal: Treino (Anos < {ano_teste}) | Teste (Ano {ano_teste})")

            mask_test = X_processed["ANO_REFERENCIA"] == ano_teste
            mask_train = ~mask_test
        else:
            # Se só temos um ano, fazemos split aleatório tradicional
            logger.warning("Apenas um ano de dados disponível. Usando Split Aleatório (80/20).")
            mask_train, mask_test = train_test_split(X_processed.index, test_size=0.2, random_state=42)

        features_to_use = Settings.FEATURES_NUMERICAS + Settings.FEATURES_CATEGORICAS

        X_train = X_processed.loc[mask_train, features_to_use]
        y_train = X_processed.loc[mask_train, Settings.TARGET_COL]

        X_test = X_processed.loc[mask_test, features_to_use]
        y_test = X_processed.loc[mask_test, Settings.TARGET_COL]

        logger.info(f"Dimensões -> Treino: {X_train.shape} | Teste: {X_test.shape}")
        logger.info(f"Taxa de Risco (Treino): {y_train.mean():.2%}")

        # ---------------------------------------------------------
        # 3. Definição do Pipeline do Modelo
        # ---------------------------------------------------------
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

        # Modelo Candidato
        candidate_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=Settings.RANDOM_STATE,
                class_weight='balanced',
                n_jobs=-1
            ))
        ])

        # ---------------------------------------------------------
        # 4. Treinamento e Avaliação
        # ---------------------------------------------------------
        candidate_model.fit(X_train, y_train)
        y_pred = candidate_model.predict(X_test)

        # Cálculo de Métricas
        new_metrics = {
            "timestamp": datetime.now().isoformat(),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "model_version": "candidate"
        }

        logger.info(f"Resultados do Modelo Candidato: {json.dumps(new_metrics, indent=2)}")

        # ---------------------------------------------------------
        # 5. Quality Gate & Promoção (Requisito 7 e 10)
        # ---------------------------------------------------------
        if self._should_promote_model(new_metrics):
            self._promote_model(candidate_model, new_metrics, X_test, y_test, y_pred)
        else:
            logger.warning("ABORTANDO: O modelo candidato não superou os critérios de qualidade para promoção.")

    def _should_promote_model(self, new_metrics: Dict[str, Any]) -> bool:
        """
        Decide se o novo modelo deve substituir o antigo.
        Regra: O novo F1-Score não pode cair mais que 5% em relação ao anterior.
        """
        if not os.path.exists(Settings.METRICS_FILE):
            logger.info("Nenhuma métrica anterior encontrada (Primeiro Deploy). Aprovado.")
            return True

        try:
            with open(Settings.METRICS_FILE, "r") as f:
                current_metrics = json.load(f)

            logger.info(f"Métricas em Produção: F1={current_metrics.get('f1_score', 0)}")

            # Definição do limiar (Threshold)
            # Aceitamos uma queda de até 5% se houver ganho em outras áreas, ou exigimos melhora estrita.
            # Aqui: F1 deve ser pelo menos 95% do anterior.
            current_f1 = current_metrics.get("f1_score", 0)
            threshold = current_f1 * 0.95

            if new_metrics["f1_score"] >= threshold:
                logger.info("Quality Gate: APROVADO.")
                return True
            else:
                logger.warning(f"Quality Gate: REPROVADO (Novo F1 {new_metrics['f1_score']} < Limite {threshold:.4f})")
                return False

        except Exception as e:
            logger.warning(f"Erro ao ler métricas antigas ({e}). Forçando aprovação por segurança.")
            return True

    def _promote_model(self, model, metrics, X_test, y_test, y_pred):
        """
        Promove o modelo candidato a produção:
        1. Faz backup do modelo antigo.
        2. Salva o novo modelo (.joblib).
        3. Salva métricas (.json).
        4. Gera dados de referência para monitoramento (.csv).
        """
        logger.info("Iniciando promoção do modelo...")

        # 1. Backup (Versionamento Simples)
        if os.path.exists(Settings.MODEL_PATH):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{Settings.MODEL_PATH}.{timestamp}.bak"
            shutil.copy(Settings.MODEL_PATH, backup_path)
            logger.info(f"Backup criado: {backup_path}")

        # 2. Salvar Modelo
        os.makedirs(os.path.dirname(Settings.MODEL_PATH), exist_ok=True)
        dump(model, Settings.MODEL_PATH)
        logger.info(f"Modelo salvo em produção: {Settings.MODEL_PATH}")

        # 3. Salvar Métricas
        metrics["model_version"] = datetime.now().strftime("v%Y.%m.%d")
        os.makedirs(os.path.dirname(Settings.METRICS_FILE), exist_ok=True)
        with open(Settings.METRICS_FILE, "w") as f:
            json.dump(metrics, f)

        # 4. Salvar Reference Data (Evidently)
        # Importante: O Reference Data DEVE ser o dataset de TESTE (dados não vistos no treino)
        reference_df = X_test.copy()
        reference_df[Settings.TARGET_COL] = y_test
        reference_df["prediction"] = y_pred

        os.makedirs(os.path.dirname(Settings.REFERENCE_PATH), exist_ok=True)
        reference_df.to_csv(Settings.REFERENCE_PATH, index=False)
        logger.info(f"Reference Data atualizado para monitoramento: {Settings.REFERENCE_PATH}")


# Instância pronta para importação
trainer = MLPipeline()
