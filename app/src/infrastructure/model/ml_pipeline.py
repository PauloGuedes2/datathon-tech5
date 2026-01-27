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
        Cria a variável alvo RISCO_DEFASAGEM baseada no DESEMPENHO ATUAL.
        O Target continua olhando para o presente (é o gabarito),
        mas as Features olharão para o passado.
        """
        if "DEFASAGEM" in df.columns:
            # Prioriza a defasagem explícita (idade/série)
            df[Settings.TARGET_COL] = df["DEFASAGEM"].apply(
                lambda x: 1 if (isinstance(x, (int, float)) and x < 0) else 0
            )
        elif "INDE" in df.columns:
            # Fallback para nota baixa (Regra de Negócio: INDE < 6.0 é risco)
            df["INDE"] = pd.to_numeric(df["INDE"], errors='coerce')
            df[Settings.TARGET_COL] = (df["INDE"] < 6.0).astype(int)
        elif "PEDRA" in df.columns:
            # Fallback para classificação Pedra
            df[Settings.TARGET_COL] = df["PEDRA"].astype(str).str.upper().apply(
                lambda x: 1 if "QUARTZO" in x else 0
            )
        else:
            raise ValueError("Não foi possível criar o target: Colunas DEFASAGEM, INDE ou PEDRA ausentes.")

        return df

    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera features históricas (Lag) comparando o ano atual com o anterior do MESMO aluno.
        Isso captura a TENDÊNCIA (se o aluno está melhorando ou piorando).
        """
        logger.info("Gerando Features Históricas (Lag)...")

        # 1. Ordenação Crítica: Agrupar por aluno e ordenar por ano é vital para o .shift() funcionar
        if "NOME" not in df.columns or "ANO_REFERENCIA" not in df.columns:
            logger.warning("Colunas NOME ou ANO_REFERENCIA ausentes. Pulando criação de Lag Features.")
            return df

        df = df.sort_values(by=["NOME", "ANO_REFERENCIA"])

        # 2. Métricas que queremos rastrear o histórico
        metricas_para_historico = ["INDE", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN"]

        for col in metricas_para_historico:
            if col in df.columns:
                col_name = f"{col}_ANTERIOR"
                # Pega o valor da linha anterior (Ano T-1) do mesmo aluno
                df[col_name] = df.groupby("NOME")[col].shift(1)

                # Preenche com 0 para alunos novos (sem histórico)
                df[col_name] = df[col_name].fillna(0)

        # 3. Cria flag de 'Aluno Novo'
        # Se INDE_ANTERIOR é 0, assumimos que é o primeiro ano dele na base
        if "INDE_ANTERIOR" in df.columns:
            df["ALUNO_NOVO"] = (df["INDE_ANTERIOR"] == 0).astype(int)
        else:
            df["ALUNO_NOVO"] = 1

        return df

    def train(self, df: pd.DataFrame):
        """
        Executa o pipeline completo de treinamento com Lógica Preditiva (Anti-Leakage).
        """
        logger.info("Iniciando pipeline de treinamento Enterprise (Modo Preditivo)...")

        # ---------------------------------------------------------
        # 1. Preparação dos Dados
        # ---------------------------------------------------------

        # A. Cria o Target (Gabarito do Presente)
        df = self.create_target(df)

        # B. Cria Features do Passado (Lag)
        df = self.create_lag_features(df)

        # C. Processamento (Limpeza e Padronização)
        X_processed = self.processor.process(df)

        # ---------------------------------------------------------
        # 2. REMOÇÃO DE DATA LEAKAGE (CRUCIAL)
        # ---------------------------------------------------------
        # Removemos as colunas que compõem o Target (Notas Atuais).
        # O modelo só pode ver: Dados Demográficos + Histórico (Ano Anterior).

        # Certifique-se de ter adicionado COLUNAS_PROIBIDAS_NO_TREINO no settings.py
        cols_proibidas = getattr(Settings, "COLUNAS_PROIBIDAS_NO_TREINO", [])

        cols_to_drop = [c for c in cols_proibidas if c in X_processed.columns]
        if cols_to_drop:
            logger.info(f"Removendo colunas de vazamento (Leakage): {cols_to_drop}")
            X_processed = X_processed.drop(columns=cols_to_drop)

        # Recolocamos colunas auxiliares necessárias para o split
        X_processed[Settings.TARGET_COL] = df[Settings.TARGET_COL]

        # Precisamos do ANO_REFERENCIA original para fazer o split temporal
        if "ANO_REFERENCIA" in df.columns:
            X_processed["ANO_REFERENCIA"] = df["ANO_REFERENCIA"]
        else:
            X_processed["ANO_REFERENCIA"] = datetime.now().year

        # ---------------------------------------------------------
        # 3. Split Temporal (Evita Data Leakage do Futuro)
        # ---------------------------------------------------------
        anos_disponiveis = sorted(X_processed["ANO_REFERENCIA"].unique())

        if len(anos_disponiveis) > 1:
            # Se temos múltiplos anos, o último ano é SEMPRE teste
            ano_teste = anos_disponiveis[-1]
            logger.info(f"Split Temporal: Treino (Anos < {ano_teste}) | Teste (Ano {ano_teste})")

            mask_test = X_processed["ANO_REFERENCIA"] == ano_teste
            mask_train = ~mask_test
        else:
            logger.warning("Apenas um ano de dados disponível. Usando Split Aleatório (80/20).")
            mask_train, mask_test = train_test_split(X_processed.index, test_size=0.2, random_state=42)

        # Seleciona apenas as features permitidas na Whitelist (Settings)
        # A Whitelist agora deve conter INDE_ANTERIOR, etc.
        features_to_use = [f for f in Settings.FEATURES_NUMERICAS + Settings.FEATURES_CATEGORICAS
                           if f in X_processed.columns]

        X_train = X_processed.loc[mask_train, features_to_use]
        y_train = X_processed.loc[mask_train, Settings.TARGET_COL]

        X_test = X_processed.loc[mask_test, features_to_use]
        y_test = X_processed.loc[mask_test, Settings.TARGET_COL]

        logger.info(f"Dimensões -> Treino: {X_train.shape} | Teste: {X_test.shape}")
        logger.info(f"Taxa de Risco (Treino): {y_train.mean():.2%}")

        # ---------------------------------------------------------
        # 4. Definição do Pipeline do Modelo
        # ---------------------------------------------------------
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
        # 5. Treinamento e Avaliação
        # ---------------------------------------------------------
        candidate_model.fit(X_train, y_train)
        y_pred = candidate_model.predict(X_test)

        new_metrics = {
            "timestamp": datetime.now().isoformat(),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "model_version": "candidate"
        }

        logger.info(f"Resultados do Modelo Candidato: {json.dumps(new_metrics, indent=2)}")

        # ---------------------------------------------------------
        # 6. Quality Gate & Promoção
        # ---------------------------------------------------------
        if self._should_promote_model(new_metrics):
            self._promote_model(candidate_model, new_metrics, X_test, y_test, y_pred)
        else:
            logger.warning("ABORTANDO: O modelo candidato não superou os critérios de qualidade.")

    def _should_promote_model(self, new_metrics: Dict[str, Any]) -> bool:
        """
        Decide se o novo modelo deve substituir o antigo.
        Aceita queda de até 5% no F1-Score para permitir variação natural.
        """
        if not os.path.exists(Settings.METRICS_FILE):
            logger.info("Nenhuma métrica anterior encontrada (Primeiro Deploy). Aprovado.")
            return True

        try:
            with open(Settings.METRICS_FILE, "r") as f:
                current_metrics = json.load(f)

            current_f1 = current_metrics.get("f1_score", 0)
            threshold = current_f1 * 0.95

            logger.info(
                f"Comparando F1: Novo ({new_metrics['f1_score']}) vs Atual ({current_f1}) [Threshold: {threshold:.4f}]")

            if new_metrics["f1_score"] >= threshold:
                logger.info("Quality Gate: APROVADO.")
                return True
            else:
                logger.warning(f"Quality Gate: REPROVADO.")
                return False

        except Exception as e:
            logger.warning(f"Erro ao ler métricas antigas ({e}). Forçando aprovação.")
            return True

    def _promote_model(self, model, metrics, X_test, y_test, y_pred):
        """
        Promove o modelo a produção: Backup, Save Model, Save Metrics, Save Reference Data.
        """
        logger.info("Iniciando promoção do modelo...")

        # 1. Backup
        if os.path.exists(Settings.MODEL_PATH):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{Settings.MODEL_PATH}.{timestamp}.bak"
            shutil.copy(Settings.MODEL_PATH, backup_path)

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
        # Importante: O Reference Data deve conter as features usadas no treino (Lag features)
        # e o target real.
        reference_df = X_test.copy()
        reference_df[Settings.TARGET_COL] = y_test
        reference_df["prediction"] = y_pred

        os.makedirs(os.path.dirname(Settings.REFERENCE_PATH), exist_ok=True)
        reference_df.to_csv(Settings.REFERENCE_PATH, index=False)
        logger.info(f"Reference Data atualizado: {Settings.REFERENCE_PATH}")


# Instância pronta para importação
trainer = MLPipeline()
