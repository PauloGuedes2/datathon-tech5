import os

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

from src.config.settings import Settings
from src.util.logger import logger


class MonitoringService:
    def generate_dashboard(self) -> str:
        """
        Gera o relatório HTML de Data Drift comparando Treino (Reference) vs Produção (Current).
        Retorna: String contendo o HTML.
        """
        # 1. Verificações Iniciais
        if not os.path.exists(Settings.REFERENCE_PATH):
            return "<h1>Erro: Dataset de Referência não encontrado. Treine o modelo primeiro.</h1>"

        if not os.path.exists(Settings.LOG_PATH):
            return "<h1>Aviso: Nenhum dado de produção ainda. Faça algumas predições na API primeiro.</h1>"

        try:
            # 2. Carrega os Dados de Referência (CSV gerado no treino)
            reference_data = pd.read_csv(Settings.REFERENCE_PATH)

            # 3. Carrega e Processa os Dados de Produção (JSONL gerado na inferência)
            # Lê o arquivo JSON Lines
            try:
                current_data_raw = pd.read_json(Settings.LOG_PATH, lines=True)
            except ValueError:
                return "<h1>Aviso: Arquivo de logs vazio ou inválido.</h1>"

            if current_data_raw.empty:
                return "<h1>Aviso: Arquivo de logs sem dados.</h1>"

            # O JSON tem estrutura aninhada:
            # { "input_features": {...}, "prediction_result": {...}, ... }
            # Precisamos transformar isso numa tabela plana (features + predição)

            # Normaliza (aplaina) as features
            features_df = pd.json_normalize(current_data_raw["input_features"])

            # Normaliza (aplaina) os resultados da predição
            preds_df = pd.json_normalize(current_data_raw["prediction_result"])

            # Concatena lateralmente para formar um único DataFrame plano
            current_data = pd.concat([features_df, preds_df], axis=1)

            # Renomeia a coluna 'class' para 'prediction' para bater com o dataset de referência
            # (No JSON salvamos como 'class', mas o Reference Data usa 'prediction')
            if "class" in current_data.columns:
                current_data.rename(columns={"class": "prediction"}, inplace=True)

            # Remove registros inválidos ou vazios
            reference_data = reference_data.dropna(subset=["prediction"])
            current_data = current_data.dropna(subset=["prediction"])

            # Garante que temos colunas compatíveis (Interseção)
            common_cols = list(set(reference_data.columns) & set(current_data.columns))

            # Validação de volume mínimo de dados
            if len(current_data) < 5:
                return "<h1>Aguardando mais dados... (Mínimo 5 requisições para gerar relatório confiável)</h1>"

            # 4. Configuração do Evidently (Column Mapping)
            column_mapping = ColumnMapping()

            # Define dinamicamente quais colunas numéricas/categóricas existem em ambos os datasets
            column_mapping.numerical_features = [c for c in Settings.FEATURES_NUMERICAS if c in common_cols]
            column_mapping.categorical_features = [c for c in Settings.FEATURES_CATEGORICAS if c in common_cols]

            if "prediction" in current_data.columns:
                column_mapping.prediction = "prediction"

            # 5. Gera o Relatório
            drift_report = Report(metrics=[
                DataDriftPreset(),  # Drift nas features (Input)
                TargetDriftPreset()  # Drift na predição (Output)
            ])

            drift_report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )

            # Retorna o HTML como string
            return drift_report.get_html()

        except Exception as e:
            logger.error(f"Erro ao gerar dashboard: {e}")
            return f"<h1>Erro interno ao gerar relatório: {str(e)}</h1>"
