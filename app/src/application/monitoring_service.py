import os

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

from src.config.settings import Settings
from src.util.logger import logger

""" Módulo de Serviço para Monitoramento de Data Drift e Target Drift."""


class MonitoringService:

    @staticmethod
    def generate_dashboard() -> str:
        """
        Gera o relatório HTML de Data Drift comparando Treino (Reference) vs Produção (Current).
        Retorna: String contendo o HTML.
        """
        if not os.path.exists(Settings.REFERENCE_PATH):
            return "<h1>Erro: Dataset de Referência não encontrado. Treine o modelo primeiro.</h1>"

        if not os.path.exists(Settings.LOG_PATH):
            return "<h1>Aviso: Nenhum dado de produção ainda. Faça algumas predições na API primeiro.</h1>"

        try:
            reference_data = pd.read_csv(Settings.REFERENCE_PATH)
            try:
                current_data_raw = pd.read_json(Settings.LOG_PATH, lines=True)
            except ValueError:
                return "<h1>Aviso: Arquivo de logs vazio ou inválido.</h1>"

            if current_data_raw.empty:
                return "<h1>Aviso: Arquivo de logs sem dados.</h1>"

            features_df = pd.json_normalize(current_data_raw["input_features"])
            preds_df = pd.json_normalize(current_data_raw["prediction_result"])
            current_data = pd.concat([features_df, preds_df], axis=1)

            if "class" in current_data.columns:
                current_data.rename(columns={"class": "prediction"}, inplace=True)

            reference_data = reference_data.dropna(subset=["prediction"])
            current_data = current_data.dropna(subset=["prediction"])

            common_cols = list(set(reference_data.columns) & set(current_data.columns))

            if len(current_data) < 5:
                return "<h1>Aguardando mais dados... (Mínimo 5 requisições para gerar relatório confiável)</h1>"

            column_mapping = ColumnMapping()

            column_mapping.numerical_features = [c for c in Settings.FEATURES_NUMERICAS if c in common_cols]
            column_mapping.categorical_features = [c for c in Settings.FEATURES_CATEGORICAS if c in common_cols]

            if "prediction" in current_data.columns:
                column_mapping.prediction = "prediction"

            drift_report = Report(metrics=[
                DataDriftPreset(),
                TargetDriftPreset()
            ])

            drift_report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )

            return drift_report.get_html()

        except Exception as e:
            logger.error(f"Erro ao gerar dashboard: {e}")
            return f"<h1>Erro interno ao gerar relatório: {str(e)}</h1>"
