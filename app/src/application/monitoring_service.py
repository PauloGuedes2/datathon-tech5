import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently import ColumnMapping

from src.config.settings import Settings
from src.util.logger import logger


class MonitoringService:
    def generate_dashboard(self) -> str:
        """
        Gera o relatório HTML de Data Drift comparando Treino vs Produção.
        Retorna: String contendo o HTML.
        """
        # 1. Verifica se existem dados suficientes
        if not os.path.exists(Settings.REFERENCE_PATH):
            return "<h1>Erro: Dataset de Referência não encontrado. Treine o modelo primeiro.</h1>"

        if not os.path.exists(Settings.LOG_PATH):
            return "<h1>Aviso: Nenhum dado de produção ainda. Faça algumas predições na API primeiro.</h1>"

        try:
            # 2. Carrega os Dados
            reference_data = pd.read_csv(Settings.REFERENCE_PATH)
            current_data = pd.read_csv(Settings.LOG_PATH)

            # Garante que temos colunas compatíveis (Interseção)
            common_cols = list(set(reference_data.columns) & set(current_data.columns))

            # Se tiver poucos dados, o Evidently pode reclamar, mas vamos tentar
            if len(current_data) < 5:
                return "<h1>Aguardando mais dados... (Mínimo 5 requisições para gerar relatório)</h1>"

            # 3. Configuração do Evidently (Column Mapping)
            # Precisamos dizer quem é categórico e quem é numérico
            column_mapping = ColumnMapping()
            column_mapping.numerical_features = [c for c in Settings.FEATURES_NUMERICAS if c in common_cols]
            column_mapping.categorical_features = [c for c in Settings.FEATURES_CATEGORICAS if c in common_cols]

            # Se salvamos o target/prediction, configuramos aqui
            if "prediction" in current_data.columns:
                column_mapping.prediction = "prediction"

            # 4. Gera o Relatório
            drift_report = Report(metrics=[
                DataDriftPreset(),  # Detecta mudança na distribuição dos dados (Drift)
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