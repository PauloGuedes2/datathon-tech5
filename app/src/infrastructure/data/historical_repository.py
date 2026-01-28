import pandas as pd
from src.config.settings import Settings
from src.util.logger import logger


class HistoricalRepository:
    _instance = None
    _data = None

    def __new__(cls):
        # Singleton para carregar os dados apenas uma vez na memória
        if cls._instance is None:
            cls._instance = super(HistoricalRepository, cls).__new__(cls)
            cls._instance._load_data()
        return cls._instance

    def _load_data(self):
        """Carrega o dataset de referência (o mesmo usado no treino ou o arquivo raw)"""
        try:
            # Idealmente, use o arquivo processado ou o raw unificado.
            # Aqui assumo que o reference_data.csv tem as colunas necessárias.
            # Se não tiver, aponte para o Excel original processado.
            logger.info("Carregando base histórica para Feature Store...")

            # Exemplo: Carregando o CSV gerado pelo pipeline de treino
            if pd.io.common.file_exists(Settings.REFERENCE_PATH):
                self._data = pd.read_csv(Settings.REFERENCE_PATH)
            else:
                # Fallback: Tenta carregar do dataloader se o CSV não existir
                from src.infrastructure.data.data_loader import DataLoader
                self._data = DataLoader().load_data()

            if 'RA' not in self._data.columns:
                # Fallback ou erro se não tiver RA no histórico
                logger.warning("Coluna RA não encontrada no histórico! A busca smart falhará.")
                return

            self._data['RA'] = self._data['RA'].astype(str).str.strip()
            self._data = self._data.sort_values(by=['RA', 'ANO_REFERENCIA'])

            logger.info(f"Feature Store carregada com {len(self._data)} registros.")
        except Exception as e:
            logger.error(f"Erro ao carregar Feature Store: {e}")
            self._data = pd.DataFrame()

    def get_student_history(self, student_ra: str) -> dict:
        """
        Busca as métricas do ano anterior para um aluno.
        Retorna um dicionário com os valores ou 0.0 se for aluno novo.
        """
        if self._data.empty:
            return {}

        ra_target = str(student_ra).strip()

        # Filtra pelo aluno
        student_history = self._data[self._data['RA'] == ra_target]

        if student_history.empty:
            return None  # Aluno não encontrado (Novo na ONG)

        # Pega o registro mais recente (assumindo que é o do ano anterior)
        # Atenção: Aqui você pode refinar para garantir que é o ano T-1
        last_record = student_history.iloc[-1]

        # Mapeia as colunas do dataset para os campos esperados pelo modelo (sufixo _ANTERIOR)
        return {
            "INDE_ANTERIOR": float(last_record.get("INDE", 0.0)),
            "IAA_ANTERIOR": float(last_record.get("IAA", 0.0)),
            "IEG_ANTERIOR": float(last_record.get("IEG", 0.0)),
            "IPS_ANTERIOR": float(last_record.get("IPS", 0.0)),
            "IDA_ANTERIOR": float(last_record.get("IDA", 0.0)),
            "IPP_ANTERIOR": float(last_record.get("IPP", 0.0)),
            "IPV_ANTERIOR": float(last_record.get("IPV", 0.0)),
            "IAN_ANTERIOR": float(last_record.get("IAN", 0.0)),
            "ALUNO_NOVO": 0  # Se achou histórico, não é novo
        }