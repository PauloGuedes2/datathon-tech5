import os
import pandas as pd
from src.config.settings import Settings
from src.util.logger import logger


class HistoricalRepository:
    _instance = None
    _data = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HistoricalRepository, cls).__new__(cls)
            cls._instance._load_data()
        return cls._instance

    def _load_data(self):
        """Carrega o dataset de referência (o mesmo usado no treino ou o arquivo raw)"""
        try:
            logger.info("Carregando base histórica para Feature Store...")
            # Tenta carregar do CSV de referência
            if os.path.exists(Settings.REFERENCE_PATH):
                self._data = pd.read_csv(Settings.REFERENCE_PATH)

                # Validação: Se o CSV estiver obsoleto (sem RA), recarrega do Excel
                if 'RA' not in self._data.columns:
                    logger.warning("CSV de referência obsoleto (sem RA). Recarregando do Excel...")
                    from src.infrastructure.data.data_loader import DataLoader
                    self._data = DataLoader().load_data()
            else:
                from src.infrastructure.data.data_loader import DataLoader
                self._data = DataLoader().load_data()

            if 'RA' not in self._data.columns:
                logger.warning("Coluna RA não encontrada no histórico! A busca smart falhará.")
                return

            # Normaliza RA para texto
            self._data['RA'] = self._data['RA'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
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
        if self._data is None or self._data.empty:
            return {}

        ra_target = str(student_ra).strip()

        # Filtra pelo aluno
        student_history = self._data[self._data['RA'] == ra_target]

        if student_history.empty:
            return None  # Aluno não encontrado (Novo na ONG)

        # Pega o registro mais recente
        last_record = student_history.iloc[-1]

        # --- FUNÇÃO AUXILIAR DE SEGURANÇA ---
        def _safe_get(col_name):
            val = last_record.get(col_name, 0.0)
            try:
                # Tenta converter para float
                return float(val)
            except (ValueError, TypeError):
                # Se for texto ("INCLUIR", "N/A"), retorna 0.0
                return 0.0

        # ------------------------------------

        return {
            "INDE_ANTERIOR": _safe_get("INDE"),
            "IAA_ANTERIOR": _safe_get("IAA"),
            "IEG_ANTERIOR": _safe_get("IEG"),
            "IPS_ANTERIOR": _safe_get("IPS"),
            "IDA_ANTERIOR": _safe_get("IDA"),
            "IPP_ANTERIOR": _safe_get("IPP"),
            "IPV_ANTERIOR": _safe_get("IPV"),
            "IAN_ANTERIOR": _safe_get("IAN"),
            "ALUNO_NOVO": 0
        }