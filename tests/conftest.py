import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


@pytest.fixture()
def sample_student_dict():
    return {
        "RA": "123",
        "IDADE": 10,
        "ANO_INGRESSO": 2020,
        "GENERO": "Masculino",
        "TURMA": "A",
        "INSTITUICAO_ENSINO": "Escola",
        "FASE": "1A",
        "NOME": "Aluno",
        "INDE_ANTERIOR": 5.0,
        "IAA_ANTERIOR": 1.0,
        "IEG_ANTERIOR": 2.0,
        "IPS_ANTERIOR": 3.0,
        "IDA_ANTERIOR": 4.0,
        "IPP_ANTERIOR": 5.0,
        "IPV_ANTERIOR": 6.0,
        "IAN_ANTERIOR": 7.0,
        "ALUNO_NOVO": 0,
    }


@pytest.fixture()
def sample_student_input():
    return {
        "RA": "123",
        "IDADE": 10,
        "ANO_INGRESSO": 2020,
        "GENERO": "Masculino",
        "TURMA": "A",
        "INSTITUICAO_ENSINO": "Escola",
        "FASE": "1A",
    }


@pytest.fixture()
def base_dataframe():
    return pd.DataFrame(
        [
            {
                "RA": "1",
                "ANO_REFERENCIA": 2023,
                "ANO_INGRESSO": 2021,
                "INDE": 5.0,
                "IAA": 1.0,
                "IEG": 2.0,
                "IPS": 3.0,
                "IDA": 4.0,
                "IPP": 5.0,
                "IPV": 6.0,
                "IAN": 7.0,
                "GENERO": "Masculino",
                "TURMA": "A",
                "INSTITUICAO_ENSINO": "Escola",
                "FASE": "1A",
                "IDADE": 10,
            }
        ]
    )
