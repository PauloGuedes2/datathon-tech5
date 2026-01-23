import pytest
from fastapi.testclient import TestClient
import pandas as pd
import tempfile
import os
from unittest.mock import Mock

from main import App


@pytest.fixture(scope="session")
def client():
    """
    Fixture que fornece cliente de teste para a API FastAPI.
    
    Returns:
        TestClient configurado com a aplicação FastAPI
        
    Scope:
        session - Reutilizado em toda a sessão de testes
    """
    app = App().app
    return TestClient(app)


@pytest.fixture
def sample_dataframe():
    """
    Fixture que fornece DataFrame de exemplo para testes.
    
    Returns:
        DataFrame com dados de teste estruturados
    """
    return pd.DataFrame({
        'IDADE_22': [14, 15, 16],
        'CG': [7.5, 6.8, 8.2],
        'CF': [7.0, 6.5, 7.8],
        'CT': [7.2, 6.9, 8.0],
        'IAA': [6.8, 6.2, 7.5],
        'IEG': [7.1, 6.7, 7.9],
        'IPS': [6.9, 6.4, 7.6],
        'IDA': [7.0, 6.6, 7.7],
        'MATEM': [6.5, 6.0, 7.3],
        'PORTUG': [7.3, 6.8, 8.1],
        'INGLES': [6.8, 6.3, 7.4],
        'GENERO': ['M', 'F', 'M'],
        'TURMA': ['A', 'B', 'A'],
        'INSTITUICAO_DE_ENSINO': ['ESCOLA MUNICIPAL', 'ESCOLA ESTADUAL', 'ESCOLA MUNICIPAL'],
        'DEFAS': [-1, 0, 2]
    })


@pytest.fixture
def sample_student_data():
    """
    Fixture que fornece dados de estudante para testes.
    
    Returns:
        Dict com dados válidos de estudante
    """
    return {
        "IDADE_22": 14,
        "CG": 7.5,
        "CF": 7.0,
        "CT": 7.2,
        "IAA": 6.8,
        "IEG": 7.1,
        "IPS": 6.9,
        "IDA": 7.0,
        "MATEM": 6.5,
        "PORTUG": 7.3,
        "INGLES": 6.8,
        "GENERO": "M",
        "TURMA": "A",
        "INSTITUICAO_DE_ENSINO": "ESCOLA MUNICIPAL"
    }


@pytest.fixture
def temp_excel_file(sample_dataframe):
    """
    Fixture que cria arquivo Excel temporário para testes.
    
    Args:
        sample_dataframe: DataFrame para salvar no arquivo
        
    Returns:
        Caminho do arquivo Excel temporário
    """
    # Cria arquivo temporário
    fd, temp_path = tempfile.mkstemp(suffix='.xlsx')
    
    try:
        # Fecha o file descriptor para permitir que pandas escreva
        os.close(fd)
        
        # Salva o DataFrame
        sample_dataframe.to_excel(temp_path, index=False)
        
        yield temp_path
        
    finally:
        # Tenta remover o arquivo com retry para Windows
        import time
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                break
            except PermissionError:
                if attempt < max_attempts - 1:
                    time.sleep(0.1)  # Aguarda um pouco antes de tentar novamente
                else:
                    # Se ainda não conseguir, apenas avisa (não falha o teste)
                    import warnings
                    warnings.warn(f"Não foi possível remover arquivo temporário: {temp_path}")
            except FileNotFoundError:
                # Arquivo já foi removido, tudo bem
                break


@pytest.fixture
def mock_model():
    """
    Fixture que fornece modelo mockado para testes.
    
    Returns:
        Mock object configurado como modelo sklearn
    """
    import numpy as np
    
    model = Mock()
    # Mock predict_proba retorna array numpy 2D
    model.predict_proba.return_value = np.array([[0.7, 0.3]])
    
    # Mock para feature importance com tamanhos corretos
    mock_clf = Mock()
    # 11 features numéricas + 3 categóricas = 14 features
    mock_clf.feature_importances_ = np.array([0.1, 0.2, 0.15, 0.05, 0.1, 0.08, 0.12, 0.06, 0.04, 0.07, 0.03, 0.02, 0.01, 0.005])
    
    model.named_steps = {
        'clf': mock_clf
    }
    return model
