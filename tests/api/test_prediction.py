from unittest.mock import patch, Mock


class TestPredictionEndpoint:
    """
    Classe de testes para o endpoint de predição.
    
    Testa:
        - Predições bem-sucedidas
        - Validação de entrada
        - Tratamento de erros
        - Formatos de resposta
    """

    def test_predict_success(self, client, sample_student_data):
        """
        Testa predição bem-sucedida com dados válidos.
        
        Args:
            client: Cliente de teste FastAPI
            sample_student_data: Dados válidos de estudante
            
        Verifica:
            - Status code 200
            - Estrutura da resposta
            - Tipos de dados corretos
            - Valores dentro dos ranges esperados
        """
        with patch('src.application.risk_service.RiskService') as mock_service:
            mock_instance = Mock()
            mock_instance.predict_risk.return_value = {
                "risk_probability": 0.3245,
                "risk_label": "BAIXO RISCO",
                "message": "O estudante possui 32.5% de chance de defasagem."
            }
            mock_service.return_value = mock_instance

            response = client.post("/api/v1/predict", json=sample_student_data)

            assert response.status_code == 200

            body = response.json()

            assert "risk_probability" in body
            assert "risk_label" in body
            assert "message" in body

            assert 0.0 <= body["risk_probability"] <= 1.0
            assert body["risk_label"] in ["ALTO RISCO", "BAIXO RISCO"]
            assert isinstance(body["message"], str)

    def test_predict_missing_field(self, client):
        """
        Testa validação quando campos obrigatórios estão ausentes.
        
        Args:
            client: Cliente de teste FastAPI
            
        Verifica:
            - Status code 422 (Validation Error)
            - Rejeição de dados incompletos
        """
        payload = {
            "IDADE_22": 14,
            "CG": 7.5
            # Campos obrigatórios ausentes
        }

        response = client.post("/api/v1/predict", json=payload)

        assert response.status_code == 422  # Validation error

    def test_predict_invalid_type(self, client):
        """
        Testa validação com tipos de dados inválidos.
        
        Args:
            client: Cliente de teste FastAPI
            
        Verifica:
            - Status code 422 (Validation Error)
            - Rejeição de tipos incorretos
        """
        payload = {
            "IDADE_22": "quatorze",  # inválido - deve ser int
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

        response = client.post("/api/v1/predict", json=payload)

        assert response.status_code == 422

    def test_predict_empty_payload(self, client):
        """
        Testa comportamento com payload vazio.
        
        Args:
            client: Cliente de teste FastAPI
            
        Verifica:
            - Status code 422
            - Rejeição de dados vazios
        """
        response = client.post("/api/v1/predict", json={})
        assert response.status_code == 422

    def test_predict_null_values(self, client):
        """
        Testa comportamento com valores nulos.
        
        Args:
            client: Cliente de teste FastAPI
            
        Verifica:
            - Status code 422
            - Rejeição de valores nulos
        """
        payload = {
            "IDADE_22": None,
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

        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422
