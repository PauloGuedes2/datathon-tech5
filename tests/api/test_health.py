class TestHealthEndpoint:
    """
    Classe de testes para o endpoint de health check.
    
    Testa:
        - Disponibilidade do serviço
        - Formato da resposta
        - Status codes corretos
    """

    def test_health_check_success(self, client):
        """
        Testa se o health check retorna status correto.
        
        Args:
            client: Cliente de teste FastAPI
            
        Verifica:
            - Status code 200
            - Estrutura da resposta
            - Valores esperados
        """
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "service": "passos-magicos-api"
        }

    def test_health_check_response_format(self, client):
        """
        Testa formato detalhado da resposta do health check.
        
        Args:
            client: Cliente de teste FastAPI
            
        Verifica:
            - Presença de campos obrigatórios
            - Tipos de dados corretos
        """
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "service" in data
        assert isinstance(data["status"], str)
        assert isinstance(data["service"], str)
        assert data["status"] == "ok"
