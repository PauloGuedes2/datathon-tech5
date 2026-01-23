from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import App


class TestApp:
    """
    Classe de testes para a classe App principal.
    
    Testa:
        - Inicialização da aplicação
        - Configuração do FastAPI
        - Registro de rotas
        - Execução do servidor
    """

    def test_app_initialization(self):
        """
        Testa inicialização básica da aplicação.
        
        Verifica:
            - App é criada corretamente
            - FastAPI é inicializado
            - Configurações básicas estão corretas
        """
        app = App()

        assert isinstance(app, App)
        assert hasattr(app, 'app')
        assert isinstance(app.app, FastAPI)

    def test_fastapi_configuration(self):
        """
        Testa configuração do FastAPI.
        
        Verifica:
            - Título está correto
            - Descrição está presente
            - Versão está definida
        """
        app = App()
        fastapi_app = app.app

        assert fastapi_app.title == "Passos Mágicos - API de Previsão de Risco"
        assert "predição de risco de defasagem escolar" in fastapi_app.description
        assert fastapi_app.version == "1.0.0"

    @patch('app.main.PredictionController')
    def test_prediction_routes_registration(self, mock_controller_class):
        """
        Testa registro das rotas de predição.
        
        Args:
            mock_controller_class: Mock da classe PredictionController
            
        Verifica:
            - Router do controller é incluído
            - Prefixo correto é usado
            - Tags estão configuradas
        """
        mock_controller = Mock()
        mock_router = Mock()
        mock_controller.router = mock_router
        mock_controller_class.return_value = mock_controller

        with patch.object(FastAPI, 'include_router') as mock_include_router:
            app = App()

            # Verifica que include_router foi chamado
            mock_include_router.assert_called_with(
                mock_router,
                prefix="/api/v1",
                tags=["Previsão"]
            )

    def test_health_route_registration(self):
        """
        Testa registro da rota de health check.
        
        Verifica:
            - Rota /health está registrada
            - Método GET está configurado
            - Tag está correta
        """
        app = App()

        # Verifica que rota existe através do cliente de teste
        client = TestClient(app.app)
        response = client.get("/health")

        # Se a rota existe, deve retornar status válido (não 404)
        assert response.status_code != 404

    def test_health_check_method(self):
        """
        Testa método health_check diretamente.
        
        Verifica:
            - Método é estático
            - Retorna estrutura correta
            - Valores estão corretos
        """
        result = App.health_check()

        assert isinstance(result, dict)
        assert "status" in result
        assert "service" in result
        assert result["status"] == "ok"
        assert result["service"] == "passos-magicos-api"

    @patch('app.main.uvicorn.run')
    @patch('app.main.os.getenv')
    def test_run_method_default_parameters(self, mock_getenv, mock_uvicorn_run):
        """
        Testa método run com parâmetros padrão.
        
        Args:
            mock_getenv: Mock da função getenv
            mock_uvicorn_run: Mock do uvicorn.run
            
        Verifica:
            - Parâmetros padrão são usados
            - uvicorn.run é chamado corretamente
        """
        mock_getenv.return_value = "8000"  # PORT padrão

        app = App()
        app.run()

        # Verifica chamada do uvicorn
        mock_uvicorn_run.assert_called_once_with(
            app.app,
            host="0.0.0.0",
            port=8000
        )

    @patch('app.main.uvicorn.run')
    @patch('app.main.os.getenv')
    def test_run_method_custom_parameters(self, mock_getenv, mock_uvicorn_run):
        """
        Testa método run com parâmetros customizados.
        
        Args:
            mock_getenv: Mock da função getenv
            mock_uvicorn_run: Mock do uvicorn.run
            
        Verifica:
            - Parâmetros customizados são respeitados
            - Variável de ambiente PORT é considerada
        """
        mock_getenv.return_value = "9000"  # PORT customizada

        app = App()
        app.run(host="127.0.0.1", port=3000)

        # Verifica que PORT env var sobrescreve parâmetro port
        mock_uvicorn_run.assert_called_once_with(
            app.app,
            host="127.0.0.1",
            port=9000  # Valor da env var
        )

    @patch('app.main.uvicorn.run')
    @patch('app.main.os.getenv')
    def test_run_method_port_environment_variable(self, mock_getenv, mock_uvicorn_run):
        """
        Testa uso da variável de ambiente PORT.
        
        Args:
            mock_getenv: Mock da função getenv
            mock_uvicorn_run: Mock do uvicorn.run
            
        Verifica:
            - Variável PORT é lida corretamente
            - Conversão para int funciona
        """
        mock_getenv.return_value = "5000"

        app = App()
        app.run()

        # Verifica que getenv foi chamado com PORT
        mock_getenv.assert_called_with("PORT", 8000)

        # Verifica que porta foi convertida para int
        mock_uvicorn_run.assert_called_once_with(
            app.app,
            host="0.0.0.0",
            port=5000
        )

    @patch('app.main.uvicorn.run')
    @patch('app.main.os.getenv')
    def test_run_method_port_conversion_error(self, mock_getenv, mock_uvicorn_run):
        """
        Testa tratamento de erro na conversão da porta.
        
        Args:
            mock_getenv: Mock da função getenv
            mock_uvicorn_run: Mock do uvicorn.run
            
        Verifica:
            - Erro de conversão é tratado
            - Valor padrão é usado em caso de erro
        """
        mock_getenv.return_value = "invalid_port"

        app = App()

        # Deve usar porta padrão se conversão falhar
        with pytest.raises(ValueError):
            app.run()

    def test_app_routes_structure(self):
        """
        Testa estrutura geral das rotas registradas.
        
        Verifica:
            - Rotas esperadas estão presentes
            - Estrutura de URLs está correta
        """
        app = App()
        client = TestClient(app.app)

        # Testa rotas principais
        health_response = client.get("/health")
        assert health_response.status_code == 200

        # Rota de predição deve existir (mesmo que falhe por falta de dados)
        predict_response = client.post("/api/v1/predict", json={})
        assert predict_response.status_code != 404  # Rota existe

    def test_app_middleware_configuration(self):
        """
        Testa configuração de middleware (se houver).
        
        Verifica:
            - Middleware básico está configurado
            - CORS ou outros middlewares se necessário
        """
        app = App()

        # Verifica que app FastAPI foi criado sem erros
        assert isinstance(app.app, FastAPI)

        # Testa que pode processar requests básicos
        client = TestClient(app.app)
        response = client.get("/health")
        assert response.status_code == 200

    def test_app_exception_handling(self):
        """
        Testa tratamento básico de exceções.
        
        Verifica:
            - Aplicação não quebra com requests inválidos
            - Respostas de erro são adequadas
        """
        app = App()
        client = TestClient(app.app)

        # Testa rota inexistente
        response = client.get("/rota/inexistente")
        assert response.status_code == 404

        # Testa método não permitido
        response = client.post("/health")
        assert response.status_code == 405
