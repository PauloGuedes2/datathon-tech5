from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException

from src.api.controller import PredictionController, get_risk_service
from src.api.schemas import StudentDTO


class TestPredictionController:
    """
    Classe de testes para PredictionController.
    
    Testa:
        - Inicialização do controller
        - Registro de rotas
        - Método de predição
        - Tratamento de erros
    """

    def test_controller_initialization(self):
        """
        Testa inicialização do controller.
        
        Verifica:
            - Controller é criado corretamente
            - Router é inicializado
            - Rotas são registradas
        """
        controller = PredictionController()

        assert controller.router is not None
        assert len(controller.router.routes) > 0

    def test_register_routes(self):
        """
        Testa registro de rotas no controller.
        
        Verifica:
            - Rota /predict está registrada
            - Método POST está configurado
            - Response model está definido
        """
        controller = PredictionController()

        # Verifica se a rota foi registrada
        routes = [route.path for route in controller.router.routes]
        assert "/predict" in routes

        # Verifica método HTTP
        predict_route = next(route for route in controller.router.routes if route.path == "/predict")
        assert "POST" in predict_route.methods

    @patch('src.api.controller.Student')
    def test_predict_success(self, mock_student_class, sample_student_data):
        """
        Testa método predict com sucesso.
        
        Args:
            mock_student_class: Mock da classe Student
            sample_student_data: Dados válidos de estudante
            
        Verifica:
            - Criação correta do Student
            - Chamada do serviço
            - Retorno da predição
        """
        # Setup mocks
        mock_student_instance = Mock()
        mock_student_class.return_value = mock_student_instance

        mock_service = Mock()
        mock_service.predict_risk.return_value = {
            "risk_probability": 0.3245,
            "risk_label": "BAIXO RISCO",
            "message": "O estudante possui 32.5% de chance de defasagem."
        }

        # Criar DTO
        student_dto = StudentDTO(**sample_student_data)

        # Executar método
        result = PredictionController.predict(student_dto, mock_service)

        # Verificações
        mock_student_class.assert_called_once()
        mock_service.predict_risk.assert_called_once_with(mock_student_instance)

        assert "risk_probability" in result
        assert "risk_label" in result
        assert "message" in result

    @patch('src.api.controller.Student')
    def test_predict_service_exception(self, mock_student_class, sample_student_data):
        """
        Testa tratamento de exceção no serviço.
        
        Args:
            mock_student_class: Mock da classe Student
            sample_student_data: Dados de estudante
            
        Verifica:
            - HTTPException é levantada
            - Status code 500
            - Mensagem de erro correta
        """
        # Setup mocks
        mock_student_instance = Mock()
        mock_student_class.return_value = mock_student_instance

        mock_service = Mock()
        mock_service.predict_risk.side_effect = Exception("Erro no modelo")

        # Criar DTO
        student_dto = StudentDTO(**sample_student_data)

        # Verificar exceção
        with pytest.raises(HTTPException) as exc_info:
            PredictionController.predict(student_dto, mock_service)

        assert exc_info.value.status_code == 500
        assert "Erro no modelo" in str(exc_info.value.detail)

    @patch('src.api.controller.Student')
    def test_predict_student_creation_error(self, mock_student_class, sample_student_data):
        """
        Testa erro na criação do objeto Student.
        
        Args:
            mock_student_class: Mock da classe Student
            sample_student_data: Dados de estudante
            
        Verifica:
            - HTTPException é levantada
            - Erro é propagado corretamente
        """
        # Setup mock para falhar
        mock_student_class.side_effect = ValueError("Dados inválidos")

        mock_service = Mock()
        student_dto = StudentDTO(**sample_student_data)

        # Verificar exceção
        with pytest.raises(HTTPException) as exc_info:
            PredictionController.predict(student_dto, mock_service)

        assert exc_info.value.status_code == 500
        assert "Dados inválidos" in str(exc_info.value.detail)


class TestDependencyInjection:
    """
    Classe de testes para injeção de dependência.
    
    Testa:
        - Factory function get_risk_service
        - Criação de instâncias
        - Configuração de dependências
    """

    @patch('src.api.controller.RiskService')
    def test_get_risk_service(self, mock_risk_service_class):
        """
        Testa factory function get_risk_service.
        
        Args:
            mock_risk_service_class: Mock da classe RiskService
            
        Verifica:
            - RiskService é instanciado
            - Instância é retornada
        """
        mock_instance = Mock()
        mock_risk_service_class.return_value = mock_instance

        result = get_risk_service()

        mock_risk_service_class.assert_called_once()
        assert result == mock_instance

    @patch('src.api.controller.RiskService')
    def test_get_risk_service_exception(self, mock_risk_service_class):
        """
        Testa comportamento quando RiskService falha na criação.
        
        Args:
            mock_risk_service_class: Mock da classe RiskService
            
        Verifica:
            - Exceção é propagada
            - Erro é tratado adequadamente
        """
        mock_risk_service_class.side_effect = Exception("Erro na inicialização")

        with pytest.raises(Exception) as exc_info:
            get_risk_service()

        assert "Erro na inicialização" in str(exc_info.value)
