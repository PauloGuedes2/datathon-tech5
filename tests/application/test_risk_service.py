from unittest.mock import Mock, patch

import pytest

from src.application.risk_service import RiskService
from src.domain.student import Student


class TestRiskService:
    """
    Classe de testes para RiskService.
    
    Testa:
        - Inicialização do serviço
        - Predição de risco
        - Classificação de risco
        - Integração com ML pipeline
    """

    @patch('src.application.risk_service.MLPipeline')
    def test_service_initialization(self, mock_ml_pipeline_class):
        """
        Testa inicialização do serviço.
        
        Args:
            mock_ml_pipeline_class: Mock da classe MLPipeline
            
        Verifica:
            - MLPipeline é instanciado
            - Método load é chamado
            - Serviço é inicializado corretamente
        """
        mock_pipeline = Mock()
        mock_ml_pipeline_class.return_value = mock_pipeline

        service = RiskService()

        mock_ml_pipeline_class.assert_called_once()
        mock_pipeline.load.assert_called_once()
        assert service.ml_pipeline == mock_pipeline

    @patch('src.application.risk_service.MLPipeline')
    def test_predict_risk_low_risk(self, mock_ml_pipeline_class, sample_student_data):
        """
        Testa predição de baixo risco.
        
        Args:
            mock_ml_pipeline_class: Mock da classe MLPipeline
            sample_student_data: Dados de estudante
            
        Verifica:
            - Probabilidade baixa resulta em BAIXO RISCO
            - Formato da resposta
            - Valores corretos
        """
        # Setup mock
        mock_pipeline = Mock()
        mock_pipeline.predict_proba.return_value = 0.3245  # Baixo risco
        mock_ml_pipeline_class.return_value = mock_pipeline

        service = RiskService()
        student = Student(data=sample_student_data)

        result = service.predict_risk(student)

        assert result["risk_probability"] == 0.3245
        assert result["risk_label"] == "BAIXO RISCO"
        assert "32.5%" in result["message"]
        assert "defasagem" in result["message"].lower()

    @patch('src.application.risk_service.MLPipeline')
    def test_predict_risk_high_risk(self, mock_ml_pipeline_class, sample_student_data):
        """
        Testa predição de alto risco.
        
        Args:
            mock_ml_pipeline_class: Mock da classe MLPipeline
            sample_student_data: Dados de estudante
            
        Verifica:
            - Probabilidade alta resulta em ALTO RISCO
            - Formato da resposta
            - Valores corretos
        """
        # Setup mock
        mock_pipeline = Mock()
        mock_pipeline.predict_proba.return_value = 0.7850  # Alto risco
        mock_ml_pipeline_class.return_value = mock_pipeline

        service = RiskService()
        student = Student(data=sample_student_data)

        result = service.predict_risk(student)

        assert result["risk_probability"] == 0.7850
        assert result["risk_label"] == "ALTO RISCO"
        assert "78.5%" in result["message"]

    @patch('src.application.risk_service.MLPipeline')
    def test_predict_risk_threshold_boundary(self, mock_ml_pipeline_class, sample_student_data):
        """
        Testa comportamento no limite do threshold.
        
        Args:
            mock_ml_pipeline_class: Mock da classe MLPipeline
            sample_student_data: Dados de estudante
            
        Verifica:
            - Valor exatamente no threshold (0.5)
            - Classificação correta
        """
        # Setup mock - exatamente no threshold
        mock_pipeline = Mock()
        mock_pipeline.predict_proba.return_value = 0.5000
        mock_ml_pipeline_class.return_value = mock_pipeline

        service = RiskService()
        student = Student(data=sample_student_data)

        result = service.predict_risk(student)

        assert result["risk_probability"] == 0.5000
        # Threshold é > 0.5, então 0.5 deve ser BAIXO RISCO
        assert result["risk_label"] == "BAIXO RISCO"

    @patch('src.application.risk_service.MLPipeline')
    def test_predict_risk_just_above_threshold(self, mock_ml_pipeline_class, sample_student_data):
        """
        Testa comportamento logo acima do threshold.
        
        Args:
            mock_ml_pipeline_class: Mock da classe MLPipeline
            sample_student_data: Dados de estudante
            
        Verifica:
            - Valor ligeiramente acima do threshold
            - Classificação como ALTO RISCO
        """
        # Setup mock - ligeiramente acima do threshold
        mock_pipeline = Mock()
        mock_pipeline.predict_proba.return_value = 0.5001
        mock_ml_pipeline_class.return_value = mock_pipeline

        service = RiskService()
        student = Student(data=sample_student_data)

        result = service.predict_risk(student)

        assert result["risk_probability"] == 0.5001
        assert result["risk_label"] == "ALTO RISCO"

    @patch('src.application.risk_service.MLPipeline')
    @patch('src.application.risk_service.pd.DataFrame')
    def test_dataframe_creation(self, mock_dataframe_class, mock_ml_pipeline_class, sample_student_data):
        """
        Testa criação do DataFrame para predição.
        
        Args:
            mock_dataframe_class: Mock da classe DataFrame
            mock_ml_pipeline_class: Mock da classe MLPipeline
            sample_student_data: Dados de estudante
            
        Verifica:
            - DataFrame é criado com dados corretos
            - Estrutura adequada para o modelo
        """
        # Setup mocks
        mock_pipeline = Mock()
        mock_pipeline.predict_proba.return_value = 0.3
        mock_ml_pipeline_class.return_value = mock_pipeline

        mock_df = Mock()
        mock_dataframe_class.return_value = mock_df

        service = RiskService()
        student = Student(data=sample_student_data)

        service.predict_risk(student)

        # Verifica se DataFrame foi criado com os dados do estudante
        mock_dataframe_class.assert_called_once_with([sample_student_data])
        mock_pipeline.predict_proba.assert_called_once_with(mock_df)

    @patch('src.application.risk_service.MLPipeline')
    def test_predict_risk_extreme_values(self, mock_ml_pipeline_class, sample_student_data):
        """
        Testa comportamento com valores extremos de probabilidade.
        
        Args:
            mock_ml_pipeline_class: Mock da classe MLPipeline
            sample_student_data: Dados de estudante
            
        Verifica:
            - Valores 0.0 e 1.0
            - Classificação correta
            - Formatação da mensagem
        """
        mock_pipeline = Mock()
        mock_ml_pipeline_class.return_value = mock_pipeline

        service = RiskService()
        student = Student(data=sample_student_data)

        # Teste com 0.0 (risco mínimo)
        mock_pipeline.predict_proba.return_value = 0.0
        result = service.predict_risk(student)

        assert result["risk_probability"] == 0.0
        assert result["risk_label"] == "BAIXO RISCO"
        assert "0.0%" in result["message"]

        # Teste com 1.0 (risco máximo)
        mock_pipeline.predict_proba.return_value = 1.0
        result = service.predict_risk(student)

        assert result["risk_probability"] == 1.0
        assert result["risk_label"] == "ALTO RISCO"
        assert "100.0%" in result["message"]

    @patch('src.application.risk_service.MLPipeline')
    def test_predict_risk_pipeline_error(self, mock_ml_pipeline_class, sample_student_data):
        """
        Testa tratamento de erro no pipeline ML.
        
        Args:
            mock_ml_pipeline_class: Mock da classe MLPipeline
            sample_student_data: Dados de estudante
            
        Verifica:
            - Exceção é propagada
            - Erro é tratado adequadamente
        """
        # Setup mock para falhar
        mock_pipeline = Mock()
        mock_pipeline.predict_proba.side_effect = Exception("Erro no modelo")
        mock_ml_pipeline_class.return_value = mock_pipeline

        service = RiskService()
        student = Student(data=sample_student_data)

        with pytest.raises(Exception) as exc_info:
            service.predict_risk(student)

        assert "Erro no modelo" in str(exc_info.value)

    @patch('src.application.risk_service.MLPipeline')
    def test_predict_risk_response_structure(self, mock_ml_pipeline_class, sample_student_data):
        """
        Testa estrutura completa da resposta.
        
        Args:
            mock_ml_pipeline_class: Mock da classe MLPipeline
            sample_student_data: Dados de estudante
            
        Verifica:
            - Todos os campos obrigatórios
            - Tipos de dados corretos
            - Valores válidos
        """
        mock_pipeline = Mock()
        mock_pipeline.predict_proba.return_value = 0.6789
        mock_ml_pipeline_class.return_value = mock_pipeline

        service = RiskService()
        student = Student(data=sample_student_data)

        result = service.predict_risk(student)

        # Verifica estrutura
        assert isinstance(result, dict)
        assert len(result) == 3

        # Verifica campos obrigatórios
        required_fields = ["risk_probability", "risk_label", "message"]
        for field in required_fields:
            assert field in result

        # Verifica tipos
        assert isinstance(result["risk_probability"], float)
        assert isinstance(result["risk_label"], str)
        assert isinstance(result["message"], str)

        # Verifica valores válidos
        assert 0.0 <= result["risk_probability"] <= 1.0
        assert result["risk_label"] in ["ALTO RISCO", "BAIXO RISCO"]
        assert len(result["message"]) > 0
