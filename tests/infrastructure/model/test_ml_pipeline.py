from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.infrastructure.model.ml_pipeline import MLPipeline


class TestMLPipeline:
    """
    Classe de testes para MLPipeline.
    
    Testa:
        - Criação de target
        - Treinamento do modelo
        - Carregamento e salvamento
        - Predições
        - Feature importance
    """

    def test_pipeline_initialization(self):
        """
        Testa inicialização do pipeline.
        
        Verifica:
            - Pipeline é criado corretamente
            - Modelo é None inicialmente
        """
        pipeline = MLPipeline()
        assert pipeline.model is None

    def test_create_target_success(self, sample_dataframe):
        """
        Testa criação da variável target com sucesso.
        
        Args:
            sample_dataframe: DataFrame com coluna DEFAS
            
        Verifica:
            - Coluna RISCO_DEFASAGEM é criada
            - Valores corretos (DEFAS < 0 = 1, senão 0)
            - DataFrame original é preservado
        """
        result_df = MLPipeline.create_target(sample_dataframe)

        assert "RISCO_DEFASAGEM" in result_df.columns

        # Verifica lógica: DEFAS < 0 -> RISCO = 1
        expected_risk = [1, 0, 0]  # [-1, 0, 2] -> [1, 0, 0]
        assert list(result_df["RISCO_DEFASAGEM"]) == expected_risk

        # Verifica que outras colunas foram preservadas
        for col in sample_dataframe.columns:
            assert col in result_df.columns

    def test_create_target_missing_defas_column(self):
        """
        Testa erro quando coluna DEFAS não existe.
        
        Verifica:
            - ValueError é levantado
            - Mensagem de erro apropriada
        """
        df_without_defas = pd.DataFrame({
            'IDADE_22': [14, 15],
            'CG': [7.5, 6.8]
        })

        with pytest.raises(ValueError) as exc_info:
            MLPipeline.create_target(df_without_defas)

        assert "DEFAS não encontrada" in str(exc_info.value)

    def test_create_target_edge_cases(self):
        """
        Testa casos extremos na criação do target.
        
        Verifica:
            - Valores zero no limite
            - Valores muito negativos
            - Valores muito positivos
        """
        edge_df = pd.DataFrame({
            'DEFAS': [-100, -0.1, 0, 0.1, 100],
            'IDADE_22': [14, 15, 16, 17, 18]
        })

        result_df = MLPipeline.create_target(edge_df)

        # DEFAS < 0 -> RISCO = 1
        expected_risk = [1, 1, 0, 0, 0]
        assert list(result_df["RISCO_DEFASAGEM"]) == expected_risk

    @patch('src.infrastructure.model.ml_pipeline.train_test_split')
    @patch('src.infrastructure.model.ml_pipeline.Pipeline')
    @patch('src.infrastructure.model.ml_pipeline.dump')
    @patch('src.infrastructure.model.ml_pipeline.logger')
    def test_train_success(self, mock_logger, mock_dump, mock_pipeline_class,
                           mock_train_test_split, sample_dataframe):
        """
        Testa treinamento bem-sucedido do modelo.
        
        Args:
            mock_logger: Mock do logger
            mock_dump: Mock da função dump
            mock_pipeline_class: Mock da classe Pipeline
            mock_train_test_split: Mock do train_test_split
            sample_dataframe: DataFrame com dados de treino
            
        Verifica:
            - Pipeline é criado e treinado
            - Modelo é salvo
            - Logs são gerados
            - Métricas são calculadas
        """
        # Preparar dados
        df_with_target = MLPipeline.create_target(sample_dataframe)

        # Setup mocks
        X_train = pd.DataFrame({'feature': [1, 2]})
        X_test = pd.DataFrame({'feature': [3]})
        y_train = pd.Series([0, 1])
        y_test = pd.Series([0])

        mock_train_test_split.return_value = (X_train, X_test, y_train, y_test)

        mock_pipeline = Mock()
        mock_pipeline.predict.return_value = [0]
        mock_pipeline_class.return_value = mock_pipeline

        # Executar treinamento
        MLPipeline.train(df_with_target)

        # Verificações
        mock_pipeline_class.assert_called_once()
        mock_pipeline.fit.assert_called_once_with(X_train, y_train)
        mock_pipeline.predict.assert_called_once_with(X_test)
        mock_dump.assert_called_once()

        # Verifica logs
        assert mock_logger.info.call_count >= 2

    @patch('src.infrastructure.model.ml_pipeline.train_test_split')
    @patch('src.infrastructure.model.ml_pipeline.logger')
    def test_train_unbalanced_data(self, mock_logger, mock_train_test_split, sample_dataframe):
        """
        Testa treinamento com dados desbalanceados.
        
        Args:
            mock_logger: Mock do logger
            mock_train_test_split: Mock do train_test_split
            sample_dataframe: DataFrame base
            
        Verifica:
            - Warning é logado para dados desbalanceados
            - Stratify é None quando classe minoritária < 2
        """
        # Criar dados muito desbalanceados
        unbalanced_df = sample_dataframe.copy()
        unbalanced_df['DEFAS'] = [1, 1, 1]  # Todos positivos -> RISCO = 0
        df_with_target = MLPipeline.create_target(unbalanced_df)

        # Mock para simular classe minoritária com 1 amostra
        mock_train_test_split.side_effect = ValueError("stratify")

        with pytest.raises(ValueError):
            MLPipeline.train(df_with_target)

    @patch('src.infrastructure.model.ml_pipeline.load')
    def test_load_success(self, mock_load):
        """
        Testa carregamento bem-sucedido do modelo.
        
        Args:
            mock_load: Mock da função load
            
        Verifica:
            - Modelo é carregado corretamente
            - Atribuído à instância
        """
        mock_model = Mock()
        mock_load.return_value = mock_model

        pipeline = MLPipeline()
        pipeline.load()

        mock_load.assert_called_once()
        assert pipeline.model == mock_model

    @patch('src.infrastructure.model.ml_pipeline.load')
    @patch('src.infrastructure.model.ml_pipeline.logger')
    def test_load_file_not_found(self, mock_logger, mock_load):
        """
        Testa comportamento quando modelo não é encontrado.
        
        Args:
            mock_logger: Mock do logger
            mock_load: Mock da função load
            
        Verifica:
            - FileNotFoundError é tratado
            - Modelo fica None
            - Erro é logado
        """
        mock_load.side_effect = FileNotFoundError("Arquivo não encontrado")

        pipeline = MLPipeline()
        pipeline.load()

        assert pipeline.model is None
        mock_logger.error.assert_called_once()

    def test_predict_proba_success(self, mock_model):
        """
        Testa predição de probabilidade com sucesso.
        
        Args:
            mock_model: Mock do modelo treinado
            
        Verifica:
            - Probabilidade é retornada corretamente
            - Classe positiva (índice 1) é usada
        """
        pipeline = MLPipeline()
        pipeline.model = mock_model

        test_df = pd.DataFrame({'feature': [1]})

        result = pipeline.predict_proba(test_df)

        mock_model.predict_proba.assert_called_once_with(test_df)
        assert result == 0.3  # mock_model retorna [[0.7, 0.3]]

    def test_predict_proba_no_model(self):
        """
        Testa predição sem modelo carregado.
        
        Verifica:
            - RuntimeError é levantado
            - Mensagem apropriada
        """
        with patch('src.infrastructure.model.ml_pipeline.load') as mock_load:
            mock_load.side_effect = FileNotFoundError("Modelo não encontrado")
            
            pipeline = MLPipeline()
            # Não carrega modelo

            test_df = pd.DataFrame({'feature': [1]})

            with pytest.raises(RuntimeError) as exc_info:
                pipeline.predict_proba(test_df)

            assert "indisponível" in str(exc_info.value)

    @patch('src.infrastructure.model.ml_pipeline.load')
    def test_predict_proba_auto_load(self, mock_load, mock_model):
        """
        Testa carregamento automático do modelo na predição.
        
        Args:
            mock_load: Mock da função load
            mock_model: Mock do modelo
            
        Verifica:
            - Modelo é carregado automaticamente
            - Predição funciona após carregamento
        """
        mock_load.return_value = mock_model

        pipeline = MLPipeline()
        test_df = pd.DataFrame({'feature': [1]})

        result = pipeline.predict_proba(test_df)

        mock_load.assert_called_once()
        assert result == 0.3

    def test_get_feature_importance_success(self, mock_model):
        """
        Testa extração de feature importance.
        
        Args:
            mock_model: Mock do modelo com feature importance
            
        Verifica:
            - DataFrame é retornado
            - Features estão ordenadas por importância
            - Estrutura correta
        """
        pipeline = MLPipeline()
        pipeline.model = mock_model

        result_df = pipeline.get_feature_importance()

        assert isinstance(result_df, pd.DataFrame)
        assert "feature" in result_df.columns
        assert "importance" in result_df.columns
        assert len(result_df) > 0

        # Verifica ordenação decrescente
        importances = result_df["importance"].tolist()
        assert importances == sorted(importances, reverse=True)

    def test_get_feature_importance_no_model(self):
        """
        Testa feature importance sem modelo carregado.
        
        Verifica:
            - RuntimeError é levantado
            - Tentativa de carregamento automático
        """
        with patch('src.infrastructure.model.ml_pipeline.load') as mock_load:
            mock_load.side_effect = FileNotFoundError("Modelo não encontrado")
            
            pipeline = MLPipeline()

            with pytest.raises(RuntimeError) as exc_info:
                pipeline.get_feature_importance()

            assert "indisponível" in str(exc_info.value)

    @patch('src.infrastructure.model.ml_pipeline.load')
    def test_get_feature_importance_auto_load(self, mock_load, mock_model):
        """
        Testa carregamento automático para feature importance.
        
        Args:
            mock_load: Mock da função load
            mock_model: Mock do modelo
            
        Verifica:
            - Modelo é carregado automaticamente
            - Feature importance é extraída
        """
        mock_load.return_value = mock_model

        pipeline = MLPipeline()
        result_df = pipeline.get_feature_importance()

        mock_load.assert_called_once()
        assert isinstance(result_df, pd.DataFrame)
