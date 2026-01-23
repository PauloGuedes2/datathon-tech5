from unittest.mock import patch

import pandas as pd

from src.infrastructure.model.feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    """
    Classe de testes para FeatureEngineer.
    
    Testa:
        - Inicialização do transformador
        - Treinamento (fit)
        - Transformação (transform)
        - Tratamento de valores não vistos
        - Integração com sklearn pipeline
    """

    def test_feature_engineer_initialization(self):
        """
        Testa inicialização do FeatureEngineer.
        
        Verifica:
            - Instância é criada corretamente
            - Encoders dict está vazio inicialmente
        """
        fe = FeatureEngineer()
        assert isinstance(fe.encoders, dict)
        assert len(fe.encoders) == 0

    @patch('src.infrastructure.model.feature_engineer.Settings')
    def test_fit_success(self, mock_settings):
        """
        Testa treinamento bem-sucedido do transformador.
        
        Args:
            mock_settings: Mock das configurações
            
        Verifica:
            - LabelEncoders são criados para features categóricas
            - Encoders são treinados com dados corretos
            - Self é retornado (sklearn pattern)
        """
        # Setup mock settings
        mock_settings.FEATURES_CATEGORICAS = ['GENERO', 'TURMA']

        # Dados de treino
        X_train = pd.DataFrame({
            'GENERO': ['M', 'F', 'M'],
            'TURMA': ['A', 'B', 'A'],
            'IDADE_22': [14, 15, 16]  # Feature numérica (ignorada)
        })

        fe = FeatureEngineer()
        result = fe.fit(X_train)

        # Verifica retorno
        assert result == fe

        # Verifica encoders criados
        assert 'GENERO' in fe.encoders
        assert 'TURMA' in fe.encoders
        assert 'IDADE_22' not in fe.encoders  # Não é categórica

        # Verifica que encoders foram treinados
        assert hasattr(fe.encoders['GENERO'], 'classes_')
        assert hasattr(fe.encoders['TURMA'], 'classes_')

    @patch('src.infrastructure.model.feature_engineer.Settings')
    def test_fit_missing_columns(self, mock_settings):
        """
        Testa fit quando algumas colunas categóricas não existem.
        
        Args:
            mock_settings: Mock das configurações
            
        Verifica:
            - Apenas colunas existentes são processadas
            - Não há erro para colunas ausentes
        """
        mock_settings.FEATURES_CATEGORICAS = ['GENERO', 'TURMA', 'INEXISTENTE']

        X_train = pd.DataFrame({
            'GENERO': ['M', 'F'],
            'TURMA': ['A', 'B']
            # INEXISTENTE não está presente
        })

        fe = FeatureEngineer()
        fe.fit(X_train)

        # Verifica que apenas colunas existentes foram processadas
        assert 'GENERO' in fe.encoders
        assert 'TURMA' in fe.encoders
        assert 'INEXISTENTE' not in fe.encoders

    @patch('src.infrastructure.model.feature_engineer.Settings')
    def test_transform_success(self, mock_settings):
        """
        Testa transformação bem-sucedida.
        
        Args:
            mock_settings: Mock das configurações
            
        Verifica:
            - Valores categóricos são encodados corretamente
            - DataFrame original não é modificado
            - Estrutura é preservada
        """
        mock_settings.FEATURES_CATEGORICAS = ['GENERO', 'TURMA']

        # Dados de treino
        X_train = pd.DataFrame({
            'GENERO': ['M', 'F', 'M'],
            'TURMA': ['A', 'B', 'A'],
            'IDADE_22': [14, 15, 16]
        })

        # Dados de teste (mesmos valores)
        X_test = pd.DataFrame({
            'GENERO': ['F', 'M'],
            'TURMA': ['B', 'A'],
            'IDADE_22': [17, 18]
        })

        fe = FeatureEngineer()
        fe.fit(X_train)
        X_transformed = fe.transform(X_test)

        # Verifica que transformação foi aplicada
        assert not X_transformed.equals(X_test)  # Dados foram modificados

        # Verifica que colunas categóricas foram encodadas
        assert X_transformed['GENERO'].dtype in ['int64', 'int32']
        assert X_transformed['TURMA'].dtype in ['int64', 'int32']

        # Verifica que coluna numérica não foi alterada
        assert X_transformed['IDADE_22'].equals(X_test['IDADE_22'])

    @patch('src.infrastructure.model.feature_engineer.Settings')
    def test_transform_unseen_values(self, mock_settings):
        """
        Testa transformação com valores não vistos durante fit.
        
        Args:
            mock_settings: Mock das configurações
            
        Verifica:
            - Valores não vistos são mapeados para -1
            - Valores conhecidos são encodados normalmente
        """
        mock_settings.FEATURES_CATEGORICAS = ['GENERO']

        # Treino apenas com 'M' e 'F'
        X_train = pd.DataFrame({
            'GENERO': ['M', 'F', 'M']
        })

        # Teste com valor não visto 'X'
        X_test = pd.DataFrame({
            'GENERO': ['M', 'X', 'F']  # 'X' não foi visto no treino
        })

        fe = FeatureEngineer()
        fe.fit(X_train)
        X_transformed = fe.transform(X_test)

        # Verifica mapeamento
        transformed_values = X_transformed['GENERO'].tolist()

        # 'X' deve ser mapeado para -1
        assert -1 in transformed_values

        # Valores conhecidos devem ter encoding válido (0 ou 1)
        known_values = [v for v in transformed_values if v != -1]
        assert all(v in [0, 1] for v in known_values)

    @patch('src.infrastructure.model.feature_engineer.Settings')
    def test_transform_empty_dataframe(self, mock_settings):
        """
        Testa transformação com DataFrame vazio.
        
        Args:
            mock_settings: Mock das configurações
            
        Verifica:
            - DataFrame vazio é retornado
            - Não há erro
        """
        mock_settings.FEATURES_CATEGORICAS = ['GENERO']

        X_train = pd.DataFrame({
            'GENERO': ['M', 'F']
        })

        X_empty = pd.DataFrame()

        fe = FeatureEngineer()
        fe.fit(X_train)
        X_transformed = fe.transform(X_empty)

        assert X_transformed.empty
        assert isinstance(X_transformed, pd.DataFrame)

    @patch('src.infrastructure.model.feature_engineer.Settings')
    def test_transform_without_fit(self, mock_settings):
        """
        Testa transformação sem treinamento prévio.
        
        Args:
            mock_settings: Mock das configurações
            
        Verifica:
            - Não há erro (encoders dict vazio)
            - DataFrame é retornado inalterado
        """
        mock_settings.FEATURES_CATEGORICAS = ['GENERO']

        X_test = pd.DataFrame({
            'GENERO': ['M', 'F'],
            'IDADE_22': [14, 15]
        })

        fe = FeatureEngineer()
        # Não chama fit()
        X_transformed = fe.transform(X_test)

        # DataFrame deve ser retornado inalterado
        pd.testing.assert_frame_equal(X_transformed, X_test)

    @patch('src.infrastructure.model.feature_engineer.Settings')
    def test_fit_transform_pipeline(self, mock_settings):
        """
        Testa integração fit + transform (padrão sklearn).
        
        Args:
            mock_settings: Mock das configurações
            
        Verifica:
            - Fit e transform funcionam em sequência
            - Resultado é consistente
        """
        mock_settings.FEATURES_CATEGORICAS = ['GENERO', 'TURMA']

        X = pd.DataFrame({
            'GENERO': ['M', 'F', 'M', 'F'],
            'TURMA': ['A', 'B', 'A', 'B'],
            'IDADE_22': [14, 15, 16, 17]
        })

        fe = FeatureEngineer()

        # Fit + Transform
        fe.fit(X)
        X_transformed = fe.transform(X)

        # Verifica que todas as linhas foram processadas
        assert len(X_transformed) == len(X)

        # Verifica que colunas categóricas foram encodadas
        assert X_transformed['GENERO'].dtype in ['int64', 'int32']
        assert X_transformed['TURMA'].dtype in ['int64', 'int32']

        # Verifica valores únicos (deve ter 2 valores para cada feature categórica)
        assert len(X_transformed['GENERO'].unique()) == 2
        assert len(X_transformed['TURMA'].unique()) == 2

    @patch('src.infrastructure.model.feature_engineer.Settings')
    def test_string_conversion(self, mock_settings):
        """
        Testa conversão automática para string.
        
        Args:
            mock_settings: Mock das configurações
            
        Verifica:
            - Valores numéricos em colunas categóricas são convertidos
            - Conversão não afeta o encoding
        """
        mock_settings.FEATURES_CATEGORICAS = ['CATEGORIA_NUMERICA']

        # Dados com valores numéricos em coluna categórica
        X_train = pd.DataFrame({
            'CATEGORIA_NUMERICA': [1, 2, 1, 2]  # Números como categorias
        })

        X_test = pd.DataFrame({
            'CATEGORIA_NUMERICA': [2, 1]
        })

        fe = FeatureEngineer()
        fe.fit(X_train)
        X_transformed = fe.transform(X_test)

        # Verifica que encoding funcionou
        assert X_transformed['CATEGORIA_NUMERICA'].dtype in ['int64', 'int32']

        # Verifica que valores foram mapeados corretamente
        unique_values = X_transformed['CATEGORIA_NUMERICA'].unique()
        assert len(unique_values) == 2  # Dois valores únicos
        assert all(v in [0, 1] for v in unique_values)  # Encodados como 0, 1
