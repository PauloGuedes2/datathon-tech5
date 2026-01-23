from unittest.mock import patch

import pandas as pd
import pytest

from src.infrastructure.data.data_loader import DataLoader


class TestDataLoader:
    """
    Classe de testes para DataLoader.
    
    Testa:
        - Carregamento de arquivos Excel
        - Normalização de nomes de colunas
        - Tratamento de valores nulos
        - Tratamento de erros
    """

    def test_data_loader_initialization(self):
        """
        Testa inicialização do DataLoader.
        
        Verifica:
            - Instância é criada corretamente
            - Não há estado inicial
        """
        loader = DataLoader()
        assert isinstance(loader, DataLoader)

    @patch('src.infrastructure.data.data_loader.Settings')
    @patch('src.infrastructure.data.data_loader.pd.read_excel')
    @patch('src.infrastructure.data.data_loader.logger')
    def test_load_data_success(self, mock_logger, mock_read_excel, mock_settings, sample_dataframe):
        """
        Testa carregamento bem-sucedido de dados.
        
        Args:
            mock_logger: Mock do logger
            mock_read_excel: Mock da função read_excel
            mock_settings: Mock das configurações
            sample_dataframe: DataFrame de exemplo
            
        Verifica:
            - Arquivo é carregado corretamente
            - Limpeza de colunas é aplicada
            - Logs são gerados
        """
        mock_settings.DATA_PATH = "/fake/path/data.xlsx"
        mock_read_excel.return_value = sample_dataframe

        loader = DataLoader()
        result_df = loader.load_data()

        # Verifica chamadas
        mock_read_excel.assert_called_once_with("/fake/path/data.xlsx")
        mock_logger.info.assert_called()

        # Verifica resultado
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0

    @patch('src.infrastructure.data.data_loader.Settings')
    @patch('src.infrastructure.data.data_loader.pd.read_excel')
    def test_load_data_file_not_found(self, mock_read_excel, mock_settings):
        """
        Testa comportamento quando arquivo não é encontrado.
        
        Args:
            mock_read_excel: Mock da função read_excel
            mock_settings: Mock das configurações
            
        Verifica:
            - FileNotFoundError é propagado
            - Erro é tratado adequadamente
        """
        mock_settings.DATA_PATH = "/fake/path/nonexistent.xlsx"
        mock_read_excel.side_effect = FileNotFoundError("Arquivo não encontrado")

        loader = DataLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_data()

    def test_clean_columns_normalization(self):
        """
        Testa normalização de nomes de colunas.
        
        Verifica:
            - Conversão para maiúsculo
            - Remoção de acentos
            - Substituição de espaços por underscore
            - Remoção de caracteres especiais
        """
        # DataFrame com nomes problemáticos
        df_messy = pd.DataFrame({
            'Idade do Aluno': [14, 15],
            'Nota Português': [7.5, 8.0],
            'Gênero': ['M', 'F'],
            'Instituição de Ensino': ['Escola A', 'Escola B']
        })

        result_df = DataLoader._clean_columns(df_messy)

        expected_columns = [
            'IDADE_DO_ALUNO',
            'NOTA_PORTUGUES',
            'GENERO',
            'INSTITUICAO_DE_ENSINO'
        ]

        assert list(result_df.columns) == expected_columns

    def test_clean_columns_special_characters(self):
        """
        Testa normalização com caracteres especiais.
        
        Verifica:
            - Remoção de acentos (ã, ç, é, etc.)
            - Tratamento de caracteres unicode
            - Preservação de underscores
        """
        df_special = pd.DataFrame({
            'Avaliação': [1],
            'Situação': [2],
            'Matemática': [3],
            'Português': [4],
            'Inglês': [5]
        })

        result_df = DataLoader._clean_columns(df_special)

        expected_columns = [
            'AVALIACAO',
            'SITUACAO',
            'MATEMATICA',
            'PORTUGUES',
            'INGLES'
        ]

        assert list(result_df.columns) == expected_columns

    def test_clean_columns_null_handling(self):
        """
        Testa tratamento de valores nulos.
        
        Verifica:
            - NaN são substituídos por 0
            - Estrutura do DataFrame é preservada
            - Tipos de dados são mantidos quando possível
        """
        df_with_nulls = pd.DataFrame({
            'IDADE': [14, None, 16],
            'NOTA': [7.5, None, 8.0],
            'NOME': ['João', None, 'Maria']
        })

        result_df = DataLoader._clean_columns(df_with_nulls)

        # Verifica que não há valores nulos
        assert not result_df.isnull().any().any()

        # Verifica substituição por 0
        assert result_df.loc[1, 'IDADE'] == 0
        assert result_df.loc[1, 'NOTA'] == 0.0
        assert result_df.loc[1, 'NOME'] == 0  # String None também vira 0

    def test_clean_columns_empty_dataframe(self):
        """
        Testa limpeza com DataFrame vazio.
        
        Verifica:
            - DataFrame vazio é retornado
            - Não há erro
        """
        empty_df = pd.DataFrame()

        result_df = DataLoader._clean_columns(empty_df)

        assert result_df.empty
        assert isinstance(result_df, pd.DataFrame)

    def test_clean_columns_single_row(self):
        """
        Testa limpeza com uma única linha.
        
        Verifica:
            - Processamento funciona com dados mínimos
            - Estrutura é preservada
        """
        single_row_df = pd.DataFrame({
            'Nome do Estudante': ['João'],
            'Idade em 2022': [14]
        })

        result_df = DataLoader._clean_columns(single_row_df)

        assert len(result_df) == 1
        assert list(result_df.columns) == ['NOME_DO_ESTUDANTE', 'IDADE_EM_2022']
        assert result_df.iloc[0, 0] == 'João'
        assert result_df.iloc[0, 1] == 14

    def test_clean_columns_numeric_column_names(self):
        """
        Testa normalização com nomes de colunas numéricos.
        
        Verifica:
            - Números são convertidos para string
            - Processamento funciona normalmente
        """
        numeric_columns_df = pd.DataFrame({
            123: [1, 2],
            456.78: [3, 4],
            'Normal': [5, 6]
        })

        result_df = DataLoader._clean_columns(numeric_columns_df)

        # Verifica que colunas numéricas foram convertidas
        columns = list(result_df.columns)
        assert '123' in str(columns)
        assert 'NORMAL' in columns

    def test_clean_columns_whitespace_handling(self):
        """
        Testa tratamento de espaços em branco.
        
        Verifica:
            - Espaços no início/fim são removidos
            - Espaços múltiplos são tratados
            - Tabs e quebras de linha são normalizados
        """
        whitespace_df = pd.DataFrame({
            '  Nome  ': [1],
            'Idade\t': [2],
            '\nNota\n': [3],
            'Escola   Municipal': [4]
        })

        result_df = DataLoader._clean_columns(whitespace_df)

        expected_columns = [
            'NOME',
            'IDADE',
            'NOTA',
            'ESCOLA_MUNICIPAL'  # Múltiplos espaços viram um único underscore
        ]

        assert list(result_df.columns) == expected_columns

    @patch('src.infrastructure.data.data_loader.logger')
    def test_clean_columns_logging(self, mock_logger):
        """
        Testa logging durante limpeza de colunas.
        
        Args:
            mock_logger: Mock do logger
            
        Verifica:
            - Log é gerado com colunas normalizadas
            - Informação é útil para debug
        """
        test_df = pd.DataFrame({
            'Coluna Teste': [1, 2],
            'Outra Coluna': [3, 4]
        })

        DataLoader._clean_columns(test_df)

        # Verifica que log foi chamado
        mock_logger.info.assert_called_once()

        # Verifica conteúdo do log
        log_call = mock_logger.info.call_args[0][0]
        assert "Colunas normalizadas" in log_call
        assert "COLUNA_TESTE" in log_call
        assert "OUTRA_COLUNA" in log_call

    def test_integration_load_and_clean(self, temp_excel_file):
        """
        Testa integração completa de carregamento e limpeza.
        
        Args:
            temp_excel_file: Arquivo Excel temporário
            
        Verifica:
            - Carregamento + limpeza funcionam juntos
            - Resultado final está correto
        """
        with patch('src.infrastructure.data.data_loader.Settings') as mock_settings:
            mock_settings.DATA_PATH = temp_excel_file

            loader = DataLoader()
            result_df = loader.load_data()

            # Verifica que dados foram carregados e limpos
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) > 0

            # Verifica que colunas foram normalizadas
            columns = list(result_df.columns)
            assert all(col.isupper() for col in columns if isinstance(col, str))

            # Verifica que não há valores nulos
            assert not result_df.isnull().any().any()
