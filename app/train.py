from src.infrastructure.data.data_loader import DataLoader
from src.infrastructure.model.ml_pipeline import trainer
from src.util.logger import logger

if __name__ == "__main__":
    logger.info("Iniciando Pipeline de Treinamento...")

    try:
        # 1. Carregamento
        loader = DataLoader()
        raw_df = loader.load_data()

        # 2. Cria Target
        df_target = trainer.create_target(raw_df)

        # 3. Treina e Salva (Modelo + Dados de Referência)
        trainer.train(df_target)

        logger.info("Processo concluído com sucesso!")

    except Exception as e:
        logger.exception("Ocorreu um erro fatal durante o pipeline.")
        exit(1)