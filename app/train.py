from src.infrastructure.data.data_loader import DataLoader
from src.infrastructure.model.ml_pipeline import trainer
from src.util.logger import logger

if __name__ == "__main__":
    logger.info("Iniciando Pipeline de Treinamento...")

    try:
        loader = DataLoader()
        raw_df = loader.load_data()

        df_target = trainer.create_target(raw_df)

        trainer.train(df_target)

        logger.info("Processo concluído com sucesso!")

    except Exception as e:
        logger.exception(f"Ocorreu um erro fatal durante o pipeline de treinamento: {str(e)}")
        exit(1)