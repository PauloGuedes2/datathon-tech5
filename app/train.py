from src.infrastructure.data.data_loader import DataLoader
from src.infrastructure.model.ml_pipeline import MLPipeline
from src.util.logger import logger

if __name__ == "__main__":
    logger.info("Iniciando Pipeline de Treinamento...")

    loader = DataLoader()
    raw_df = loader.load_data()

    trainer = MLPipeline()
    df_target = trainer.create_target(raw_df)
    trainer.train(df_target)

    logger.info("Processo concluído.")

    importance_df = trainer.get_feature_importance()
    logger.info(f"\nFeature Importance:\n{importance_df}")