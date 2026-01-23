from src.infrastructure.data.data_loader import DataLoader
from src.infrastructure.model.ml_pipeline import MLPipeline
from src.util.logger import logger

if __name__ == "__main__":
    """
    Executa o pipeline completo de treinamento do modelo.
    
    Processo:
        1. Carrega dados do Excel
        2. Cria variável target baseada em DEFAS
        3. Treina modelo Random Forest
        4. Salva modelo treinado
        5. Exibe importância das features
    """
    logger.info("Iniciando Pipeline de Treinamento...")

    loader = DataLoader()
    raw_df = loader.load_data()

    trainer = MLPipeline()
    df_target = trainer.create_target(raw_df)
    trainer.train(df_target)

    logger.info("Processo concluído.")

    importance_df = trainer.get_feature_importance()
    logger.info(f"\nFeature Importance:\n{importance_df}")