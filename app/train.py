import logging

from src.data.data_loader import DataLoader
from src.model.classification import StudentRiskModel
from src.model.feature_engineer import FeatureEngineer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 1. Carregar Dados
    loader = DataLoader()
    df_raw = loader.load_data()

    # --- LINHAS DE DEBUG ---
    print("\n--- COLUNAS ENCONTRADAS NO EXCEL ---")
    print(df_raw.columns.tolist())
    print("------------------------------------\n")
    # -----------------------

    df_clean = loader.clean_data(df_raw)

    # 2. Criar Target (Engenharia específica para treino)
    fe = FeatureEngineer()
    df_train = fe.create_target(df_clean)

    # 3. Treinar Modelo
    trainer = StudentRiskModel()
    trainer.train(df_train)

    print("Pipeline de treinamento finalizado com sucesso!")
