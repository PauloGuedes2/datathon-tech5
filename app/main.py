import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException

from src.api.schemas import StudentData, PredictionResponse
from src.model.classification import StudentRiskModel, logger

app = FastAPI(
    title="API Passos Mágicos - Previsão de Risco",
    description="Predição de risco de defasagem escolar baseada em indicadores do PEDE.",
    version="1.0.0"
)

# Instância global do modelo
model_service = StudentRiskModel()


@app.on_event("startup")
def load_model():
    try:
        model_service.load()
        logger.info("Modelo carregado com sucesso na inicialização.")
    except Exception as e:
        logger.warning(f"Modelo não encontrado ({e}). Certifique-se de treinar o modelo primeiro.")


@app.post("/predict", response_model=PredictionResponse)
def predict_student_risk(student: StudentData):
    """
    Recebe os indicadores de um estudante e retorna o risco de defasagem.
    """
    try:
        # Converter Pydantic para DataFrame
        input_df = pd.DataFrame([student.dict()])

        # Realizar previsão
        risk_prob = model_service.predict(input_df)[0]

        # Lógica de Threshold (Pode vir do Params)
        threshold = 0.5
        risk_label = "ALTO RISCO" if risk_prob > threshold else "BAIXO RISCO"

        return {
            "risk_probability": round(risk_prob, 4),
            "risk_label": risk_label,
            "message": f"O estudante possui {risk_prob:.1%} de probabilidade de defasagem."
        }
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail="Erro interno ao processar previsão.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
