from pydantic import BaseModel, Field

class StudentData(BaseModel):
    IAN: float = Field(..., description="Indicador de Adequação ao Nível")
    IDA: float = Field(..., description="Indicador de Aprendizagem")
    IEG: float = Field(..., description="Indicador de Engajamento")
    IAA: float = Field(..., description="Indicador de Auto Avaliação")
    IPS: float = Field(..., description="Indicador Psicossocial")
    IPV: float = Field(..., description="Indicador de Ponto de Virada")
    INDE_22: float = Field(..., description="Índice de Desenvolvimento Educacional (INDE_22)")
    PEDRA_22: str = Field(..., description="Classificação Pedra (Quartzo, Ametista, etc)")
    ATINGIU_PV: str = Field(..., description="Atingiu ponto de virada? (Sim/Não)")

class PredictionResponse(BaseModel):
    risk_probability: float
    risk_label: str
    message: str