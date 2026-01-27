from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class Student(BaseModel):
    # --- Dados Demográficos (Mantém) ---
    IDADE: int = Field(..., ge=4, le=25)
    ANO_INGRESSO: int = Field(..., ge=2010, le=2026)
    GENERO: str = Field(..., pattern="^(Masculino|Feminino|Outro)$")
    TURMA: str = Field(..., min_length=1)
    INSTITUICAO_ENSINO: str = Field(..., min_length=3)
    FASE: str = Field(..., pattern="^[0-9A-Z]+$")
    NOME: Optional[str] = None

    # --- NOVOS CAMPOS: O Histórico (Lag Features) ---
    # Note que pedimos o INDE_ANTERIOR, não o atual.
    INDE_ANTERIOR: float = Field(..., ge=0, le=10, description="INDE do ano passado. Se novo, usar 0 ou média.")
    IAA_ANTERIOR: Optional[float] = 0.0
    IEG_ANTERIOR: Optional[float] = 0.0
    IPS_ANTERIOR: Optional[float] = 0.0
    IDA_ANTERIOR: Optional[float] = 0.0
    IPP_ANTERIOR: Optional[float] = 0.0
    IPV_ANTERIOR: Optional[float] = 0.0
    IAN_ANTERIOR: Optional[float] = 0.0

    # Campo Booleano para saber se o aluno é novo na ONG
    ALUNO_NOVO: int = Field(0, description="1 se entrou este ano, 0 se veterano")

    model_config = ConfigDict(populate_by_name=True)
