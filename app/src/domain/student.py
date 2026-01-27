from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class Student(BaseModel):
    """Define a estrutura de dados com validações rigorosas (Data Quality)."""

    IDADE: int = Field(..., ge=4, le=25, description="Idade entre 4 e 25 anos")
    ANO_INGRESSO: int = Field(..., ge=2010, le=2026, description="Ano de entrada válido")

    GENERO: str = Field(..., pattern="^(Masculino|Feminino|Outro)$")
    TURMA: str = Field(..., min_length=1)
    INSTITUICAO_ENSINO: str = Field(..., min_length=3)
    FASE: str = Field(..., pattern="^[0-9A-Z]+$")

    NOME: Optional[str] = None

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "IDADE": 14,
                "GENERO": "Masculino",
                "TURMA": "Fase 4B",
                "INSTITUICAO_ENSINO": "Escola Pública",
                "FASE": "4",
                "ANO_INGRESSO": 2023
            }
        }
    )