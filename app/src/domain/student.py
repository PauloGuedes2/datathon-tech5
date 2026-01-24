from pydantic import BaseModel, Field
from typing import Optional


class Student(BaseModel):
    """
    Define a estrutura de dados esperada pela API.
    Deve corresponder às features preventivas definidas no settings.py.
    """

    # Features Numéricas
    IDADE: int = Field(..., description="Idade do aluno", example=12)
    ANO_INGRESSO: int = Field(..., description="Ano que o aluno entrou na Passos Mágicos", example=2022)

    # Features Categóricas
    GENERO: str = Field(..., description="Gênero do aluno", example="Feminino")
    TURMA: str = Field(..., description="Turma atual", example="Turma A")
    INSTITUICAO_ENSINO: str = Field(..., description="Tipo de escola (Pública/Privada)", example="Escola Pública")
    FASE: str = Field(..., description="Fase do aluno na ONG", example="2")

    # Campos opcionais (apenas para garantir compatibilidade se enviaram extra, mas não usados no modelo)
    NOME: Optional[str] = Field(None, description="Nome do aluno (opcional)")

    class Config:
        json_schema_extra = {
            "example": {
                "IDADE": 14,
                "GENERO": "Masculino",
                "TURMA": "Fase 4B",
                "INSTITUICAO_ENSINO": "Escola Pública",
                "FASE": "4",
                "ANO_INGRESSO": 2023
            }
        }