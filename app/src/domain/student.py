from typing import Optional

from pydantic import BaseModel, Field, ConfigDict

""" Módulo de domínio para representar os dados do aluno, incluindo os dados demográficos e o histórico acadêmico (lag features). """


class Student(BaseModel):
    """ Representa um aluno com seus dados demográficos e históricos acadêmicos."""
    RA: str = Field(..., min_length=1, description="Registro Acadêmico Único do Aluno")
    IDADE: int = Field(..., ge=4, le=25)
    ANO_INGRESSO: int = Field(..., ge=2010, le=2026)
    GENERO: str = Field(..., pattern="^(Masculino|Feminino|Outro)$")
    TURMA: str = Field(..., min_length=1)
    INSTITUICAO_ENSINO: str = Field(..., min_length=3)
    FASE: str = Field(..., pattern="^[0-9A-Z]+$")
    NOME: Optional[str] = None
    INDE_ANTERIOR: float = Field(..., ge=0, le=10, description="INDE do ano passado. Se novo, usar 0 ou média.")
    IAA_ANTERIOR: Optional[float] = 0.0
    IEG_ANTERIOR: Optional[float] = 0.0
    IPS_ANTERIOR: Optional[float] = 0.0
    IDA_ANTERIOR: Optional[float] = 0.0
    IPP_ANTERIOR: Optional[float] = 0.0
    IPV_ANTERIOR: Optional[float] = 0.0
    IAN_ANTERIOR: Optional[float] = 0.0
    ALUNO_NOVO: int = Field(0, description="1 se entrou este ano, 0 se veterano")

    model_config = ConfigDict(populate_by_name=True)


class StudentInput(BaseModel):
    """
    O que o cliente realmente preenche na tela.
    """
    RA: str = Field(..., min_length=1, description="Registro Acadêmico Único do Aluno")
    IDADE: int = Field(..., ge=4, le=25)
    ANO_INGRESSO: int = Field(..., ge=2010, le=2026)
    GENERO: str = Field(..., pattern="^(Masculino|Feminino|Outro)$")
    TURMA: str = Field(..., min_length=1)
    INSTITUICAO_ENSINO: str = Field(..., min_length=3)
    FASE: str = Field(..., pattern="^[0-9A-Z]+$")
