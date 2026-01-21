from pydantic import BaseModel


class StudentDTO(BaseModel):
    """
    Schema de validação para dados de entrada do estudante.
    
    Attributes:
        Features numéricas:
            IDADE_22: Idade do estudante em 2022
            CG, CF, CT: Métricas de competências
            IAA, IEG, IPS, IDA: Indicadores acadêmicos
            MATEM, PORTUG, INGLES: Notas das disciplinas
            
        Features categóricas:
            GENERO: Gênero do estudante
            TURMA: Turma do estudante
            INSTITUICAO_DE_ENSINO: Instituição de ensino
    """
    IDADE_22: int
    CG: float
    CF: float
    CT: float
    IAA: float
    IEG: float
    IPS: float
    IDA: float
    MATEM: float
    PORTUG: float
    INGLES: float

    GENERO: str
    TURMA: str
    INSTITUICAO_DE_ENSINO: str
