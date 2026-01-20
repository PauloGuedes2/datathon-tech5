from pydantic import BaseModel


class StudentDTO(BaseModel):
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
