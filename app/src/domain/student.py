from dataclasses import dataclass

@dataclass
class Student:
    """
    Entidade de domínio representando um estudante.
    
    Attributes:
        data: Dicionário com todos os dados do estudante
    """
    data: dict
