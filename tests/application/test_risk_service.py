from src.application.risk_service import RiskService
from src.domain.student import Student


def test_risk_service_prediction():
    service = RiskService()

    student = Student(
        data={
            "IDADE_22": 14,
            "CG": 7.5,
            "CF": 7.0,
            "CT": 7.2,
            "IAA": 6.8,
            "IEG": 7.1,
            "IPS": 6.9,
            "IDA": 7.0,
            "MATEM": 6.5,
            "PORTUG": 7.3,
            "INGLES": 6.8,
            "GENERO": "M",
            "TURMA": "A",
            "INSTITUICAO_DE_ENSINO": "ESCOLA MUNICIPAL"
        }
    )

    result = service.predict_risk(student)

    assert isinstance(result, dict)
    assert "risk_probability" in result
    assert "risk_label" in result
