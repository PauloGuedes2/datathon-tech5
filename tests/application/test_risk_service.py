from unittest.mock import Mock

import numpy as np

from src.application.risk_service import RiskService
from src.domain.student import StudentInput
from src.config.settings import Settings


def test_predict_risk_success(monkeypatch, sample_student_dict):
    model = Mock()
    model.predict_proba.return_value = np.array([[0.2, 0.8]])

    logger_mock = Mock()
    repo_mock = Mock()

    service = RiskService(model=model)
    service.logger = logger_mock
    service.repository = repo_mock

    result = service.predict_risk(sample_student_dict)

    assert result["prediction"] == 1
    assert result["risk_label"] == "ALTO RISCO"
    logger_mock.log_prediction.assert_called_once()


def test_predict_risk_threshold_behavior(monkeypatch, sample_student_dict):
    model = Mock()
    model.predict_proba.return_value = np.array([[0.6, Settings.RISK_THRESHOLD]])

    service = RiskService(model=model)
    service.logger = Mock()

    result = service.predict_risk(sample_student_dict)

    assert result["prediction"] == 0
    assert result["risk_label"] == "BAIXO RISCO"


def test_predict_risk_model_none():
    service = RiskService(model=None)
    service.logger = Mock()

    try:
        service.predict_risk({})
    except RuntimeError as exc:
        assert "Modelo não inicializado" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_predict_risk_smart_with_history(monkeypatch, sample_student_input):
    model = Mock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])

    service = RiskService(model=model)
    service.logger = Mock()
    service.repository = Mock()
    service.repository.get_student_history.return_value = {
        "INDE_ANTERIOR": 1.0,
        "IAA_ANTERIOR": 1.0,
        "IEG_ANTERIOR": 1.0,
        "IPS_ANTERIOR": 1.0,
        "IDA_ANTERIOR": 1.0,
        "IPP_ANTERIOR": 1.0,
        "IPV_ANTERIOR": 1.0,
        "IAN_ANTERIOR": 1.0,
        "ALUNO_NOVO": 0,
    }

    result = service.predict_risk_smart(StudentInput(**sample_student_input))

    assert result["prediction"] == 1


def test_predict_risk_smart_without_history(sample_student_input):
    model = Mock()
    model.predict_proba.return_value = np.array([[0.7, 0.2]])

    service = RiskService(model=model)
    service.logger = Mock()
    service.repository = Mock()
    service.repository.get_student_history.return_value = None

    result = service.predict_risk_smart(StudentInput(**sample_student_input))

    assert result["prediction"] == 0
