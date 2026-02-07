from unittest.mock import Mock

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.api.controller import PredictionController, get_risk_service
from src.domain.student import Student, StudentInput


def test_predict_full_success(sample_student_dict):
    app = FastAPI()
    controller = PredictionController()

    service = Mock()
    service.predict_risk.return_value = {"prediction": 1}

    def override():
        return service

    app.dependency_overrides[get_risk_service] = override
    app.include_router(controller.router, prefix="/api/v1")

    client = TestClient(app)

    response = client.post("/api/v1/predict/full", json=sample_student_dict)

    assert response.status_code == 200
    assert response.json()["prediction"] == 1


def test_predict_full_error(sample_student_dict):
    app = FastAPI()
    controller = PredictionController()

    service = Mock()
    service.predict_risk.side_effect = RuntimeError("boom")

    def override():
        return service

    app.dependency_overrides[get_risk_service] = override
    app.include_router(controller.router, prefix="/api/v1")

    client = TestClient(app)

    response = client.post("/api/v1/predict/full", json=sample_student_dict)

    assert response.status_code == 500
    assert "boom" in response.json()["detail"]


def test_predict_smart_success(sample_student_input):
    app = FastAPI()
    controller = PredictionController()

    service = Mock()
    service.predict_risk_smart.return_value = {"prediction": 0}

    def override():
        return service

    app.dependency_overrides[get_risk_service] = override
    app.include_router(controller.router, prefix="/api/v1")

    client = TestClient(app)

    response = client.post("/api/v1/predict/smart", json=sample_student_input)

    assert response.status_code == 200
    assert response.json()["prediction"] == 0


def test_get_risk_service_raises_when_no_model(monkeypatch):
    from src.api import controller as controller_module

    manager = Mock()
    manager.get_model.side_effect = RuntimeError("missing")
    monkeypatch.setattr(controller_module, "model_manager", manager)

    try:
        controller_module.get_risk_service()
    except HTTPException as exc:
        assert exc.status_code == 503
    else:
        raise AssertionError("Expected HTTPException")


def test_prediction_models_validate(sample_student_dict, sample_student_input):
    student = Student(**sample_student_dict)
    student_input = StudentInput(**sample_student_input)

    assert student.RA == "123"
    assert student_input.RA == "123"
