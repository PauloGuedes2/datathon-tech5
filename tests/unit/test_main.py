from unittest.mock import Mock

from fastapi.testclient import TestClient

import main


def test_health_check():
    client = TestClient(main.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_startup_event_loads_model(monkeypatch):
    manager = Mock()
    monkeypatch.setattr(main, "ModelManager", lambda: manager)

    client = TestClient(main.app)
    with client:
        pass

    manager.load_model.assert_called_once()
