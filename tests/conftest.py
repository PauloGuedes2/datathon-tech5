import pytest
from fastapi.testclient import TestClient

from main import App


@pytest.fixture(scope="session")
def client():
    app = App().app
    return TestClient(app)
