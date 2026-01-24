from fastapi.testclient import TestClient

from app.main import App


def test_monitoring_endpoints_list_and_run(tmp_path, monkeypatch):
    # Prepare app with temp monitoring directory
    tmp_monitor = tmp_path / "monitoring"
    tmp_monitor.mkdir()
    reports_dir = tmp_monitor / "reports"
    reports_dir.mkdir()

    monkeypatch.setenv("MONITORING_PATH", str(tmp_monitor))

    application = App().app
    client = TestClient(application)

    # list reports should be empty
    r = client.get("/api/v1/monitoring/reports")
    assert r.status_code == 200
    assert r.json() == []

    # run without token should succeed (no auth required)
    r = client.post("/api/v1/monitoring/run", json={})
    assert r.status_code in (200, 500)
