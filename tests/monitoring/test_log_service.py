import os
import csv
import tempfile
from app.services.log_service import LogService


def test_log_service_writes_csv(tmp_path):
    path = tmp_path / "predictions.csv"
    svc = LogService(path=str(path))

    features = {"A": 1, "B": "x"}
    svc.log_prediction(features, "ALTO RISCO", 0.75)

    assert path.exists()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["prediction"] == "ALTO RISCO"
    assert float(rows[0]["probability"]) == 0.75
    assert "timestamp" in rows[0]

