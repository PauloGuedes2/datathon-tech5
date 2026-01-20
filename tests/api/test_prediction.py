def test_predict_success(client):
    payload = {
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

    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert "risk_probability" in body
    assert "risk_label" in body
    assert "message" in body

    assert 0.0 <= body["risk_probability"] <= 1.0
    assert body["risk_label"] in ["ALTO RISCO", "BAIXO RISCO"]

def test_predict_missing_field(client):
    payload = {
        "IDADE_22": 14,
        "CG": 7.5
    }

    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 422  # Validation error

def test_predict_invalid_type(client):
    payload = {
        "IDADE_22": "quatorze",  # inválido
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

    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 422
