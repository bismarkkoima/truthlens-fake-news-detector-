import pytest


def test_home_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Does this read like fake news?" in res.data


@pytest.mark.parametrize("model", ["logreg", "nb"])
def test_check_flags_fake(client, fake_article, model):
    res = client.post("/check", data={"text": fake_article, "model": model})
    assert res.status_code == 200
    assert b"Shows signs of fake news" in res.data


@pytest.mark.parametrize("model", ["logreg", "nb"])
def test_check_passes_real(client, real_article, model):
    res = client.post("/check", data={"text": real_article, "model": model})
    assert res.status_code == 200
    assert b"Likely genuine reporting" in res.data


def test_short_text_is_rejected(client):
    res = client.post("/check", data={"text": "too short", "model": "logreg"})
    assert res.status_code == 400
    assert b"at least 20 words" in res.data


def test_unknown_model_is_rejected(client, real_article):
    res = client.post("/check", data={"text": real_article, "model": "nope"})
    assert res.status_code == 400


def test_api_returns_verdict_and_evidence(client, fake_article):
    res = client.post("/api/check", json={"text": fake_article})
    body = res.get_json()
    assert res.status_code == 200
    assert body["label"] == "FAKE"
    assert 50 <= body["confidence"] <= 100
    assert body["evidence"] and all({"term", "weight"} <= e.keys() for e in body["evidence"])


def test_api_validation_error(client):
    res = client.post("/api/check", json={"text": ""})
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_history_records_and_filters(client, fake_article, real_article):
    client.post("/check", data={"text": fake_article, "model": "logreg"})
    client.post("/check", data={"text": real_article, "model": "nb"})

    assert b"Recent checks" in client.get("/history").data
    only_fake = client.get("/history?label=FAKE").data
    assert b"FAKE \xc2\xb7" in only_fake and b"REAL \xc2\xb7" not in only_fake


def test_api_save_false_skips_history(client, app, real_article):
    client.post("/api/check", json={"text": real_article, "save": False})
    assert app.extensions["history"].summary()["total"] == 0


def test_clear_history(client, app, real_article):
    client.post("/check", data={"text": real_article, "model": "logreg"})
    client.post("/history/clear")
    assert app.extensions["history"].summary()["total"] == 0


def test_stats_and_about(client, real_article):
    assert b"Stats show up here" in client.get("/stats").data
    client.post("/check", data={"text": real_article, "model": "logreg"})
    stats = client.get("/stats")
    assert stats.status_code == 200 and b"Checks per day" in stats.data
    assert b"Logistic Regression" in client.get("/about").data


def test_missing_model_shows_helpful_error(tmp_path):
    from detector import create_app

    app = create_app({"MODEL_PATH": tmp_path / "missing.joblib", "DATABASE": tmp_path / "db.sqlite3"})
    res = app.test_client().post("/check", data={"text": "word " * 30})
    assert res.status_code == 400
    assert b"python train.py" in res.data


def test_404(client):
    assert client.get("/nowhere").status_code == 404
