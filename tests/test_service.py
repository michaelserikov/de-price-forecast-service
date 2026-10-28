import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dpfs import daily, models
from dpfs.service import create_app


@pytest.fixture(scope="module")
def model_dir(inp, tmp_path_factory):
    folder = tmp_path_factory.mktemp("model")
    days = inp["price"].index
    models.Ensemble([models.fit("mlp", inp, days[7:300], epochs=20), models.fit("lstm", inp, days[7:300], epochs=10)]).save(folder)
    return folder


def test_health_and_forecast(inp, model_dir):
    client = TestClient(create_app(model_dir, provider=lambda day: inp))
    assert client.get("/health").json()["model_loaded"] is True
    r = client.get("/forecast", params={"day": "2023-12-01"})
    assert r.status_code == 200
    hours = r.json()["hours"]
    assert len(hours) == 24 and all(h["q10"] <= h["q50"] <= h["q90"] for h in hours)
    assert {m["kind"] for m in client.get("/model").json()["members"]} == {"mlp", "lstm"}


def test_forecast_errors(inp, model_dir, tmp_path):
    client = TestClient(create_app(model_dir, provider=lambda day: inp))
    assert client.get("/forecast", params={"day": "2030-01-01"}).status_code == 404
    assert client.get("/forecast", params={"day": "not-a-day"}).status_code == 422
    empty = TestClient(create_app(tmp_path, provider=lambda day: inp))
    assert empty.get("/health").json()["model_loaded"] is False
    assert empty.get("/forecast", params={"day": "2023-12-01"}).status_code == 503


def test_daily_scoring_and_retrain_rule(inp, model_dir, tmp_path):
    ens = models.Ensemble.load(model_dir)
    for day in ("2023-12-01", "2023-12-02"):
        d = pd.Timestamp(day)
        daily.write_forecast(ens.predict(inp, pd.DatetimeIndex([d]))[0], d, tmp_path)
    perf = daily.score(tmp_path, inp["price"])
    assert len(perf) == 2 and (perf["coverage_80"] <= 1).all()
    assert daily.needs_retrain(model_dir, pd.Timestamp("2025-01-01")) is True
    assert daily.needs_retrain(model_dir, pd.Timestamp("2023-10-28")) is False
