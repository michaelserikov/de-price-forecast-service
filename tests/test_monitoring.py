import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dpfs import checks, drift, tracking
from dpfs.service import create_app


def _copy(inp):
    return {k: v.copy() for k, v in inp.items()}


def test_checks_pass_on_clean_inputs(inp):
    r = checks.check_inputs(inp, pd.Timestamp("2023-12-01"))
    assert r["ok"] and r["errors"] == []


def test_checks_catch_missing_hours_bad_prices_and_missing_fuels(inp):
    d = pd.Timestamp("2023-12-01")
    x = _copy(inp)
    x["res"].loc[d, 5] = np.nan
    x["price"].loc[d - pd.Timedelta(days=7), 3] = 9999.0
    x["gas"].loc[d] = np.nan
    r = checks.check_inputs(x, d)
    assert not r["ok"]
    joined = " ".join(r["errors"])
    assert "res: 1 of 24 hours missing" in joined and "lag 7" not in joined and "outside" in joined and "gas" in joined


def test_checks_warn_on_a_jump_in_the_tso_forecast(inp):
    d = pd.Timestamp("2023-12-01")
    x = _copy(inp)
    x["load"].loc[d, 12] += 60_000
    x["load"].loc[d, 12] = min(x["load"].loc[d, 12], 99_000)
    r = checks.check_inputs(x, d)
    assert r["ok"] and any("load" in w for w in r["warnings"])


def test_psi_is_near_zero_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert drift.psi(a, rng.normal(size=2000)) < 0.02
    assert drift.psi(a, rng.normal(1.5, 1, size=2000)) > drift.PSI_SIGNIFICANT


def test_performance_alarm_needs_enough_days_and_a_large_ratio():
    perf = pd.DataFrame({"day": pd.date_range("2025-01-01", periods=10).strftime("%Y-%m-%d"), "mae": 30.0})
    assert drift.performance_drift(perf.head(5), 13.2)["alarm"] is False          # too few days
    assert drift.performance_drift(perf, 13.2)["alarm"] is True
    assert drift.performance_drift(perf.assign(mae=14.0), 13.2)["alarm"] is False


def test_drift_report_asks_for_retraining_when_gas_jumps(inp):
    x = _copy(inp)
    days = x["price"].index
    x["gas"].loc[days[-30:]] = 80.0
    rep = drift.report(x, days[:300], days[-28:], pd.DataFrame(columns=["day", "mae"]), 13.2)
    assert rep["retrain"] and "gas" in rep["reasons"]
    stable = drift.report(inp, days[:300], days[300:420], pd.DataFrame(columns=["day", "mae"]), 13.2)
    assert stable["features"]["load"]["status"] == "stable"


def test_monitoring_endpoint_reads_the_daily_report(inp, tmp_path):
    (tmp_path / "monitoring.json").write_text(json.dumps({"day": "2023-12-02", "retrained": False}))
    pd.DataFrame({"day": ["2023-12-01", "2023-12-02"], "mae": [10.0, 14.0], "pinball": [3.0, 4.0],
                  "coverage_80": [0.75, 0.70]}).to_csv(tmp_path / "performance.csv", index=False)
    client = TestClient(create_app(tmp_path / "no_model", provider=lambda d: inp, forecast_dir=tmp_path))
    r = client.get("/monitoring").json()
    assert r["report"]["day"] == "2023-12-02" and r["live"]["mae"] == 12.0 and r["live"]["days"] == 2


def test_forecast_rejects_inputs_that_fail_the_checks(inp, tmp_path):
    from dpfs import models
    days = inp["price"].index
    folder = tmp_path / "m"
    models.Ensemble([models.fit("mlp", inp, days[7:200], epochs=5)]).save(folder)
    x = _copy(inp)
    x["price"].loc[pd.Timestamp("2023-11-30"), 0] = -900.0
    client = TestClient(create_app(folder, provider=lambda d: x))
    r = client.get("/forecast", params={"day": "2023-12-01"})
    assert r.status_code == 422 and "price" in " ".join(r.json()["detail"]["errors"])


@pytest.mark.skipif(not tracking.enabled(), reason="MLflow not installed")
def test_tracking_logs_a_training_run(inp, tmp_path, monkeypatch):
    from dpfs import models
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("DPFS_MLFLOW_ARTIFACTS", str(tmp_path / "artifacts"))
    folder = tmp_path / "m"
    models.Ensemble([models.fit("mlp", inp, inp["price"].index[7:120], epochs=3)]).save(folder)
    run_id = tracking.log_training(folder, {"train_end": "2023-05-01", "window_days": 113}, extra={"drift": {"retrain": False}},
                                   experiment="test")
    import mlflow
    run = mlflow.get_run(run_id)
    assert run.data.params["window_days"] == "113"
    assert any(k.startswith("val_pinball_mlp") for k in run.data.metrics)
