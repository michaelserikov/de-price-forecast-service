"""Monitoring: input drift (population stability index against the training window) and live accuracy against
the backtest. Either alarm makes the daily job retrain before forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_MODERATE, PSI_SIGNIFICANT = 0.10, 0.25
LIVE_WINDOW = 14            # days of live forecasts in the accuracy check
LIVE_MIN_DAYS = 7
LIVE_RATIO_ALARM = 1.5      # live MAE above 1.5 times the backtest MAE


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index with quantile bins of the expected sample; half a count is added to every bin
    (Laplace smoothing) so that small samples with empty bins do not inflate the index."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    if len(expected) < bins or len(actual) == 0:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                       # (almost) constant in the reference window: at or above it, or below
        v = edges[0]
        edges = np.array([-np.inf, v - 1e-9 * (abs(v) + 1), v + 1e-9 * (abs(v) + 1), np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    ce = np.histogram(expected, edges)[0] + 0.5
    ca = np.histogram(actual, edges)[0] + 0.5
    e, a = ce / ce.sum(), ca / ca.sum()
    return float(((a - e) * np.log(a / e)).sum())


def features(inp: dict) -> dict:
    """Hourly matrices (days x 24) for price, load, renewables and residual load; daily series for fuels."""
    return {"price": inp["price"], "load": inp["load"], "res": inp["res"], "residual_load": inp["load"] - inp["res"],
            "eua": inp["eua"], "gas": inp["gas"]}


def _values(x, days: pd.DatetimeIndex) -> np.ndarray:
    return x.reindex(days).to_numpy(dtype=float).ravel()


def seasonal_reference(train: pd.DatetimeIndex, recent: pd.DatetimeIndex, half_width: int = 30) -> pd.DatetimeIndex:
    """Training days within half_width days of the season of the recent window (any year), so that load and
    renewables are compared with the same time of year and the seasonal cycle does not raise an alarm."""
    mid = recent[len(recent) // 2].day_of_year
    d = np.abs(train.day_of_year - mid)
    d = np.minimum(d, 365 - d)
    ref = train[d <= half_width + len(recent) // 2]
    return ref if len(ref) >= 20 else train


def feature_drift(inp: dict, train: pd.DatetimeIndex, recent: pd.DatetimeIndex) -> dict:
    """Hourly series: PSI against the same season of the training window. Fuel prices are levels that move over
    years, where PSI says little: they are flagged when the recent mean leaves the 5-95% range of training."""
    season = seasonal_reference(train, recent)
    out = {}
    for c, x in features(inp).items():
        if isinstance(x, pd.DataFrame):
            v = psi(_values(x, season), _values(x, recent))
            out[c] = {"psi": round(v, 4), "status": "significant" if v > PSI_SIGNIFICANT else
                      "moderate" if v > PSI_MODERATE else "stable"}
        else:
            ref, now = _values(x, train), _values(x, recent)
            ref, now = ref[~np.isnan(ref)], now[~np.isnan(now)]
            if len(ref) < 20 or len(now) == 0:
                out[c] = {"status": "unknown"}
                continue
            lo, hi = np.quantile(ref, [0.05, 0.95])
            m = float(now.mean())
            out[c] = {"recent_mean": round(m, 2), "train_p05": round(float(lo), 2), "train_p95": round(float(hi), 2),
                      "status": "significant" if (m < lo or m > hi) else "stable"}
    return out


def performance_drift(perf: pd.DataFrame, backtest_mae: float) -> dict:
    recent = perf.sort_values("day").tail(LIVE_WINDOW)
    if len(recent) < LIVE_MIN_DAYS:
        return {"days": int(len(recent)), "live_mae": None, "ratio": None, "alarm": False}
    mae = float(recent["mae"].mean())
    ratio = mae / backtest_mae
    return {"days": int(len(recent)), "live_mae": round(mae, 3), "backtest_mae": backtest_mae,
            "ratio": round(ratio, 3), "alarm": bool(ratio > LIVE_RATIO_ALARM)}


def report(inp: dict, train: pd.DatetimeIndex, recent: pd.DatetimeIndex, perf: pd.DataFrame, backtest_mae: float) -> dict:
    feats = feature_drift(inp, train, recent)
    live = performance_drift(perf, backtest_mae)
    # the price level itself drifts with fuel prices; the alarm uses the inputs the model cannot see coming
    input_alarm = any(feats[c]["status"] == "significant" for c in ("residual_load", "gas", "eua"))
    return {"features": feats, "live": live, "retrain": bool(input_alarm or live["alarm"]),
            "reasons": [c for c in ("residual_load", "gas", "eua") if feats[c]["status"] == "significant"]
            + (["live accuracy"] if live["alarm"] else [])}
