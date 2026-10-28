"""Synthetic inputs with a daily shape, a weekly cycle and a renewable effect; no network access."""

import numpy as np
import pandas as pd
import pytest

from dpfs.features import HOURS


def synthetic_inputs(n_days: int = 420, seed: int = 0, start: str = "2023-01-01") -> dict:
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=n_days, freq="D")
    h = np.arange(24)
    load = 55_000 + 10_000 * np.sin((h - 6) / 24 * 2 * np.pi)[None] + rng.normal(0, 1500, (n_days, 24))
    load[days.day_of_week >= 5] -= 8_000
    res = np.clip(rng.gamma(2.0, 9_000, (n_days, 1)) + 25_000 * np.exp(-((h - 13) ** 2) / 8)[None]
                  * rng.uniform(0.2, 1.0, (n_days, 1)), 0, None)
    eua = pd.Series(70 + np.cumsum(rng.normal(0, 0.5, n_days)), index=days)
    gas = pd.Series(35.0, index=days)
    price = 20 + 1.6 * gas.to_numpy()[:, None] + 0.0025 * (load - res) + rng.normal(0, 6, (n_days, 24))
    mk = lambda a: pd.DataFrame(a, index=days, columns=HOURS)  # noqa: E731
    return {"price": mk(price), "load": mk(load), "res": mk(res), "eua": eua, "gas": gas}


@pytest.fixture(scope="session")
def inp():
    return synthetic_inputs()
