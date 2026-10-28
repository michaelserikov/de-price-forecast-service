"""Point and probabilistic evaluation of day-ahead price forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .features import HOURS


def naive(price: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Day d-1 prices for Tuesday to Friday, day d-7 prices for Monday, Saturday and Sunday."""
    out = [price.loc[d - pd.Timedelta(days=7 if d.day_of_week in (0, 5, 6) else 1)].to_numpy() for d in days]
    return pd.DataFrame(out, index=days, columns=HOURS)


def metrics(actual: pd.DataFrame, fc: pd.DataFrame, naive_fc: pd.DataFrame) -> dict:
    e = (fc - actual).to_numpy()
    en = (naive_fc - actual).to_numpy()
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "rmae": float(np.abs(e).mean() / np.abs(en).mean()), "bias": float(e.mean())}


def dm_test(actual: pd.DataFrame, fc_a: pd.DataFrame, fc_b: pd.DataFrame) -> tuple[float, float]:
    """Diebold-Mariano on the daily L1 norm of errors; one-sided p-value for H1: b is more accurate than a."""
    d = ((fc_a - actual).abs().sum(axis=1) - (fc_b - actual).abs().sum(axis=1)).to_numpy()
    stat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(stat), float(1 - norm.cdf(stat))


def pinball(actual: np.ndarray, q_fc: np.ndarray, quantiles=(0.1, 0.5, 0.9)) -> float:
    """Mean pinball loss in EUR/MWh; actual (days, 24), q_fc (days, 24, n_q)."""
    e = actual[..., None] - q_fc
    q = np.asarray(quantiles)
    return float(np.maximum(q * e, (q - 1) * e).mean())


def coverage(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Share of hours inside [lo, hi]."""
    return float(((actual >= lo) & (actual <= hi)).mean())
