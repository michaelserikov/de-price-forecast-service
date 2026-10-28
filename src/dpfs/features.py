"""Daily inputs and feature tensors for day-ahead price forecasting in DE-LU.

Every forecast for the 24 hours of day d uses information of day d-1 only: prices up to d-1, the
TSOs' day-ahead forecasts of load and of wind and solar generation for d, the calendar, the last
EU allowance auction price before d and the previous month's gas price. This is the information
set of the open benchmarks of the price forecasting literature (Lago, Marcjasz, De Schutter and
Weron, 2021) and of the LEAR and gradient boosting models in de-power-market-ppa v0.5, so the
neural networks here are compared with them on identical inputs.

Note on timing: SMARD publishes the TSO wind and solar forecasts for d around 18:00 on d-1, after
the 12:00 day-ahead auction. The literature uses the same series; a trading desk would replace
them with its own weather-based forecasts available before noon.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TZ = "Europe/Berlin"
HOURS = list(range(24))
PRICE_LAGS = (1, 2, 3, 7)
EXO_LAGS = (0, 1, 7)


def daily_matrix(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Local days x 24 local hours of one hourly column (clock-change days fixed by interpolation)."""
    local = df["ts_utc"].dt.tz_convert(TZ)
    g = pd.DataFrame({"day": local.dt.tz_localize(None).dt.normalize(), "hour": local.dt.hour, "v": df[col].to_numpy()})
    m = g.groupby(["day", "hour"])["v"].mean().unstack("hour").reindex(columns=HOURS)
    return m.interpolate(axis=1, limit_direction="both")


def build_inputs(hourly: pd.DataFrame, eua: pd.DataFrame, gas_monthly: pd.Series,
                 last_day: pd.Timestamp | None = None) -> dict:
    """Daily matrices of price, load and renewable forecasts plus the fuel signals known before each day.

    Days run up to `last_day` (default: the last day with forecasts). Fuel signals are carried
    forward when the newest published value is older than the day (EEX archive, Pink Sheet lag).
    """
    df = hourly.copy()
    df["res_fc"] = df["onshore_fc"] + df["offshore_fc"] + df["solar_fc"]
    load = daily_matrix(df, "load_fc")
    res = daily_matrix(df, "res_fc")
    price = daily_matrix(df.dropna(subset=["price"]), "price")
    end = last_day if last_day is not None else load.dropna().index.max()
    days = pd.date_range(min(price.index.min(), load.index.min()), end, freq="D")
    price = price.reindex(days)
    load = load.reindex(days).interpolate(limit_direction="both")
    res = res.reindex(days).interpolate(limit_direction="both")
    e = eua.sort_values("date").assign(known_from=lambda x: (x["date"] + pd.Timedelta(days=1)).astype("datetime64[ns]"))
    left = pd.DataFrame({"day": days.astype("datetime64[ns]")})
    eua_known = pd.merge_asof(left, e[["known_from", "price"]], left_on="day", right_on="known_from",
                              direction="backward")["price"].set_axis(days).ffill()
    prev = days.to_period("M") - 1
    gas = pd.Series(gas_monthly.reindex(prev).to_numpy(), index=days).ffill()
    return {"price": price, "load": load, "res": res, "eua": eua_known, "gas": gas}


def tabular(inp: dict, days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Flat feature matrix (days x 249) as in LEAR, and a mask of its price columns."""
    p, ld, rs = inp["price"], inp["load"], inp["res"]
    blocks, is_price = [], []
    for lag in PRICE_LAGS:
        blocks.append(p.reindex(days - pd.Timedelta(days=lag)).to_numpy())
        is_price += [True] * 24
    for m in (ld, rs):
        for lag in EXO_LAGS:
            blocks.append(m.reindex(days - pd.Timedelta(days=lag)).to_numpy())
            is_price += [False] * 24
    dow = np.zeros((len(days), 7))
    dow[np.arange(len(days)), days.day_of_week] = 1
    blocks += [dow, inp["eua"].reindex(days).to_numpy()[:, None], inp["gas"].reindex(days).to_numpy()[:, None]]
    is_price += [False] * 9
    return np.hstack(blocks), np.array(is_price)


SEQ_FEATURES = (["p_lag1", "p_lag2", "p_lag3", "p_lag7", "p1_mean", "p1_min", "p1_max",
                 "load", "res", "resid", "load_lag1", "res_lag1", "load_lag7", "res_lag7",
                 "hour_sin", "hour_cos"] + [f"dow{i}" for i in range(7)] + ["eua", "gas"])
SEQ_PRICE = np.array([f.startswith("p") for f in SEQ_FEATURES])


def sequence(inp: dict, days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Feature tensor (days x 24 hours x 25) for a sequence model over the hours of the day."""
    p, ld, rs = inp["price"], inp["load"], inp["res"]
    n = len(days)
    lag = lambda m, k: m.reindex(days - pd.Timedelta(days=k)).to_numpy()  # noqa: E731
    prev = lag(p, 1)
    hours = np.arange(24)
    cols = [lag(p, 1), lag(p, 2), lag(p, 3), lag(p, 7),
            np.repeat(np.nanmean(prev, axis=1, keepdims=True), 24, 1) if n else prev,
            np.repeat(np.nanmin(prev, axis=1, keepdims=True), 24, 1) if n else prev,
            np.repeat(np.nanmax(prev, axis=1, keepdims=True), 24, 1) if n else prev,
            lag(ld, 0), lag(rs, 0), lag(ld, 0) - lag(rs, 0), lag(ld, 1), lag(rs, 1), lag(ld, 7), lag(rs, 7),
            np.tile(np.sin(2 * np.pi * hours / 24), (n, 1)), np.tile(np.cos(2 * np.pi * hours / 24), (n, 1))]
    dow = np.zeros((n, 7))
    dow[np.arange(n), days.day_of_week] = 1
    cols += [np.repeat(dow[:, [i]], 24, 1) for i in range(7)]
    cols += [np.repeat(inp["eua"].reindex(days).to_numpy()[:, None], 24, 1),
             np.repeat(inp["gas"].reindex(days).to_numpy()[:, None], 24, 1)]
    return np.stack(cols, axis=-1), SEQ_PRICE.copy()


@dataclass
class Scaler:
    """Price columns and target: asinh((x - median) / MAD); then every column standardised."""
    med: float
    mad: float
    mu: list
    sd: list

    @classmethod
    def fit(cls, X: np.ndarray, is_price: np.ndarray, Y: np.ndarray) -> "Scaler":
        med = float(np.median(Y))
        mad = float(np.median(np.abs(Y - med))) * 1.4826 or 1.0
        Xt = _price_transform(X, is_price, med, mad)
        axes = tuple(range(Xt.ndim - 1))
        mu, sd = Xt.mean(axis=axes), Xt.std(axis=axes)
        sd[sd == 0] = 1.0
        return cls(med, mad, mu.tolist(), sd.tolist())

    def x(self, X: np.ndarray, is_price: np.ndarray) -> np.ndarray:
        return ((_price_transform(X, is_price, self.med, self.mad) - np.array(self.mu)) / np.array(self.sd)).astype(np.float32)

    def y(self, Y: np.ndarray) -> np.ndarray:
        return np.arcsinh((Y - self.med) / self.mad).astype(np.float32)

    def y_inv(self, Z: np.ndarray) -> np.ndarray:
        return np.sinh(Z) * self.mad + self.med

    def to_dict(self) -> dict:
        return asdict(self)


def _price_transform(X: np.ndarray, is_price: np.ndarray, med: float, mad: float) -> np.ndarray:
    Xt = X.astype(float).copy()
    Xt[..., is_price] = np.arcsinh((Xt[..., is_price] - med) / mad)
    return Xt
