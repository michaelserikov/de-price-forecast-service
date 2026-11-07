"""Data-quality checks on the model inputs before a forecast is made.

Errors stop the forecast (the service answers 422, the daily job exits); warnings are reported and logged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import PRICE_LAGS

PRICE_BOUNDS = (-500.0, 5000.0)       # EUR/MWh, plausibility range for German day-ahead prices
LOAD_BOUNDS = (20_000.0, 100_000.0)   # MW, TSO day-ahead load forecast
RES_BOUNDS = (0.0, 150_000.0)         # MW, TSO day-ahead wind and solar forecast
SPIKE_SD = 8.0                        # robust standard deviations for a warning on day-to-day jumps
EUA_MAX_AGE_DAYS = 14                 # auctions run on most working days
GAS_MAX_AGE_MONTHS = 1                # the model uses the previous month; one month of publication lag is normal


def _missing(df: pd.DataFrame, day: pd.Timestamp) -> int:
    if day not in df.index:
        return 24
    return int(df.loc[day].isna().sum())


def check_inputs(inp: dict, day: pd.Timestamp) -> dict:
    """Checks for forecasting `day`: TSO forecasts for the day, price history for the lags, fuel prices."""
    day = pd.Timestamp(day)
    errors, warnings = [], []
    for name, bounds in (("load", LOAD_BOUNDS), ("res", RES_BOUNDS)):
        miss = _missing(inp[name], day)
        if miss:
            errors.append(f"{name}: {miss} of 24 hours missing for {day.date()}")
            continue
        v = inp[name].loc[day].to_numpy(dtype=float)
        if (v < bounds[0]).any() or (v > bounds[1]).any():
            errors.append(f"{name}: values outside {bounds[0]:g} to {bounds[1]:g} MW on {day.date()}")
    for lag in PRICE_LAGS:
        d = day - pd.Timedelta(days=lag)
        miss = _missing(inp["price"], d)
        if miss:
            errors.append(f"price: {miss} of 24 hours missing on {d.date()} (lag {lag})")
            continue
        p = inp["price"].loc[d].to_numpy(dtype=float)
        if (p < PRICE_BOUNDS[0]).any() or (p > PRICE_BOUNDS[1]).any():
            errors.append(f"price: values outside {PRICE_BOUNDS[0]:g} to {PRICE_BOUNDS[1]:g} EUR/MWh on {d.date()}")
    for name in ("eua", "gas"):
        s = inp[name]
        if day not in s.index or pd.isna(s.loc[day]):
            errors.append(f"{name}: no value for {day.date()}")
        elif s.loc[day] <= 0:
            errors.append(f"{name}: non-positive value {s.loc[day]:g}")
    # fuel prices carried forward from the last publication: the model does not see the current level
    age = inp.get("eua_age_days")
    if age is not None and day in age.index and pd.notna(age.loc[day]) and age.loc[day] > EUA_MAX_AGE_DAYS:
        warnings.append(f"eua: last auction price is {int(age.loc[day])} days old and carried forward")
    age = inp.get("gas_age_months")
    if age is not None and day in age.index and pd.notna(age.loc[day]) and age.loc[day] > GAS_MAX_AGE_MONTHS:
        warnings.append(f"gas: last published monthly price is carried forward by {int(age.loc[day])} months")
    # day-to-day jumps in the TSO forecasts, measured against the last 60 days
    for name in ("load", "res"):
        df = inp[name]
        prev = day - pd.Timedelta(days=1)
        hist = df.loc[:prev].tail(60)
        if day in df.index and prev in df.index and len(hist) > 14 and not df.loc[[prev, day]].isna().any().any():
            diffs = hist.diff().abs().to_numpy(dtype=float)
            scale = 1.4826 * float(np.nanmedian(diffs)) if np.isfinite(diffs).any() else 0.0
            jump = float((df.loc[day] - df.loc[prev]).abs().max())
            if scale > 0 and jump > SPIKE_SD * scale:
                warnings.append(f"{name}: hourly change of {jump:,.0f} MW from the day before, unusually large")
    return {"day": str(day.date()), "ok": not errors, "errors": errors, "warnings": warnings}
