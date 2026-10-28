"""Assemble the daily model inputs from SMARD and the fuel sources (all downloads cached)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from . import features as ft, fuels, smard

ROOT = Path(os.environ.get("DPFS_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"


def load_inputs(start: str, end: str, data: Path = DATA) -> dict:
    """Daily inputs from `start` to `end` (local dates); `end` may be tomorrow (forecasts only)."""
    first = (pd.Timestamp(start) - pd.Timedelta(days=1)).date().isoformat()  # local midnight is 22/23 UTC
    hourly = smard.load_hourly(first, end, data / "raw")
    cache = data / "raw" / "fuels"
    eua = fuels.eua_auctions(cache, range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1))
    gas = fuels.gas_monthly(cache, (pd.Timestamp(start) - pd.DateOffset(months=1)).strftime("%Y-%m"),
                            pd.Timestamp(end).strftime("%Y-%m"))
    inp = ft.build_inputs(hourly, eua, gas, last_day=pd.Timestamp(end))
    days = pd.date_range(start, end, freq="D")
    return {k: v.reindex(days) for k, v in inp.items()}
