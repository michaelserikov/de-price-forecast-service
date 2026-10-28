"""Download hourly time series from SMARD (Bundesnetzagentur), cached on disk.

SMARD publishes each series as weekly JSON chunks:
  index:  https://www.smard.de/app/chart_data/{filter}/{region}/index_hour.json
  chunk:  https://www.smard.de/app/chart_data/{filter}/{region}/{filter}_{region}_hour_{ts}.json
Timestamps are milliseconds since the epoch (UTC); prices are EUR/MWh, load and generation MWh per
hour. Data licence: CC BY 4.0, Bundesnetzagentur | SMARD.de.

Adapted from github.com/michaelserikov/de-power-market-ppa (same author).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.smard.de/app/chart_data"
REGION = "DE-LU"

SERIES = {
    "price": 4169,        # day-ahead price, bidding zone DE-LU, EUR/MWh
    "load_fc": 411,       # TSO day-ahead forecast of grid load, MWh
    "onshore_fc": 123,    # TSO day-ahead forecast of onshore wind generation, MWh
    "offshore_fc": 3791,  # TSO day-ahead forecast of offshore wind generation, MWh
    "solar_fc": 125,      # TSO day-ahead forecast of solar generation, MWh
}

_session = requests.Session()
_session.headers["User-Agent"] = "de-price-forecast-service (research; github.com/michaelserikov/de-price-forecast-service)"


def _get_json(url: str, retries: int = 4) -> dict:
    for attempt in range(retries):
        r = _session.get(url, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    raise RuntimeError(f"failed: {url}")


def chunk_starts(filter_id: int, region: str = REGION) -> list[int]:
    """Start timestamps (ms, UTC) of all weekly chunks of a series."""
    return _get_json(f"{BASE}/{filter_id}/{region}/index_hour.json")["timestamps"]


def parse_chunk(doc: dict) -> pd.DataFrame:
    """One chunk document to a frame with columns ts_utc (tz-aware) and value; nulls dropped."""
    rows = [(t, v) for t, v in doc.get("series", []) if v is not None]
    df = pd.DataFrame(rows, columns=["ts_ms", "value"])
    df["ts_utc"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df[["ts_utc", "value"]]


def load_series(name: str, start: str, end: str, cache_dir: Path, region: str = REGION) -> pd.DataFrame:
    """Hourly series `name` between start and end (inclusive dates, UTC), from cache or SMARD.

    A chunk is cached once it lies fully in the past, so a rerun downloads only the current week.
    """
    filter_id = SERIES[name]
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    week = pd.Timedelta(days=7)
    folder = cache_dir / f"{filter_id}_{region}"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        starts = chunk_starts(filter_id, region)
    except requests.RequestException:  # SMARD unreachable: fall back to the cached weekly chunks
        starts = sorted(int(p.stem) for p in folder.glob("*.json"))
        if not starts:
            raise
    frames = []
    for ts in starts:
        t0 = pd.Timestamp(ts, unit="ms", tz="UTC")
        if t0 + week <= start_ts or t0 >= end_ts:
            continue
        path = folder / f"{ts}.json"
        if path.exists():
            doc = json.loads(path.read_text(encoding="utf-8"))
        else:
            doc = _get_json(f"{BASE}/{filter_id}/{region}/{filter_id}_{region}_hour_{ts}.json")
            if t0 + week < pd.Timestamp.now(tz="UTC"):
                path.write_text(json.dumps(doc), encoding="utf-8")
            time.sleep(0.2)  # keep the request rate low
        frames.append(parse_chunk(doc))
    df = pd.concat(frames, ignore_index=True).drop_duplicates("ts_utc")
    df = df[(df["ts_utc"] >= start_ts) & (df["ts_utc"] < end_ts)]
    return df.rename(columns={"value": name}).sort_values("ts_utc").reset_index(drop=True)


def load_hourly(start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    """Price and TSO forecasts joined on the hour; missing hours stay NaN."""
    out = None
    for name in SERIES:
        df = load_series(name, start, end, cache_dir)
        out = df if out is None else out.merge(df, on="ts_utc", how="outer")
    return out.sort_values("ts_utc").reset_index(drop=True)
