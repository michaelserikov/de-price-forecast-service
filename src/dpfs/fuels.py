"""Fuel and carbon price signals in EUR, from open sources, cached on disk.

- Natural gas, Europe (TTF) from the World Bank Commodity Price Data ("Pink Sheet"), monthly, USD.
- EUR/USD monthly average reference rate from the ECB data API.
- EU emission allowances: clearing prices of the EEX primary auctions (daily).

Raw files go to data/raw/fuels/ and are not committed. Conversion: 1 MWh = 3.412 MMBtu.
The EEX archive used here ends in 2025 and the Pink Sheet is published with a lag of about a
month; for later days the last published value is carried forward (see features.build_inputs).

Adapted from github.com/michaelserikov/de-power-market-ppa (same author).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

WB_URL = ("https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59263ba0a2-0050012025/"
          "related/CMO-Historical-Data-Monthly.xlsx")
ECB_URL = ("https://data-api.ecb.europa.eu/service/data/EXR/M.USD.EUR.SP00.A"
           "?format=csvdata&startPeriod={start}&endPeriod={end}")
EEX_URL = ("https://www.eex.com/fileadmin/EEX/Downloads/Markets/Environmentals/"
           "EUA_Emission_Spot_Primary_Market_Auction_Report/Archive_Reports/"
           "emission-spot-primary-market-auction-report-2012-2025-data.zip")

MMBTU_PER_MWH = 3.412

_HEADERS = {"User-Agent": "de-price-forecast-service (research; github.com/michaelserikov/de-price-forecast-service)"}


def _cached(url: str, path: Path) -> bytes:
    if not path.exists():
        r = requests.get(url, headers=_HEADERS, timeout=120)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    return path.read_bytes()


def gas_monthly(cache: Path, start: str, end: str) -> pd.Series:
    """Monthly TTF gas price in EUR per MWh (World Bank Pink Sheet converted with the ECB rate)."""
    raw = pd.read_excel(io.BytesIO(_cached(WB_URL, cache / "wb_pink_sheet_monthly.xlsx")),
                        sheet_name="Monthly Prices", header=None)
    header = raw.iloc[4].tolist()
    gas_col = header.index("Natural gas, Europe")
    rows = raw.iloc[6:, [0, gas_col]]
    rows.columns = ["period", "gas_usd_mmbtu"]
    rows = rows[rows["period"].astype(str).str.match(r"^\d{4}M\d{2}$")].copy()
    rows["gas_usd_mmbtu"] = pd.to_numeric(rows["gas_usd_mmbtu"], errors="coerce")
    rows["month"] = pd.PeriodIndex(rows["period"].str.replace("M", "-"), freq="M")
    gas = rows.set_index("month")["gas_usd_mmbtu"]
    fx_raw = _cached(ECB_URL.format(start=start, end=end), cache / f"ecb_usd_eur_{start}_{end}.csv")
    fx_df = pd.read_csv(io.BytesIO(fx_raw))
    fx = pd.Series(fx_df["OBS_VALUE"].to_numpy(float), index=pd.PeriodIndex(fx_df["TIME_PERIOD"], freq="M"))
    months = pd.period_range(start, end, freq="M")
    out = gas.reindex(months) / fx.reindex(months).ffill() * MMBTU_PER_MWH
    return out.dropna().rename("gas_eur_mwh")


def eua_auctions(cache: Path, years: range) -> pd.DataFrame:
    """Daily EEX primary-auction clearing prices of EU allowances (EUR/t); years missing from the archive are skipped."""
    z = zipfile.ZipFile(io.BytesIO(_cached(EEX_URL, cache / "eex_eua_auctions_2012_2025.zip")))
    frames = []
    for year in years:
        names = [n for n in z.namelist() if n.endswith(f"-{year}-data.xlsx")]
        if not names:
            continue
        raw = pd.read_excel(io.BytesIO(z.read(names[0])), header=None)
        header_row = next(i for i, v in enumerate(raw.iloc[:, 1]) if str(v).strip() == "Date")
        cols = [str(c).strip() for c in raw.iloc[header_row]]
        price_col = next(i for i, c in enumerate(cols) if c.startswith("Auction Price"))
        status_col = cols.index("Status")
        df = raw.iloc[header_row + 1:, [1, price_col, status_col]]
        df.columns = ["date", "price", "status"]
        df = df[[str(v).strip().lower() == "successful" for v in df["status"]]]
        frames.append(df[["date", "price"]])
    out = pd.concat(frames)
    out["date"] = pd.to_datetime(out["date"])
    out["price"] = out["price"].astype(float)
    return out.sort_values("date").reset_index(drop=True)
