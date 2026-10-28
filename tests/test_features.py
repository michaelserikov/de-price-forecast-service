import numpy as np
import pandas as pd

from dpfs import features as ft, smard


def test_tabular_has_lear_inputs(inp):
    days = inp["price"].index[10:20]
    X, is_price = ft.tabular(inp, days)
    assert X.shape == (10, 249)
    assert is_price.sum() == 96
    # first block is the price of d-1
    np.testing.assert_allclose(X[0, :24], inp["price"].loc[days[0] - pd.Timedelta(days=1)].to_numpy())


def test_sequence_shape_and_alignment(inp):
    days = inp["price"].index[10:15]
    X, is_price = ft.sequence(inp, days)
    assert X.shape == (5, 24, len(ft.SEQ_FEATURES))
    assert is_price.sum() == 7
    k = ft.SEQ_FEATURES.index("load")
    np.testing.assert_allclose(X[2, :, k], inp["load"].loc[days[2]].to_numpy())


def test_scaler_roundtrip(inp):
    days = inp["price"].index[10:200]
    X, is_price = ft.tabular(inp, days)
    Y = inp["price"].loc[days].to_numpy()
    s = ft.Scaler.fit(X, is_price, Y)
    np.testing.assert_allclose(s.y_inv(s.y(Y)), Y, rtol=1e-4, atol=1e-3)
    Xs = s.x(X, is_price)
    assert np.allclose(Xs.mean(axis=0), 0, atol=1e-4)


def test_build_inputs_carries_fuel_forward():
    ts = pd.date_range("2025-12-28", "2026-01-04", freq="h", tz="UTC")
    hourly = pd.DataFrame({"ts_utc": ts, "price": 80.0, "load_fc": 50_000.0, "onshore_fc": 10_000.0,
                           "offshore_fc": 2_000.0, "solar_fc": 1_000.0})
    eua = pd.DataFrame({"date": pd.to_datetime(["2025-12-29"]), "price": [75.0]})
    gas = pd.Series([30.0], index=pd.PeriodIndex(["2025-11"], freq="M"))
    inp = ft.build_inputs(hourly, eua, gas, last_day=pd.Timestamp("2026-01-04"))
    assert inp["eua"].loc["2026-01-03"] == 75.0          # beyond the archive: last auction carried forward
    assert inp["gas"].loc["2026-01-03"] == 30.0          # December not yet published: November carried forward
    assert inp["res"].loc["2026-01-02"].iloc[0] == 13_000.0


def test_parse_chunk_drops_nulls():
    doc = {"series": [[1704067200000, 50.0], [1704070800000, None]]}
    df = smard.parse_chunk(doc)
    assert len(df) == 1 and df["ts_utc"].iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")
