"""Walk-forward backtest 2024-2025: PyTorch MLP and LSTM against the naive benchmark and LEAR+GBM.

    python -m dpfs.backtest                         # full study, monthly recalibration
    python -m dpfs.backtest --baseline ../de-power-market-ppa/results/forecast

Each month of the test period is forecast by networks trained only on the 728 days before it
(rolling window, monthly recalibration, three seeds per architecture averaged). When --baseline
points to the results of de-power-market-ppa v0.5, its LEAR+GBM forecasts for the same days are
compared directly (MAE, Diebold-Mariano). Writes results/backtest/ and figures/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from . import evaluate as ev, models
from .data import DATA, ROOT, load_inputs
from .features import HOURS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEST_START, TEST_END, DATA_START = "2024-01-01", "2025-12-31", "2021-01-01"


def walk_forward(inp: dict, days: pd.DatetimeIndex, kind: str, window: int = 728, seeds=(0, 1, 2),
                 **fit_kw) -> np.ndarray:
    """Quantile forecasts (days, 24, 3) with monthly recalibration on the `window` days before each month."""
    out = np.full((len(days), 24, len(models.QUANTILES)), np.nan)
    for _, mdays in pd.Series(days, index=days).groupby(days.to_period("M")):
        first = mdays.iloc[0]
        cal = pd.date_range(first - pd.Timedelta(days=window), first - pd.Timedelta(days=1), freq="D")
        ens = models.Ensemble([models.fit(kind, inp, cal, seed=s, **fit_kw) for s in seeds])
        pos = days.get_indexer(pd.DatetimeIndex(mdays.to_numpy()))
        out[pos] = ens.predict(inp, pd.DatetimeIndex(mdays.to_numpy()))
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", type=Path, default=ROOT.parent / "de-power-market-ppa" / "results" / "forecast")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args(argv)
    out, figs = ROOT / "results" / "backtest", ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(exist_ok=True)

    inp = load_inputs(DATA_START, TEST_END, DATA)
    days = pd.date_range(TEST_START, TEST_END, freq="D")
    actual = inp["price"].loc[days]
    q, point, runtime = {}, {"naive": ev.naive(inp["price"], days)}, {}
    for kind in ("mlp", "lstm"):
        t0 = time.time()
        q[kind] = walk_forward(inp, days, kind, seeds=tuple(range(args.seeds)))
        runtime[kind] = round(time.time() - t0, 1)
        point[kind] = pd.DataFrame(q[kind][..., 1], index=days, columns=HOURS)
        np.save(out / f"quantiles_{kind}.npy", q[kind])
    point["mlp_lstm_avg"] = (point["mlp"] + point["lstm"]) / 2
    q["mlp_lstm_avg"] = np.sort((q["mlp"] + q["lstm"]) / 2, axis=-1)
    base_file = args.baseline / "forecast_lear_gbm_avg.csv"
    if base_file.exists():
        b = pd.read_csv(base_file, index_col=0, parse_dates=True)
        b.columns = HOURS
        point["lear_gbm"] = b.reindex(days)
        point["lear_gbm_mlp_lstm"] = (point["lear_gbm"] + point["mlp_lstm_avg"]) / 2
    for name, f in point.items():
        f.to_csv(out / f"forecast_{name}.csv", float_format="%.3f")

    rows = []
    for period, sel in (("2024", days.year == 2024), ("2025", days.year == 2025), ("2024-2025", np.ones(len(days), bool))):
        for name, f in point.items():
            r = {"period": period, "model": name, **ev.metrics(actual[sel], f[sel], point["naive"][sel])}
            if name in q:
                a = actual[sel].to_numpy()
                r["pinball"] = ev.pinball(a, q[name][sel])
                r["coverage_80"] = ev.coverage(a, q[name][sel][..., 0], q[name][sel][..., 2])
            rows.append(r)
    met = pd.DataFrame(rows)
    met.to_csv(out / "metrics.csv", index=False, float_format="%.4f")
    names = list(point)
    dm = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for c in names:
            if a != c:
                dm.loc[a, c] = ev.dm_test(actual, point[a], point[c])[1]   # p-value: column better than row
    dm.to_csv(out / "dm_pvalues.csv", float_format="%.4f")
    summary = {"test_period": [TEST_START, TEST_END], "days": len(days), "seeds": args.seeds,
               "runtime_seconds": runtime, "metrics": met.round(3).to_dict(orient="records"),
               "dm_pvalues_column_better_than_row": dm.round(4).to_dict()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4))
    for name in [n for n in ("naive", "lear_gbm", "mlp", "lstm", "lear_gbm_mlp_lstm") if n in point]:
        ax.plot(HOURS, (point[name] - actual).abs().mean().to_numpy(), label=name)
    ax.set(xlabel="hour (local)", ylabel="MAE, EUR/MWh", title="Day-ahead price forecast error by hour, 2024-2025")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "mae_by_hour.png", dpi=150)
    week = pd.date_range("2025-05-05", periods=7, freq="D")
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(7 * 24)
    ax.plot(x, actual.loc[week].to_numpy().ravel(), color="black", label="actual")
    qm = q["mlp_lstm_avg"][days.get_indexer(week)]
    ax.plot(x, qm[..., 1].ravel(), label="MLP+LSTM median")
    ax.fill_between(x, qm[..., 0].ravel(), qm[..., 2].ravel(), alpha=0.3, label="10-90% band")
    ax.set(xlabel="hour of the week (5-11 May 2025)", ylabel="EUR/MWh", title="Probabilistic forecast, example week")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "example_week.png", dpi=150)
    return summary


if __name__ == "__main__":
    main()
