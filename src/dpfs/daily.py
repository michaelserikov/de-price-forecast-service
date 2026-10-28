"""Daily job: refresh data, retrain when the model is older than a month, forecast tomorrow, score yesterday.

    python -m dpfs.daily                 # run by the GitHub Actions schedule every evening (UTC)
    python -m dpfs.daily --day 2026-10-29 --retrain

Writes forecasts/<day>.csv (24 hours x 3 quantiles), updates models/latest when retrained and appends
the realised error of every past forecast whose prices are published to forecasts/performance.csv,
so live accuracy can be compared with the backtest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev, models
from .data import DATA, ROOT, load_inputs

WINDOW = 728
RETRAIN_AFTER_DAYS = 30
KINDS = ("mlp", "lstm")


def needs_retrain(model_dir: Path, today: pd.Timestamp) -> bool:
    metas = list(model_dir.glob("*/meta.json"))
    if not metas:
        return True
    end = max(pd.Timestamp(json.loads(p.read_text(encoding="utf-8"))["train_end"]) for p in metas)
    return (today - end).days > RETRAIN_AFTER_DAYS


def train(inp: dict, today: pd.Timestamp, seeds=(0, 1, 2)) -> models.Ensemble:
    cal = pd.date_range(today - pd.Timedelta(days=WINDOW), today - pd.Timedelta(days=1), freq="D")
    return models.Ensemble([models.fit(k, inp, cal, seed=s) for k in KINDS for s in seeds])


def write_forecast(q: np.ndarray, day: pd.Timestamp, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"hour": range(24), "q10": q[:, 0], "q50": q[:, 1], "q90": q[:, 2]})
    path = folder / f"{day.date()}.csv"
    df.to_csv(path, index=False, float_format="%.2f")
    return path


def score(folder: Path, price: pd.DataFrame) -> pd.DataFrame:
    """Realised MAE, pinball and 80% coverage of every stored forecast whose prices are known."""
    rows = []
    for path in sorted(folder.glob("????-??-??.csv")):
        day = pd.Timestamp(path.stem)
        if day not in price.index or price.loc[day].isna().any():
            continue
        f = pd.read_csv(path)
        a = price.loc[day].to_numpy()
        q = f[["q10", "q50", "q90"]].to_numpy()
        rows.append({"day": path.stem, "mae": float(np.abs(q[:, 1] - a).mean()),
                     "pinball": ev.pinball(a[None], q[None]), "coverage_80": ev.coverage(a, q[:, 0], q[:, 2])})
    perf = pd.DataFrame(rows, columns=["day", "mae", "pinball", "coverage_80"])
    perf.to_csv(folder / "performance.csv", index=False, float_format="%.3f")
    return perf


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--day", help="delivery day to forecast (default: tomorrow, Europe/Berlin)")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args(argv)
    today = pd.Timestamp.now(tz="Europe/Berlin").normalize().tz_localize(None)
    day = pd.Timestamp(args.day) if args.day else today + pd.Timedelta(days=1)
    inp = load_inputs((day - pd.Timedelta(days=WINDOW + 10)).date().isoformat(), day.date().isoformat(), DATA)
    model_dir = ROOT / "models" / "latest"
    retrained = args.retrain or needs_retrain(model_dir, day)
    if retrained:
        ens = train(inp, day)
        for old in model_dir.glob("*/"):
            for f in old.iterdir():
                f.unlink()
            old.rmdir()
        ens.save(model_dir)
    else:
        ens = models.Ensemble.load(model_dir)
    if inp["load"].loc[day].isna().any():
        raise SystemExit(f"TSO forecasts for {day.date()} not published yet")
    q = ens.predict(inp, pd.DatetimeIndex([day]))[0]
    path = write_forecast(q, day, ROOT / "forecasts")
    perf = score(ROOT / "forecasts", inp["price"])
    result = {"day": str(day.date()), "forecast": str(path.relative_to(ROOT)), "retrained": retrained,
              "scored_days": int(len(perf)), "live_mae": float(perf["mae"].mean()) if len(perf) else None}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
