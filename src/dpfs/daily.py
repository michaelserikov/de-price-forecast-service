"""Daily job: refresh data, retrain when the model is older than a month, forecast tomorrow, score yesterday.

    python -m dpfs.daily                 # run by the GitHub Actions schedule every evening (UTC)
    python -m dpfs.daily --day 2026-10-29 --retrain

Writes forecasts/<day>.csv (24 hours x 3 quantiles), updates models/latest when retrained and appends
the realised error of every past forecast whose prices are published to forecasts/performance.csv,
so live accuracy can be compared with the backtest. Before forecasting it checks the inputs (checks.py)
and the drift of inputs and live accuracy (drift.py); either drift alarm triggers a retrain. The check and
drift report go to forecasts/monitoring.json, and trainings and daily scores to MLflow when installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import checks, drift, evaluate as ev, models, tracking
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


def backtest_mae(root: Path = ROOT) -> float:
    """Walk-forward MAE of the deployed ensemble type (MLP + LSTM average), 2024-2025."""
    m = pd.read_csv(root / "results" / "backtest" / "metrics.csv")
    return float(m.loc[(m["period"] == "2024-2025") & (m["model"] == "mlp_lstm_avg"), "mae"].iloc[0])


def training_window(model_dir: Path) -> pd.DatetimeIndex | None:
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in model_dir.glob("*/meta.json")]
    if not metas:
        return None
    return pd.date_range(min(m["train_start"] for m in metas), max(m["train_end"] for m in metas), freq="D")


def monitor(inp: dict, day: pd.Timestamp, model_dir: Path, perf: pd.DataFrame) -> dict:
    """Drift of the inputs of the last 28 days against the current model's training window, and live accuracy."""
    window = training_window(model_dir)
    if window is None:
        return {"features": {}, "live": {}, "retrain": True, "reasons": ["no model"]}
    recent = pd.date_range(day - pd.Timedelta(days=28), day - pd.Timedelta(days=1), freq="D")
    return drift.report(inp, window, recent, perf, backtest_mae())


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--day", help="delivery day to forecast (default: tomorrow, Europe/Berlin)")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args(argv)
    today = pd.Timestamp.now(tz="Europe/Berlin").normalize().tz_localize(None)
    day = pd.Timestamp(args.day) if args.day else today + pd.Timedelta(days=1)
    inp = load_inputs((day - pd.Timedelta(days=WINDOW + 10)).date().isoformat(), day.date().isoformat(), DATA)
    folder = ROOT / "forecasts"
    folder.mkdir(parents=True, exist_ok=True)
    if inp["load"].loc[day].isna().any():
        raise SystemExit(f"TSO forecasts for {day.date()} not published yet")
    quality = checks.check_inputs(inp, day)
    if not quality["ok"]:
        (folder / "checks.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
        raise SystemExit("input checks failed: " + "; ".join(quality["errors"]))
    model_dir = ROOT / "models" / "latest"
    perf = score(folder, inp["price"])
    report = monitor(inp, day, model_dir, perf)
    reasons = ((["requested"] if args.retrain else []) + (["older than 30 days"] if needs_retrain(model_dir, day) else [])
               + report["reasons"])
    retrained = bool(reasons)
    if retrained:
        ens = train(inp, day)
        for old in model_dir.glob("*/"):
            for f in old.iterdir():
                f.unlink()
            old.rmdir()
        ens.save(model_dir)
        tracking.log_training(model_dir, {"train_end": (day - pd.Timedelta(days=1)).date(), "window_days": WINDOW,
                                          "kinds": ",".join(KINDS), "seeds": 3, "reasons": "; ".join(reasons)},
                              extra={"checks": quality, "drift": report})
    else:
        ens = models.Ensemble.load(model_dir)
    q = ens.predict(inp, pd.DatetimeIndex([day]))[0]
    path = write_forecast(q, day, folder)
    live = report.get("live", {})
    monitoring = {"day": str(day.date()), "checks": quality, "drift": report, "retrained": retrained, "reasons": reasons,
                  "scored_days": int(len(perf)), "live_mae_all": float(perf["mae"].mean()) if len(perf) else None}
    (folder / "monitoring.json").write_text(json.dumps(monitoring, indent=2), encoding="utf-8")
    tracking.log_metrics(f"daily {day.date()}", {"live_mae_14d": live.get("live_mae"), "live_ratio_14d": live.get("ratio"),
                                                  "retrained": float(retrained)}, artifacts={"monitoring": monitoring})
    result = {"day": str(day.date()), "forecast": str(path.relative_to(ROOT)), "retrained": retrained, "reasons": reasons,
              "scored_days": int(len(perf)), "live_mae": monitoring["live_mae_all"]}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
