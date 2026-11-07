"""FastAPI service: day-ahead price forecasts for DE-LU from the latest trained ensemble.

    uvicorn dpfs.service:app --port 8000
    GET /health                     liveness and model status
    GET /model                      training window and members of the loaded ensemble
    GET /                           home page: what the service does, how to use it, a form for a day
    GET /forecast?day=YYYY-MM-DD    24 hourly forecasts (10%, 50%, 90% quantiles) for a delivery day,
                                    after data-quality checks on the inputs (422 with the reasons if they fail);
                                    day may also be DD.MM.YYYY, "today" or "tomorrow", and defaults to tomorrow
                                    (today if tomorrow's TSO forecasts are not published yet)
    GET /monitoring                 latest checks and drift report of the daily job, live accuracy

The model folder is set with DPFS_MODEL_DIR (default models/latest). Inputs are fetched from SMARD
and the fuel sources through data.load_inputs and cached under data/; tests inject a provider.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from . import __version__, checks, models
from .data import ROOT, load_inputs

MODEL_DIR = Path(os.environ.get("DPFS_MODEL_DIR", ROOT / "models" / "latest"))
FORECAST_DIR = Path(os.environ.get("DPFS_FORECAST_DIR", ROOT / "forecasts"))
HISTORY_DAYS = 10  # inputs need prices of d-1 to d-7
TZ = "Europe/Berlin"
REPO = "https://github.com/michaelserikov/de-price-forecast-service"

DESCRIPTION = f"""
Hourly forecasts of German (DE-LU) day-ahead power prices for one delivery day: the median and the 10% and
90% quantiles in EUR/MWh, from an ensemble of PyTorch networks (MLP and LSTM).

At each request the service fetches the TSO forecasts of load, wind and solar from SMARD, the recent prices
and the gas and CO2 prices, checks them for missing or implausible values and runs the ensemble.

**How to use:** open **GET /forecast**, press *Try it out*, leave `day` empty for tomorrow (or enter a date
such as `2026-11-08` or `08.11.2026`) and press *Execute*. `q50` is the forecast, `q10` and `q90` bound the
range that should hold the price in 8 of 10 hours.

Research and engineering demonstration on public data, not investment advice. Code and backtest: {REPO}
"""


def parse_day(day: str | None) -> pd.Timestamp | None:
    """YYYY-MM-DD, DD.MM.YYYY, 'today' or 'tomorrow' (Berlin time); None if empty."""
    if day is None or not day.strip():
        return None
    s = day.strip().lower()
    today = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    if s in ("today", "heute"):
        return today
    if s in ("tomorrow", "morgen"):
        return today + pd.Timedelta(days=1)
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except ValueError:
            continue
    raise HTTPException(422, f"could not read the date '{day}': use YYYY-MM-DD (e.g. 2026-11-08), "
                             "DD.MM.YYYY, 'today' or 'tomorrow'")


HOME = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>German Power Price Forecast</title>
<style>
:root{--bg:#fbfaf7;--card:#fff;--ink:#1d2329;--muted:#5d6670;--line:#e3e0d8;--accent:#1f6f8b}
@media (prefers-color-scheme:dark){:root{--bg:#14181c;--card:#1c2228;--ink:#e6e8ea;--muted:#9aa4ad;--line:#2d353d;--accent:#6cb6d1}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:820px;margin:0 auto;padding:28px 16px 48px}
h1{font-size:1.5rem;margin:0 0 6px}h2{font-size:1.05rem;margin:22px 0 8px}
p,li{color:var(--ink)}.muted{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:14px 0}
code{background:var(--line);padding:1px 5px;border-radius:4px}
a{color:var(--accent)}
form{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
input,button{font:inherit;padding:7px 10px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
button{background:var(--accent);color:#fff;border:0;cursor:pointer}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:.9rem;margin-top:10px}
th,td{text-align:right;padding:4px 8px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}
th{color:var(--muted)}
</style></head><body><main>
<h1>German day-ahead power price forecast</h1>
<p>This service forecasts the hourly German (DE-LU) day-ahead power prices for one delivery day. For every hour
it gives the most likely price (<code>q50</code>) and a range from <code>q10</code> to <code>q90</code> that
should contain the price in 8 of 10 hours, in EUR/MWh.</p>
<p>At each request it fetches the TSO forecasts of load, wind and solar from SMARD, the prices of the past week
and the gas and CO2 prices, checks them for missing or implausible values and runs an ensemble of PyTorch
networks (MLP and LSTM) retrained every month. A scheduled job also stores a forecast every evening and scores
it once the real prices are published.</p>

<div class="card">
<form id="f"><label for="d">Delivery day</label><input id="d" type="date"><button>Forecast</button>
<span class="muted" id="msg"></span></form>
<div id="out"></div>
</div>

<h2>How to use the API</h2>
<ul>
<li><a href="forecast">/forecast</a>: forecast for tomorrow; <a href="forecast?day=today">/forecast?day=today</a>
or <code>/forecast?day=2026-11-08</code> for a given day (also <code>08.11.2026</code>)</li>
<li><a href="docs">/docs</a>: interactive documentation, try every call in the browser</li>
<li><a href="monitoring">/monitoring</a>: data checks, drift report and live accuracy of past forecasts</li>
<li><a href="model">/model</a>: networks in the ensemble and their training window;
<a href="health">/health</a>: service status</li>
</ul>
<p class="muted">The free instance sleeps after 15 minutes without requests; the first call then takes about a
minute. Research and engineering demonstration on public data, not investment advice.
Code, method and backtest: <a href="__REPO__">__REPO__</a>. Daily forecasts as a chart:
<a href="https://huggingface.co/spaces/michaelserikov/de-price-forecast">Hugging Face page</a>.</p>
</main>
<script>
const f=document.getElementById("f"),d=document.getElementById("d"),msg=document.getElementById("msg"),out=document.getElementById("out");
const t=new Date();t.setDate(t.getDate()+1);d.value=t.toISOString().slice(0,10);
f.onsubmit=async e=>{e.preventDefault();msg.textContent="computing, up to a minute if the service was asleep…";out.innerHTML="";
 try{const r=await fetch("forecast?day="+encodeURIComponent(d.value));const j=await r.json();
  if(!r.ok){msg.textContent="";out.textContent=typeof j.detail==="string"?j.detail:JSON.stringify(j.detail);return;}
  const avg=j.hours.reduce((a,h)=>a+h.q50,0)/24,has=j.score!==undefined;
  msg.textContent=j.day+": forecast day average "+avg.toFixed(1)+" EUR/MWh"+(has?`; auction already cleared: mean absolute error ${j.score.mae.toFixed(1)}, ${j.score.hours_inside_80pct_band} of 24 hours inside the band`:"");
  out.innerHTML=(j.warnings&&j.warnings.length?`<p class="muted">Input warnings: ${j.warnings.join("; ")}</p>`:"")+
   "<table><tr><th>Hour</th><th>q10</th><th>q50</th><th>q90</th>"+(has?"<th>Actual</th>":"")+"</tr>"+j.hours.map(h=>
   `<tr><td>${String(h.hour).padStart(2,"0")}:00</td><td>${h.q10.toFixed(1)}</td><td><b>${h.q50.toFixed(1)}</b></td><td>${h.q90.toFixed(1)}</td>${has?`<td>${h.actual.toFixed(1)}</td>`:""}</tr>`).join("")+"</table>";
 }catch(err){msg.textContent="request failed: "+err;}};
</script></body></html>"""


def default_provider(day: pd.Timestamp) -> dict:
    return load_inputs((day - pd.Timedelta(days=HISTORY_DAYS)).date().isoformat(), day.date().isoformat())


def create_app(model_dir: Path = MODEL_DIR, provider: Callable[[pd.Timestamp], dict] = default_provider,
               forecast_dir: Path = FORECAST_DIR) -> FastAPI:
    app = FastAPI(title="German day-ahead power price forecast", version=__version__, description=DESCRIPTION)
    state: dict = {"ensemble": None}

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> str:
        return HOME.replace("__REPO__", REPO)

    def ensemble() -> models.Ensemble:
        if state["ensemble"] is None:
            if not model_dir.exists() or not any(model_dir.iterdir()):
                raise HTTPException(503, f"no trained model in {model_dir}")
            state["ensemble"] = models.Ensemble.load(model_dir)
        return state["ensemble"]

    @app.get("/health")
    def health() -> dict:
        ready = model_dir.exists() and any(model_dir.glob("*/meta.json"))
        return {"status": "ok", "model_loaded": ready, "version": __version__}

    @app.get("/model")
    def model_info() -> dict:
        return {"members": [{"kind": m.kind, **m.meta} for m in ensemble().members]}

    @app.get("/forecast", summary="Hourly price forecast for one delivery day")
    def forecast(day: str | None = Query(
            None, description="Delivery day: empty for tomorrow, or 2026-11-08, 08.11.2026, today, tomorrow",
            examples=["tomorrow"])) -> dict:
        d = parse_day(day)
        if d is None:                       # tomorrow, or today before tomorrow's TSO forecasts are out
            tomorrow = parse_day("tomorrow")
            try:
                return _forecast(tomorrow)
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
                return _forecast(tomorrow - pd.Timedelta(days=1))
        return _forecast(d)

    def _forecast(d: pd.Timestamp) -> dict:
        day = d.date().isoformat()
        ens = ensemble()
        try:
            inp = provider(d)
        except Exception as exc:  # data source down or day not yet available
            raise HTTPException(502, f"inputs unavailable: {exc}") from exc
        if d not in inp["load"].index or inp["load"].loc[d].isna().any():
            raise HTTPException(404, f"TSO forecasts for {day} are not available yet")
        quality = checks.check_inputs(inp, d)
        if not quality["ok"]:
            raise HTTPException(422, {"message": "input checks failed", "errors": quality["errors"]})
        q = ens.predict(inp, pd.DatetimeIndex([d]))[0]
        hours = [{"hour": h, "q10": round(float(q[h, 0]), 2), "q50": round(float(q[h, 1]), 2),
                  "q90": round(float(q[h, 2]), 2)} for h in range(24)]
        out = {"day": day, "unit": "EUR/MWh", "zone": "DE-LU", "hours": hours, "warnings": quality["warnings"]}
        price = inp["price"]
        if d in price.index and not price.loc[d].isna().any():   # auction already cleared: show the outcome
            actual = price.loc[d].to_numpy(float)
            for h in hours:
                h["actual"] = round(float(actual[h["hour"]]), 2)
            q50 = [h["q50"] for h in hours]
            out["score"] = {"mae": round(float(abs(actual - q50).mean()), 2),
                            "hours_inside_80pct_band": int(sum(h["q10"] <= h["actual"] <= h["q90"] for h in hours))}
        return out

    @app.get("/monitoring")
    def monitoring() -> dict:
        """Latest data checks and drift report of the daily job, and the live accuracy of stored forecasts."""
        out: dict = {"report": None, "live": None}
        rep = forecast_dir / "monitoring.json"
        if rep.exists():
            out["report"] = json.loads(rep.read_text(encoding="utf-8"))
        perf = forecast_dir / "performance.csv"
        if perf.exists():
            p = pd.read_csv(perf)
            if len(p):
                out["live"] = {"days": int(len(p)), "mae": round(float(p["mae"].mean()), 3),
                               "mae_last_14_days": round(float(p.sort_values("day")["mae"].tail(14).mean()), 3),
                               "coverage_80": round(float(p["coverage_80"].mean()), 3)}
        return out

    return app


app = create_app()
