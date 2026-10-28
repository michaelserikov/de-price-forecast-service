"""FastAPI service: day-ahead price forecasts for DE-LU from the latest trained ensemble.

    uvicorn dpfs.service:app --port 8000
    GET /health                     liveness and model status
    GET /model                      training window and members of the loaded ensemble
    GET /forecast?day=YYYY-MM-DD    24 hourly forecasts (10%, 50%, 90% quantiles) for a delivery day

The model folder is set with DPFS_MODEL_DIR (default models/latest). Inputs are fetched from SMARD
and the fuel sources through data.load_inputs and cached under data/; tests inject a provider.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from . import __version__, models
from .data import ROOT, load_inputs

MODEL_DIR = Path(os.environ.get("DPFS_MODEL_DIR", ROOT / "models" / "latest"))
HISTORY_DAYS = 10  # inputs need prices of d-1 to d-7


def default_provider(day: pd.Timestamp) -> dict:
    return load_inputs((day - pd.Timedelta(days=HISTORY_DAYS)).date().isoformat(), day.date().isoformat())


def create_app(model_dir: Path = MODEL_DIR, provider: Callable[[pd.Timestamp], dict] = default_provider) -> FastAPI:
    app = FastAPI(title="de-price-forecast-service", version=__version__)
    state: dict = {"ensemble": None}

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

    @app.get("/forecast")
    def forecast(day: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$")) -> dict:
        d = pd.Timestamp(day)
        ens = ensemble()
        try:
            inp = provider(d)
        except Exception as exc:  # data source down or day not yet available
            raise HTTPException(502, f"inputs unavailable: {exc}") from exc
        if d not in inp["load"].index or inp["load"].loc[d].isna().any():
            raise HTTPException(404, f"TSO forecasts for {day} are not available yet")
        q = ens.predict(inp, pd.DatetimeIndex([d]))[0]
        hours = [{"hour": h, "q10": round(float(q[h, 0]), 2), "q50": round(float(q[h, 1]), 2),
                  "q90": round(float(q[h, 2]), 2)} for h in range(24)]
        return {"day": day, "unit": "EUR/MWh", "zone": "DE-LU", "hours": hours}

    return app


app = create_app()
