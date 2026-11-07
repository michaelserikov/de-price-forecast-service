"""Experiment tracking with MLflow (optional: install with `pip install -e ".[mlops]"`).

Every training of the daily ensemble and every daily score is logged as a run in a local SQLite store
(`mlflow.db` and `mlartifacts/`, or MLFLOW_TRACKING_URI): parameters, validation losses, data checks, drift
report and the model files. Browse with `mlflow ui --backend-store-uri sqlite:///mlflow.db`. Without MLflow installed the functions do nothing, so the service and the daily job run the same.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .data import ROOT

try:
    import mlflow
except ImportError:          # pragma: no cover
    mlflow = None

EXPERIMENT = "dpfs-daily"


def enabled() -> bool:
    return mlflow is not None


def _setup(experiment: str) -> None:
    """SQLite tracking store (the file store is deprecated in MLflow 3) and a local artifact folder."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    artifacts = Path(os.environ.get("DPFS_MLFLOW_ARTIFACTS", ROOT / "mlartifacts"))
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        artifacts.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(experiment, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(experiment)


def log_training(model_dir: Path, params: dict, extra: dict | None = None, experiment: str = EXPERIMENT) -> str | None:
    """Logs one training of the ensemble; returns the run id."""
    if mlflow is None:
        return None
    _setup(experiment)
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(model_dir).glob("*/meta.json"))]
    with mlflow.start_run(run_name=f"train {params.get('train_end', '')}") as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        for i, m in enumerate(metas):
            if "val_pinball" in m:
                mlflow.log_metric(f"val_pinball_{m.get('kind', 'member')}_{m.get('seed', i)}", float(m["val_pinball"]))
        for name, obj in (extra or {}).items():
            mlflow.log_dict(obj, f"{name}.json")
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
        return run.info.run_id


def log_metrics(name: str, metrics: dict, params: dict | None = None, artifacts: dict | None = None,
                experiment: str = EXPERIMENT) -> str | None:
    """Logs a set of metrics, for example a backtest or a daily score."""
    if mlflow is None:
        return None
    _setup(experiment)
    with mlflow.start_run(run_name=name) as run:
        if params:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
        for fname, obj in (artifacts or {}).items():
            mlflow.log_dict(obj, f"{fname}.json")
        return run.info.run_id
