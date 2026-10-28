"""PyTorch models for the 24 day-ahead prices of a day, trained with the quantile (pinball) loss.

- MLP: the 249 LEAR inputs of a day -> 24 hours x 3 quantiles.
- LSTM: a bidirectional LSTM over the 24 hours of the delivery day; each hour sees its own price
  lags, load and renewable forecasts and the calendar, and the recurrence shares information
  across the day (morning and evening peaks, the solar dip).

Both predict the 10%, 50% and 90% quantiles in the transformed price space of features.Scaler;
the median is the point forecast and the 10-90% band an 80% interval. Quantiles are sorted after
prediction so they never cross. Training uses Adam, early stopping on the most recent 10% of the
calibration window and a fixed seed per ensemble member.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import features as ft

QUANTILES = (0.1, 0.5, 0.9)


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: tuple[int, ...] = (256, 128), dropout: float = 0.1, n_q: int = 3):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 24 * n_q))
        self.net, self.n_q = nn.Sequential(*layers), n_q

    def forward(self, x: torch.Tensor) -> torch.Tensor:       # (batch, n_in) -> (batch, 24, n_q)
        return self.net(x).view(-1, 24, self.n_q)


class LSTM(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64, n_q: int = 3):
        super().__init__()
        self.rnn = nn.LSTM(n_in, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, n_q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:       # (batch, 24, n_in) -> (batch, 24, n_q)
        out, _ = self.rnn(x)
        return self.head(out)


def pinball(pred: torch.Tensor, y: torch.Tensor, quantiles=QUANTILES) -> torch.Tensor:
    """Mean quantile loss; pred (batch, 24, n_q), y (batch, 24)."""
    q = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device)
    e = y.unsqueeze(-1) - pred
    return torch.maximum(q * e, (q - 1) * e).mean()


ARCH = {"mlp": (MLP, ft.tabular), "lstm": (LSTM, ft.sequence)}


@dataclass
class Fitted:
    """One trained network with its scaler; predicts prices in EUR/MWh."""
    kind: str
    model: nn.Module
    scaler: ft.Scaler
    meta: dict = field(default_factory=dict)

    def predict(self, inp: dict, days: pd.DatetimeIndex) -> np.ndarray:
        X, is_price = ARCH[self.kind][1](inp, days)
        Xs = np.nan_to_num(self.scaler.x(X, is_price))
        self.model.eval()
        with torch.no_grad():
            z = self.model(torch.from_numpy(Xs)).numpy()
        return np.sort(self.scaler.y_inv(z), axis=-1)             # (days, 24, 3), quantiles sorted

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), folder / "weights.pt")
        (folder / "meta.json").write_text(json.dumps({"kind": self.kind, "scaler": self.scaler.to_dict(), **self.meta},
                                                     indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path) -> "Fitted":
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        kind, scaler = meta.pop("kind"), ft.Scaler(**meta.pop("scaler"))
        model = ARCH[kind][0](meta["n_in"])
        model.load_state_dict(torch.load(folder / "weights.pt", weights_only=True))
        return cls(kind, model, scaler, meta)


def fit(kind: str, inp: dict, cal_days: pd.DatetimeIndex, seed: int = 0, epochs: int = 300,
        lr: float = 1e-3, batch: int = 32, patience: int = 30) -> Fitted:
    """Train one network on the calibration days (rows with missing inputs or prices are dropped)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, is_price = ARCH[kind][1](inp, cal_days)
    Y = inp["price"].reindex(cal_days).to_numpy()
    flat = X.reshape(len(X), -1)
    ok = ~np.isnan(flat).any(axis=1) & ~np.isnan(Y).any(axis=1)
    X, Y = X[ok], Y[ok]
    scaler = ft.Scaler.fit(X, is_price, Y)
    Xs, Ys = torch.from_numpy(scaler.x(X, is_price)), torch.from_numpy(scaler.y(Y))
    n_val = max(1, len(Xs) // 10)                                 # most recent days for early stopping
    Xtr, Ytr, Xva, Yva = Xs[:-n_val], Ys[:-n_val], Xs[-n_val:], Ys[-n_val:]
    model = ARCH[kind][0](X.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best, best_state, wait = np.inf, None, 0
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(Xtr), generator=gen).split(batch):
            opt.zero_grad()
            loss = pinball(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(pinball(model(Xva), Yva))
        if val < best - 1e-5:
            best, best_state, wait = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    meta = {"n_in": int(X.shape[-1]), "seed": seed, "train_start": str(cal_days[0].date()),
            "train_end": str(cal_days[-1].date()), "n_train": int(len(Xs)), "val_pinball": best}
    return Fitted(kind, model, scaler, meta)


class Ensemble:
    """Average of several fitted networks (quantile by quantile)."""

    def __init__(self, members: list[Fitted]):
        self.members = members

    def predict(self, inp: dict, days: pd.DatetimeIndex) -> np.ndarray:
        return np.sort(np.mean([m.predict(inp, days) for m in self.members], axis=0), axis=-1)

    def save(self, folder: Path) -> None:
        for i, m in enumerate(self.members):
            m.save(folder / f"{m.kind}_{i}")

    @classmethod
    def load(cls, folder: Path) -> "Ensemble":
        return cls([Fitted.load(p) for p in sorted(folder.iterdir()) if (p / "meta.json").exists()])
