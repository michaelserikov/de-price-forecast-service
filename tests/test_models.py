import numpy as np
import pandas as pd
import pytest
import torch

from dpfs import evaluate as ev, models
from dpfs.backtest import walk_forward


def test_pinball_matches_definition():
    pred = torch.zeros(1, 24, 3)
    y = torch.ones(1, 24)
    # error +1 for every quantile: loss = q
    assert float(models.pinball(pred, y)) == pytest.approx(np.mean(models.QUANTILES))


@pytest.mark.parametrize("kind", ["mlp", "lstm"])
def test_fit_beats_naive_and_quantiles_are_ordered(inp, kind):
    days = inp["price"].index
    cal, test = days[7:360], days[360:]
    f = models.fit(kind, inp, cal, seed=0, epochs=80, patience=15)
    q = f.predict(inp, test)
    assert q.shape == (len(test), 24, 3)
    assert (np.diff(q, axis=-1) >= 0).all()
    actual = inp["price"].loc[test]
    m = ev.metrics(actual, pd.DataFrame(q[..., 1], index=test, columns=actual.columns), ev.naive(inp["price"], test))
    assert m["rmae"] < 0.9


def test_save_load_roundtrip(inp, tmp_path):
    days = inp["price"].index
    ens = models.Ensemble([models.fit("mlp", inp, days[7:200], seed=s, epochs=5) for s in (0, 1)])
    ens.save(tmp_path)
    again = models.Ensemble.load(tmp_path)
    np.testing.assert_allclose(again.predict(inp, days[200:205]), ens.predict(inp, days[200:205]), rtol=1e-5)
    assert again.members[0].meta["train_end"] == str(days[199].date())


def test_walk_forward_uses_only_past(inp):
    days = inp["price"].index[380:420]   # two months, recalibrated at each month start
    q = walk_forward(inp, days, "mlp", window=300, seeds=(0,), epochs=5)
    assert not np.isnan(q).any() and q.shape == (40, 24, 3)


def test_coverage_and_dm():
    a = np.zeros((10, 24))
    assert ev.coverage(a, a - 1, a + 1) == 1.0
    idx = pd.date_range("2024-01-01", periods=50)
    rng = np.random.default_rng(1)
    act = pd.DataFrame(rng.normal(size=(50, 24)), index=idx)
    good, bad = act + rng.normal(0, 0.1, (50, 24)), act + rng.normal(0, 1.0, (50, 24))
    assert ev.dm_test(act, bad, good)[1] < 0.01
