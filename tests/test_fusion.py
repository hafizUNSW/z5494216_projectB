"""Unit tests for src.fusion (sentiment tilt and fusion backtest).

Runnable standalone (`python tests/test_fusion.py`) or via pytest.
Synthetic data only.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src import fusion as f
from src import portfolios as p


def make_wide(n: int = 600, k: int = 6, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    ret = rng.normal(0.0005, 0.01, (n, k))
    return pd.DataFrame(ret, index=dates, columns=[f"A{i}" for i in range(k)])


def make_signal(wide: pd.DataFrame, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.0, 0.5, wide.shape), index=wide.index, columns=wide.columns
    )


def test_tilt_preserves_sum_and_caps():
    wide = make_wide()
    sig = make_signal(wide)
    _, base_w = p.walk_forward_backtest(wide, "equal_weight", window=252, rebalance=21)
    fused_w = f.apply_sentiment_tilt(base_w, sig, kappa=1.0, cap=0.20)
    sums = fused_w.groupby("rebalance_date")["weight"].sum()
    assert np.allclose(sums, 1.0, atol=1e-4)
    assert fused_w["weight"].max() <= 0.20 + 1e-6


def test_tilt_with_no_signal_keeps_base_weights():
    wide = make_wide()
    sig = pd.DataFrame(np.nan, index=wide.index, columns=wide.columns)
    _, base_w = p.walk_forward_backtest(wide, "equal_weight", window=252, rebalance=21)
    fused_w = f.apply_sentiment_tilt(base_w, sig, kappa=1.0, cap=0.20)
    base = base_w.set_index(["rebalance_date", "ticker"])["weight"]
    fused = fused_w.set_index(["rebalance_date", "ticker"])["weight"]
    assert np.allclose(base.reindex(fused.index).to_numpy(), fused.to_numpy(), atol=1e-6)


def test_tilt_changes_weights_with_signal():
    wide = make_wide()
    sig = make_signal(wide)
    _, base_w = p.walk_forward_backtest(wide, "equal_weight", window=252, rebalance=21)
    fused_w = f.apply_sentiment_tilt(base_w, sig, kappa=3.0, cap=0.20)
    base = base_w.pivot(index="rebalance_date", columns="ticker", values="weight")
    fused = fused_w.pivot(index="rebalance_date", columns="ticker", values="weight")
    assert (base - fused).abs().max().max() > 1e-3


def test_apply_weights_schedule():
    wide = make_wide()
    _, base_w = p.walk_forward_backtest(wide, "equal_weight", window=252, rebalance=21)
    returns = f.apply_weights(wide, base_w)
    # First rebalance is at index 251; weights earn from index 252 onward.
    assert returns.index[0] == wide.index[252]
    # Equal-weight + no signal drift: returns equal asset means on live dates.
    for dt in returns.index:
        assert np.isclose(returns[dt], wide.loc[dt].mean(), atol=1e-9)


def test_fused_fund_returns_align_with_base():
    wide = make_wide()
    sig = make_signal(wide)
    base_r, base_w, fused_r, fused_w = f.fused_fund_returns(
        wide, "equal_weight", sig, window=252, rebalance=21, family="equity", kappa=1.0
    )
    assert base_r.index.equals(fused_r.index)
    assert len(base_w) == len(fused_w)
    assert not base_r.equals(fused_r)


def test_null_tilt_matches_base_returns():
    wide = make_wide()
    sig = pd.DataFrame(np.zeros(wide.shape), index=wide.index, columns=wide.columns)
    base_r, _, fused_r, _ = f.fused_fund_returns(
        wide, "equal_weight", sig, window=252, rebalance=21, family="equity", kappa=1.0
    )
    assert np.allclose(base_r.to_numpy(), fused_r.to_numpy(), atol=1e-9)


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  {t.__name__} OK")


if __name__ == "__main__":
    _run()
    print("all fusion tests passed")
