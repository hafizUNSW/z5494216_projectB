"""Unit tests for src.portfolios (fund optimisation and backtests).

Runnable standalone (`python tests/test_portfolios.py`) or via pytest.
Synthetic data only - no network, no data files.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src import portfolios as p


def make_wide(n: int = 600, k: int = 6, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vols = np.array([0.005, 0.008, 0.012, 0.020, 0.030, 0.050])[:k]
    means = np.array([0.0015, 0.0010, 0.0008, 0.0003, 0.0002, 0.0001])[:k]
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    ret = means + rng.normal(0.0, 1.0, (n, k)) * vols
    return pd.DataFrame(ret, index=dates, columns=[f"A{i}" for i in range(k)])


def test_equal_weight():
    wide = make_wide()
    w = p.optimize_weights(wide, "equal_weight", cap=0.2)
    assert np.allclose(w.to_numpy(), 1.0 / wide.shape[1])
    assert np.isclose(w.sum(), 1.0)


def test_minimum_variance_never_increases_in_sample_risk():
    wide = make_wide(k=6)
    w_ew = p.optimize_weights(wide, "equal_weight", cap=0.5)
    w_mv = p.optimize_weights(wide, "minimum_variance", cap=0.5)
    var_ew = w_ew.to_numpy() @ (wide.cov().to_numpy() @ w_ew.to_numpy())
    var_mv = w_mv.to_numpy() @ (wide.cov().to_numpy() @ w_mv.to_numpy())
    assert var_mv <= var_ew + 1e-9


def test_max_sharpe_prefers_positive_mean_asset():
    rng = np.random.default_rng(11)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    base = rng.normal(0.0, 0.01, (300, 3))
    base[:, 0] += 0.004  # asset A0 with a clear positive drift
    wide = pd.DataFrame(base, index=dates, columns=["A0", "A1", "A2"])
    w = p.optimize_weights(wide, "max_sharpe", cap=1.0)
    assert w["A0"] > w["A1"]
    assert w["A0"] > w["A2"]


def test_max_sharpe_falls_back_when_no_positive_mean():
    rng = np.random.default_rng(13)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    base = rng.normal(-0.001, 0.01, (300, 3))
    wide = pd.DataFrame(base, index=dates, columns=["A0", "A1", "A2"])
    w_sharpe = p.optimize_weights(wide, "max_sharpe", cap=1.0)
    w_mv = p.optimize_weights(wide, "minimum_variance", cap=1.0)
    assert np.allclose(w_sharpe.to_numpy(), w_mv.to_numpy())


def test_risk_parity_equal_contributions():
    wide = make_wide(k=6)
    cov = p.shrink_covariance(wide.cov().to_numpy())
    w = p.optimize_weights(wide, "risk_parity", cap=1.0)
    rc = w.to_numpy() * (cov @ w.to_numpy())
    assert rc.max() - rc.min() < 1e-4
    assert np.isclose(w.sum(), 1.0)


def test_risk_parity_respects_caps():
    wide = make_wide(k=6)
    w = p.optimize_weights(wide, "risk_parity", cap=0.2)
    assert w.max() <= 0.2 + 1e-9
    assert np.isclose(w.sum(), 1.0)


def test_unknown_method_rejected():
    wide = make_wide()
    try:
        p.optimize_weights(wide, "black_litterman", cap=0.2)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_walk_forward_no_look_ahead_first_weight():
    wide = make_wide(n=600, k=6)
    _, weights = p.walk_forward_backtest(wide, "minimum_variance", window=252, rebalance=21)
    # First rebalance uses exactly the first 252 rows, nothing after.
    expected = p.optimize_weights(wide.iloc[:252], "minimum_variance", cap=0.2)
    first = (weights[weights["rebalance_date"] == weights["rebalance_date"].min()]
             .set_index("ticker")["weight"])
    assert np.allclose(first.sort_index().to_numpy(), expected.sort_index().to_numpy(), atol=1e-4)


def test_walk_forward_returns_start_after_window():
    wide = make_wide(n=600, k=6)
    returns, _ = p.walk_forward_backtest(wide, "equal_weight", window=252, rebalance=21)
    # No return can use data before the window ends: first live date is index 252.
    assert returns.index[0] == wide.index[252]
    assert len(returns) == len(wide) - 252


def test_walk_forward_equal_weight_matches_asset_means():
    # Equal-weight uses no estimation, so every fund return must equal the
    # cross-sectional mean of asset returns that period - validates schedule.
    wide = make_wide(n=300, k=5)
    returns, _ = p.walk_forward_backtest(wide, "equal_weight", window=100, rebalance=21)
    for dt in returns.index:
        assert np.isclose(returns[dt], wide.loc[dt].mean(), atol=1e-9)


def test_growth_of_one():
    s = pd.Series([0.10, 0.10], index=pd.date_range("2021-01-01", periods=2, freq="B"))
    g = p.growth_of_one(s)
    assert np.isclose(g.iloc[-1], 1.1 * 1.1)


def test_drawdown_and_metrics():
    s = pd.Series([0.05, -0.05, 0.05], index=pd.date_range("2021-01-01", periods=3, freq="B"))
    dd = p.drawdown(s)
    assert (dd <= 1e-12).all()
    assert np.isclose(dd.min(), 1.05 * 0.95 / 1.05 - 1.0)
    m = p.performance_metrics(s, periods_per_year=252)
    assert set(m) == {"annualized_return", "annualized_volatility", "sharpe_ratio",
                      "max_drawdown", "growth_of_1"}
    assert np.isclose(m["growth_of_1"], (1.05 * 0.95 * 1.05))


def test_performance_metrics_zero_vol_sharpe_nan():
    s = pd.Series([0.001] * 10)
    m = p.performance_metrics(s, periods_per_year=252)
    # Near-zero volatility must not raise and the metrics dict is complete.
    assert set(m) == {"annualized_return", "annualized_volatility", "sharpe_ratio",
                      "max_drawdown", "growth_of_1"}
    assert m["annualized_volatility"] >= 0.0


def test_fact_sheet_holdings():
    wide = make_wide(n=600, k=6)
    returns, weights = p.walk_forward_backtest(wide, "equal_weight", window=252, rebalance=21)
    sheet = p.fact_sheet("Test Fund", "equity", returns, weights, periods_per_year=252)
    assert sheet["fund"] == "Test Fund"
    assert len(sheet["holdings"]) == 6
    assert np.isclose(sum(h["weight"] for h in sheet["holdings"]), 1.0, atol=1e-3)


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  {t.__name__} OK")


if __name__ == "__main__":
    _run()
    print("all portfolio tests passed")
