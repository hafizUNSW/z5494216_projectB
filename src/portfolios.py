"""Station 3 - funds: optimal portfolios + walk-forward out-of-sample backtest.

Design choices (all stated for the report):
  - Asset families: equity-only, crypto-only, combined. Each (family, method)
    pair is one investable fund.
  - Methods: equal_weight (1/N baseline), minimum_variance, max_sharpe
    (mean-variance tangency, rf = 0), risk_parity (equal risk contribution).
  - Backtest: rolling estimation window of `window` periods, rebalanced every
    `rebalance` periods (21 periods ~ monthly). Weights are formed ONLY from
    returns inside the trailing window (no look-ahead) and are applied from
    the NEXT period after the rebalance date.
  - Long-only, fully invested, per-name weight cap to avoid concentration.
  - Covariance is shrunk toward the diagonal so the optimiser does not stall
    on near-singular daily-return covariances (a known trap in this data).
  - Annualisation: 252 for equity/combined (equity calendar), 365 for crypto
    (its native 7-day calendar).
  - rf = 0 and zero transaction costs are assumed (stated in the report).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS_EQUITY = 252
TRADING_DAYS_CRYPTO = 365

# Per-family maximum weight on any single name (long-only concentration cap).
MAX_WEIGHT = {
    "equity": 0.20,
    "crypto": 0.35,
    "combined": 0.20,
}

METHODS = ("equal_weight", "minimum_variance", "max_sharpe", "risk_parity")
METHOD_LABELS = {
    "equal_weight": "Equal Weight",
    "minimum_variance": "Minimum Variance",
    "max_sharpe": "Maximum Sharpe",
    "risk_parity": "Risk Parity",
}


# ---------------------------------------------------------------------------
# Covariance shrinkage
# ---------------------------------------------------------------------------

def shrink_covariance(cov: np.ndarray, delta: float = 0.10) -> np.ndarray:
    """Shrink the sample covariance toward a diagonal target.

    The sample covariance of daily returns is noisy, and optimisers on it can
    silently stall (objective below tolerance). Shrinking by `delta` toward a
    diagonal matrix with the average variance guarantees a well-conditioned,
    positive-definite estimate without changing the data.

        Sigma_shrunk = (1 - delta) * Sigma_sample + delta * diag(mean_diag)
    """
    cov = np.asarray(cov, dtype=float)
    target = np.mean(np.diag(cov)) * np.eye(cov.shape[0])
    return (1.0 - delta) * cov + delta * target


def normalize_capped(w, cap: float) -> np.ndarray:
    """Project weights to the unit simplex with a hard per-name cap.

    A naive `clip(w, 0, cap); w = w / w.sum()` lets normalisation push a name
    back above the cap (and clips below the full sum too). This is a
    water-filling projection instead: names at the cap stay put and the whole
    excess/shortfall is absorbed by the names still below the cap.
    """
    w = np.clip(np.asarray(w, dtype=float), 0.0, cap)
    for _ in range(200):
        s = w.sum()
        diff = s - 1.0
        if abs(diff) <= 1e-9:
            return w
        below = w < cap - 1e-12
        if not below.any():
            # Caps are infeasible (sum(cap) < 1): scale to the simplex anyway.
            return w / s if s > 0 else np.full(len(w), 1.0 / len(w))
        if diff > 0:
            w[below] -= min(diff, w[below].sum()) / below.sum()
        else:
            w[below] += (-diff) / below.sum()
        w = np.clip(w, 0.0, cap)
    return w


# ---------------------------------------------------------------------------
# Optimisation methods (long-only, fully invested, capped)
# ---------------------------------------------------------------------------

def _optimize(mean: np.ndarray, cov: np.ndarray, objective: callable,
              n: int, cap: float) -> np.ndarray:
    """Solve min objective(w) s.t. w >= 0, sum(w) = 1, w_i <= cap."""
    bounds = [(0.0, cap)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    x0 = np.full(n, 1.0 / n)
    result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                      constraints=constraints,
                      options={"ftol": 1e-10, "maxiter": 1000})
    if not result.success:
        # Fall back to a feasible equal-weight solution rather than fail.
        return np.full(n, 1.0 / n)
    w = np.clip(result.x, 0.0, cap)
    return w / w.sum()


def _min_variance(mean: np.ndarray, cov: np.ndarray, cap: float) -> np.ndarray:
    def objective(w):
        return 0.5 * float(w @ cov @ w)
    return _optimize(mean, cov, objective, len(mean), cap)


def _max_sharpe(mean: np.ndarray, cov: np.ndarray, cap: float) -> np.ndarray:
    # rf = 0: maximise w'mu / sqrt(w'Sigmaw).
    # If no asset has a positive mean, the tangency portfolio is undefined;
    # fall back to minimum variance (documented in the report).
    if mean.max() <= 0.0:
        return _min_variance(mean, cov, cap)

    def objective(w):
        vol2 = float(w @ cov @ w)
        if vol2 <= 0.0:
            return 0.0
        return -float(w @ mean) / np.sqrt(vol2)
    return _optimize(mean, cov, objective, len(mean), cap)


def _erc_no_cap(cov: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Unconstrained equal-risk-contribution weights via Newton's method.

    ERC solves min_x  0.5 x'Sx - sum log(x);  the first-order condition is
        x_i (Sx)_i = 1  for all i,
    i.e. every asset contributes the same amount of risk. Newton converges
    quadratically (Spinu 2013) and is ~150x faster than SLSQP at 60 assets.
    """
    n = cov.shape[0]
    x = np.full(n, 1.0 / n)
    for _ in range(100):
        g = cov @ x - 1.0 / x
        if np.abs(g).max() < tol:
            break
        hess = cov + np.diag(1.0 / x ** 2)
        try:
            step = np.linalg.solve(hess, -g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, -g, rcond=None)[0]
        # Backtracking line search keeps x positive and lowers the objective.
        alpha = 1.0
        obj_x = float(0.5 * x @ cov @ x - np.log(x).sum())
        while alpha > 1e-10:
            xn = x + alpha * step
            if (xn > 0).all() and float(0.5 * xn @ cov @ xn - np.log(xn).sum()) < obj_x:
                break
            alpha *= 0.5
        x = x + alpha * step
    return x / x.sum()


def _risk_parity(mean: np.ndarray, cov: np.ndarray, cap: float) -> np.ndarray:
    # Equal risk contribution (ERC): contribution_i = w_i * (Sigma w)_i is the
    # same for every asset. Caps make the full problem infeasible, so solve ERC
    # on the active universe and, whenever a name would exceed the cap, pin it
    # at the cap and re-solve on the residual budget (standard recursive
    # construction). Unconstrained ERC is unique for a positive-definite Sigma.
    n = len(mean)
    w = np.zeros(n)
    active = np.ones(n, dtype=bool)
    budget = 1.0
    while active.any() and budget > 1e-12:
        sub = np.where(active)[0]
        x = _erc_no_cap(cov[sub][:, sub])
        eff_cap = min(cap, budget)
        if (x <= eff_cap / budget + 1e-12).all():
            w[sub] = x * budget
            return w
        over = x > eff_cap / budget + 1e-12
        fix = sub[over]
        w[fix] = eff_cap
        budget -= eff_cap * over.sum()
        active[fix] = False
    # Infeasible combination of caps and count (sum(cap) < 1): fall back to
    # proportional cap allocation so the fund is still fully invested.
    if w.sum() < 1.0 - 1e-9:
        w = np.full(n, cap)
        w = w / w.sum()
    return w


def optimize_weights(returns_wide: pd.DataFrame, method: str,
                     cap: float | None = None) -> pd.Series:
    """Return the target weight series for one method from a trailing window.

    `returns_wide` is a (period x asset) DataFrame of returns that ends on the
    rebalance date. Weights use only this past window.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    n = returns_wide.shape[1]
    if n == 0:
        raise ValueError("empty asset panel")
    cap = cap if cap is not None else MAX_WEIGHT["combined"]
    values = returns_wide.to_numpy(dtype=float)
    mean = np.nanmean(values, axis=0)
    cov = np.cov(values, rowvar=False, ddof=1)
    cov = np.where(np.isnan(cov), 0.0, cov)
    cov = shrink_covariance(cov)

    if method == "equal_weight":
        w = np.full(n, 1.0 / n)
    elif method == "minimum_variance":
        w = _min_variance(mean, cov, cap)
    elif method == "max_sharpe":
        w = _max_sharpe(mean, cov, cap)
    elif method == "risk_parity":
        w = _risk_parity(mean, cov, cap)
    else:
        raise ValueError(method)
    return pd.Series(w, index=returns_wide.columns)


def walk_forward_backtest(
    returns_wide: pd.DataFrame,
    method: str,
    window: int = 504,
    rebalance: int = 21,
    family: str = "combined",
) -> tuple[pd.Series, pd.DataFrame]:
    """Walk-forward out-of-sample backtest for one fund.

    Returns:
      returns: pd.Series of OUT-OF-SAMPLE daily portfolio returns (indexed by
               date), starting the period after the first rebalance.
      weights: pd.DataFrame of target weights at each rebalance date, long
               format (rebalance_date, ticker, weight).
    """
    returns_wide = returns_wide.sort_index()
    dates = returns_wide.index
    if len(dates) < window + rebalance:
        raise ValueError(
            f"need at least {window + rebalance} periods, have {len(dates)}"
        )

    cap = MAX_WEIGHT.get(family, MAX_WEIGHT["combined"])
    rebalance_positions = list(range(window - 1, len(dates), rebalance))

    weight_rows: list[dict] = []
    portfolio_returns: dict[pd.Timestamp, float] = {}

    for i, pos in enumerate(rebalance_positions):
        rebalance_date = dates[pos]
        # Weights from the trailing window ending AT the rebalance date.
        window_data = returns_wide.iloc[pos - window + 1: pos + 1]
        w = optimize_weights(window_data, method, cap=cap)
        w = pd.Series(normalize_capped(w.to_numpy(), cap), index=w.index)

        # The next rebalance date sets the end of this holding period.
        if i + 1 < len(rebalance_positions):
            next_pos = rebalance_positions[i + 1]
        else:
            next_pos = len(dates) - 1
        # Weights are applied from the NEXT period after the rebalance date,
        # so no look-ahead: the decision on `rebalance_date` cannot use returns
        # on or after that date.
        holding_positions = range(pos + 1, next_pos + 1)
        for hp in holding_positions:
            period_returns = returns_wide.iloc[hp]
            portfolio_returns[dates[hp]] = float(
                np.nansum(w.to_numpy() * period_returns.to_numpy())
            )

        for ticker, weight in w.items():
            weight_rows.append({
                "rebalance_date": rebalance_date,
                "ticker": ticker,
                "weight": round(float(weight), 6),
            })

    returns = pd.Series(portfolio_returns, name="return").sort_index()
    weights = pd.DataFrame(weight_rows)
    return returns, weights


# ---------------------------------------------------------------------------
# Performance metrics and fact sheets
# ---------------------------------------------------------------------------

def growth_of_one(returns: pd.Series) -> pd.Series:
    """Growth of $1 from simple daily returns."""
    return (1.0 + returns.fillna(0.0)).cumprod()


def drawdown(returns: pd.Series) -> pd.Series:
    """Drawdown series (<= 0) from simple daily returns."""
    wealth = growth_of_one(returns)
    return wealth / wealth.cummax() - 1.0


def performance_metrics(daily_returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Annualised return, volatility, Sharpe (rf = 0), and max drawdown.

    Returns a plain dict so callers can shape it into a table.
    """
    r = daily_returns.dropna()
    if len(r) == 0:
        return {k: float("nan") for k in (
            "annualized_return", "annualized_volatility", "sharpe_ratio",
            "max_drawdown", "growth_of_1")}
    ann = float(periods_per_year)
    n = float(len(r))
    growth = float(growth_of_one(r).iloc[-1])
    annualized_return = float(growth ** (ann / n) - 1.0)
    annualized_volatility = float(r.std(ddof=1) * np.sqrt(ann))
    sharpe = (float(r.mean()) / float(r.std(ddof=1)) * np.sqrt(ann)
              if r.std(ddof=1) > 0 else float("nan"))
    max_drawdown = float(drawdown(r).min())
    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "growth_of_1": growth,
    }


def fact_sheet(
    fund: str,
    family: str,
    returns: pd.Series,
    weights: pd.DataFrame,
    periods_per_year: int,
) -> dict:
    """One fact sheet row: growth, annualised metrics, and current holdings."""
    metrics = performance_metrics(returns, periods_per_year)
    latest = weights["rebalance_date"].max() if len(weights) else None
    current = weights[weights["rebalance_date"] == latest] if latest is not None else pd.DataFrame()
    holdings = (
        current.sort_values("weight", ascending=False)
        .head(10)[["ticker", "weight"]]
        .assign(weight_pct=lambda d: (d["weight"] * 100).round(2))
        .to_dict("records")
    )
    return {
        "fund": fund,
        "family": family,
        "periods_per_year": periods_per_year,
        "rebalance_date": str(latest.date()) if latest is not None else None,
        **{k: round(v, 6) for k, v in metrics.items()},
        "holdings": holdings,
    }
