"""Station 3 (extension) - fuse equity sentiment into the funds.

The fusion is a sentiment tilt on the EQUITY funds (sentiment applies to
equity data only). At each rebalance, the base weights are tilted toward
names with a stronger (look-ahead-safe) news-sentiment signal:

    w_i' ~ w_i * (1 + kappa * z_i)

where z_i is the lagged, smoothed per-ticker sentiment on the rebalance date
(from src.sentiment.ticker_signal_wide: aligned score shifted 1 trading day,
then a trailing 21-day mean - so the tilt at date t uses only sentiment from
dates < t). Tickers without a usable signal keep their base weight. Weights
are renormalised to 1 and the family concentration cap is re-applied.

An honest negative result is good work; the point of the extension is the
measured before-vs-after effect, not a guaranteed improvement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolios import normalize_capped, walk_forward_backtest


def apply_sentiment_tilt(
    weights_long: pd.DataFrame,
    sentiment_signal: pd.DataFrame,
    kappa: float = 1.0,
    cap: float = 0.20,
) -> pd.DataFrame:
    """Tilt one fund's rebalance weights toward high-sentiment names.

    weights_long:   (rebalance_date, ticker, weight) from the base backtest.
    sentiment_signal: date-indexed DataFrame (one column per ticker) of the
                      lagged, smoothed sentiment, first tradable on its date.
    Returns the tilted weights in the same long format.
    """
    pivot = weights_long.pivot(index="rebalance_date", columns="ticker", values="weight")
    signal = sentiment_signal.reindex(columns=pivot.columns)

    tilted_rows: list[dict] = []
    for rebalance_date, base in pivot.iterrows():
        if rebalance_date in signal.index:
            z = signal.loc[rebalance_date].astype(float)
        else:
            z = pd.Series(np.nan, index=pivot.columns)
        z = z.clip(-1.0, 1.0)
        tilt = (1.0 + kappa * z).fillna(1.0)  # no signal -> base weight
        w = base.to_numpy(dtype=float) * tilt.to_numpy(dtype=float)
        w = normalize_capped(w, cap)
        for ticker, weight in zip(pivot.columns, w):
            tilted_rows.append({
                "rebalance_date": rebalance_date,
                "ticker": ticker,
                "weight": round(float(weight), 6),
            })
    return pd.DataFrame(tilted_rows)


def apply_weights(returns_wide: pd.DataFrame, weights_long: pd.DataFrame) -> pd.Series:
    """Apply rebalance weights to returns over the same holding schedule.

    Weights chosen at rebalance date d0 earn returns from the next period
    through the next rebalance date (inclusive), matching the base backtest
    exactly (no look-ahead: a decision on d0 uses only data <= d0).
    """
    pivot = weights_long.pivot(index="rebalance_date", columns="ticker", values="weight")
    rdates = sorted(pivot.index)
    rows: dict[pd.Timestamp, float] = {}
    for i, d0 in enumerate(rdates):
        w = pivot.loc[d0]
        end = rdates[i + 1] if i + 1 < len(rdates) else returns_wide.index[-1]
        holding = returns_wide.loc[(returns_wide.index > d0) & (returns_wide.index <= end)]
        for dt, period in holding.iterrows():
            rows[dt] = float(np.nansum(w.to_numpy(dtype=float) * period.to_numpy(dtype=float)))
    return pd.Series(rows, name="return").sort_index()


def fused_fund_returns(
    returns_wide: pd.DataFrame,
    method: str,
    sentiment_signal: pd.DataFrame,
    window: int = 504,
    rebalance: int = 21,
    family: str = "equity",
    kappa: float = 1.0,
) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Run the base fund, tilt it with sentiment, and return both.

    Returns (base_returns, base_weights, fused_returns, fused_weights).
    """
    base_returns, base_weights = walk_forward_backtest(
        returns_wide, method, window=window, rebalance=rebalance, family=family
    )
    fused_weights = apply_sentiment_tilt(
        base_weights, sentiment_signal, kappa=kappa, cap=0.20
    )
    fused_returns = apply_weights(returns_wide, fused_weights)
    return base_returns, base_weights, fused_returns, fused_weights
