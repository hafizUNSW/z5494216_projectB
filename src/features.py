"""Station 2 - Feature Engineering & Text Assembly.

Builds:
  - Daily simple returns (equities and crypto) — the primary feature
  - Rolling 21-day volatility (optional, for risk evaluation)
  - Rolling 21-day Sharpe ratio (optional, for risk-adjusted performance)
  - Crypto returns merged onto equity trading calendar
  - Combined returns panel (long format)
  - Daily headline panel aligned to the equity trading calendar

Calendar handling (critical — read before modifying):
  1. Compute returns WITHIN each panel first:
     - Equities: ~252 trading days/yr (weekdays only, market holidays excluded)
     - Crypto: ~365 days/yr (trades every calendar day including weekends)
  2. THEN left-merge crypto returns onto the equity trading calendar.
     This intentionally drops weekend-only crypto moves — correct, because
     a fund trading on equity days cannot act on weekend crypto moves.
  3. NEVER merge price levels across calendars and difference afterwards.
     That fabricates spurious returns on days where only one asset class traded.

Timezone handling:
  - Headline dates are UTC/timezone-aware (datetime64[us, UTC]).
  - Price dates are tz-naive (datetime64[ns]).
  - Normalise timezone and dtype BEFORE any merge, or merge_asof will error.

Part A is assembly only. Scoring sentiment (Station 3) happens in Part B.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature 1: Daily simple returns (required)
# ---------------------------------------------------------------------------

def compute_returns(prices: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Compute simple daily returns per ticker using adjusted close.

    Why useful:
      Daily returns are the single feature the portfolio optimiser in Part B
      requires. They measure the day-to-day percentage change in investment
      value, which feeds into covariance estimation, risk budgets, and
      Sharpe-ratio optimisation.

    How computed:
      r_t = (adjClose_t / adjClose_{t-1}) - 1
      Applied per ticker via pct_change() — never across tickers.

    What to expect:
      - Mean close to zero (~0.05% for equities, ~0.2% for crypto).
      - Standard deviation ~2.3% for equities, ~5.2% for crypto.
      - Fat tails (excess kurtosis > 3): extreme moves are more common
        than a normal distribution predicts.
      - Occasional outliers (>30% in a single day): genuine market events
        (e.g. OXY during the March 2020 oil crash).

    Provenance:
      - Source: equity_prices or crypto_prices from src/data_access.py
      - Derived: return = adjClose_t / adjClose_{t-1} - 1
      - Frequency: daily (same as source)
      - Adjusted close accounts for dividends and splits.
    """
    df = prices.sort_values(["ticker", "date"]).copy()
    df["return"] = df.groupby("ticker")[price_col].pct_change()
    return df


# ---------------------------------------------------------------------------
# Feature 2: Rolling 21-day volatility (optional)
# ---------------------------------------------------------------------------

def compute_rolling_volatility(
    returns_df: pd.DataFrame,
    window: int = 21,
    annualise: bool = True,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Compute rolling realised volatility over a trailing window.

    Why useful:
      Rolling volatility captures time-varying risk — it shows when markets
      are calm (low vol) versus stressed (high vol). The portfolio optimiser
      in Part B may use recent volatility to scale position sizes or to
      evaluate risk-adjusted performance. A 21-day (1-month) window balances
      responsiveness to regime changes against noise reduction.

    How computed:
      sigma_t = sqrt( (1/(w-1)) * Σ_{i=t-w+1}^{t} (r_i - r̄)^2 ) * sqrt(ann)

      where w = window length (21 days), r̄ is the window mean, and
      ann = 252 for equities (or 365 for crypto if annualise=True).

      This is the sample standard deviation of daily returns over the
      trailing 21-day window, annualised by multiplying by sqrt(trading_days).

    What to expect:
      - Ranges from ~10% to ~60% annualised for equities (low in calm
        markets, spikes during crashes like March 2020).
      - Ranges from ~40% to ~150% for crypto (higher baseline, larger spikes).
      - First 20 observations are NaN (window warmup).

    Provenance:
      - Source: returns (computed from adjClose)
      - Derived: 21-day rolling standard deviation of returns, annualised
      - Frequency: daily (same as returns)
    """
    df = returns_df.copy()
    ann_factor = np.sqrt(trading_days) if annualise else 1.0

    df["rolling_vol"] = (
        df.groupby("ticker")["return"]
        .transform(lambda s: s.rolling(window=window, min_periods=window).std() * ann_factor)
    )
    return df


# ---------------------------------------------------------------------------
# Feature 3: Rolling 21-day Sharpe ratio (optional)
# ---------------------------------------------------------------------------

def compute_rolling_sharpe(
    returns_df: pd.DataFrame,
    window: int = 21,
    rf: float = 0.0,
    annualise: bool = True,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Compute rolling Sharpe ratio over a trailing window.

    Why useful:
      The Sharpe ratio measures risk-adjusted return: how much excess return
      the asset delivers per unit of risk. A rolling window shows whether
      an asset's risk-reward profile is improving or deteriorating over time.
      The optimiser in Part B can use rolling Sharpe to rank assets or to
      tilt portfolios toward consistently risk-reward-efficient names.

    How computed:
      Sharpe_t = (r̄_t - rf) / sigma_t * sqrt(ann)

      where r̄_t is the trailing 21-day mean return, rf is the daily
      risk-free rate (set to 0 here, as stated in PROJECT_BRIEF.md), sigma_t is
      the trailing 21-day standard deviation, and ann = sqrt(252) for
      equities or sqrt(365) for crypto.

      With rf = 0, this simplifies to: Sharpe_t = (r̄_t / sigma_t) * sqrt(ann)

    What to expect:
      - Most values between -2 and +2 for equities (daily Sharpe scaled up).
      - Crypto may show wider range due to higher volatility.
      - Sustained positive values indicate consistently profitable periods;
        negative values indicate the asset is losing money per unit of risk.
      - First 20 observations are NaN (window warmup).

    Provenance:
      - Source: returns (computed from adjClose)
      - Derived: 21-day rolling (mean/std) of returns, annualised
      - Frequency: daily (same as returns)
    """
    df = returns_df.copy()
    ann_factor = np.sqrt(trading_days) if annualise else 1.0

    def _sharpe(s):
        mean_ret = s.rolling(window=window, min_periods=window).mean()
        std_ret = s.rolling(window=window, min_periods=window).std()
        return ((mean_ret - rf) / std_ret) * ann_factor

    df["rolling_sharpe"] = df.groupby("ticker")["return"].transform(_sharpe)
    return df


# ---------------------------------------------------------------------------
# Combined returns panel
# ---------------------------------------------------------------------------

def build_combined_returns_panel(
    equity_returns: pd.DataFrame,
    crypto_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Left-merge crypto returns onto the equity trading calendar.

    Steps:
      1. Pivot crypto returns to wide (date x ticker) on its own 365-day
         calendar. Returns are already computed on the native calendar.
      2. Pivot equity returns to wide (date x ticker) on its ~252-day
         trading calendar.
      3. Left-join crypto onto the equity index — this intentionally drops
         weekend-only crypto moves, which is correct (the fund trades on
         equity days only).
      4. Melt back to long format with columns: date, ticker, return,
         asset_class.

    NEVER merge price levels across the two calendars and difference
    afterwards — that fabricates spurious returns on days where only
    one asset class traded.

    Provenance:
      - Source: equity returns (computed from adjClose) + crypto returns
        (computed from adjClose on native 365-day calendar)
      - Derived: combined panel on equity trading calendar
      - Frequency: ~252 days/yr (equity calendar)
      - Coverage: 50 equities + 10 crypto (crypto NaN on non-trading days
        are dropped in the melt)
    """
    # Pivot crypto to wide (date x ticker) — returns already computed on
    # the native 365-day calendar
    crypto_wide = (
        crypto_returns[["date", "ticker", "return"]]
        .pivot_table(index="date", columns="ticker", values="return")
        .sort_index()
    )

    # Pivot equity to wide (date x ticker) — ~252 trading days/yr
    eq_wide = (
        equity_returns[["date", "ticker", "return"]]
        .pivot_table(index="date", columns="ticker", values="return")
        .sort_index()
    )

    # Left-merge crypto onto equity calendar
    # This drops weekend-only crypto moves — intentional and correct
    combined_wide = eq_wide.join(crypto_wide, how="left")

    # Melt back to long
    combined = combined_wide.reset_index().melt(
        id_vars="date", var_name="ticker", value_name="return"
    )
    combined = combined.dropna(subset=["return"])

    # Tag asset class
    eq_tickers = set(equity_returns["ticker"].unique())
    combined["asset_class"] = combined["ticker"].apply(
        lambda t: "equity" if t in eq_tickers else "crypto"
    )
    return combined.sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Headline text panel assembly
# ---------------------------------------------------------------------------

def assemble_headline_panel(
    headlines: pd.DataFrame,
    equity_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Assemble headlines into a daily panel aligned to the equity trading calendar.

    Steps:
      1. Normalise date to tz-naive (already done in etl, but guard here).
         Headline dates are UTC/timezone-aware; price dates are tz-naive.
         Without normalisation, merge operations silently fail or error.
      2. Map each headline to its equity trading day: same day if trading
         day, otherwise the next trading day. This handles weekends and
         market holidays correctly.
      3. Drop exact-duplicate headlines (ticker+date+title — already done
         in etl, but reinforced here).
      4. Keep raw headline text intact — no stopword removal, no casing
         changes beyond what VADER requires in Part B.

    Why keep raw text:
      VADER (Valence Aware Dictionary and sEntiment Reasoner) relies on
      word valence, capitalisation, punctuation (!, ALL CAPS), and
      negation words (not, never, no). Stripping stopwords, lowercasing,
      or removing punctuation would destroy these signals. The sentiment
      model is built in Part B (Station 3), not here — this station only
      assembles the panel.

    Provenance:
      - Source: news_headlines from src/data_access.py
      - Derived: trade_date mapped to equity trading calendar
      - Frequency: daily (one row per headline; aggregate by trade_date
        + sector for daily counts)
      - Coverage: 50 stocks, headlines (title, url, publisher)
    """
    df = headlines.copy()

    # Ensure tz-naive — headline dates are UTC timezone-aware, prices are not
    if hasattr(df["date"].dtype, "tz") and df["date"].dtype.tz is not None:
        df["date"] = df["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    df["date"] = pd.to_datetime(df["date"])

    # Build the equity trading calendar if not provided
    if equity_dates is None:
        # Use all unique dates from the headlines themselves (they cover equity dates)
        equity_dates = pd.DatetimeIndex(sorted(df["date"].unique()))

    # Forward-asof map: each headline date -> the next equity trading day on or after it.
    # This correctly handles weekends (Sat/Sun headlines -> Monday) and
    # market holidays (e.g. Christmas -> next trading day).
    idx = pd.DatetimeIndex(sorted(equity_dates))
    # searchsorted gives the insertion index; clip to last trading date
    positions = idx.searchsorted(df["date"], side="left")
    positions = np.clip(positions, 0, len(idx) - 1)
    df["trade_date"] = idx[positions]

    # Assemble daily panel: one row per headline, aligned to trade_date
    panel = (
        df[["trade_date", "ticker", "sector", "title", "url", "publisher"]]
        .sort_values(["trade_date", "ticker", "title"])
        .reset_index(drop=True)
    )
    return panel
