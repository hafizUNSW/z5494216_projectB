"""Station 3 - sentiment model and sector index from news headlines.

The model is VADER (from nltk) extended with a finance-domain lexicon, the
Station 3 model step built on the Part A daily headline panel.

Decisions (all stated for the report):
  - Headline text is scored RAW: VADER reads casing, punctuation, booster and
    negation words, so no stopword removal, lowercasing, or punctuation
    stripping before scoring.
  - The extension installs finance words, multi-word phrases, and booster
    words in memory (never into the installed package). A phrase only works if
    its head word is in the lexicon, so missing head words are added with a
    tiny +/- valence, exactly as the Week 8 course method does.
  - A headline with no dictionary hits still yields a compound score (often 0);
    a score of zero is "no signal", not evidence of neutrality.
  - Ticker-days with no headlines are treated as NEUTRAL (0): absence of news
    carries no directional signal, and keeping the panel dense avoids
    survivorship artefacts in the sector average.
  - The sector index equal-weights tickers within each sector each day.
  - The signal is LAGGED one trading day so day t's trade uses only sentiment
    from day t-1 or earlier (a Saturday or Monday headline aligned to Monday is
    first usable for Tuesday). A smoothed (21-day) lagged version is produced
    for the fusion so the tilt uses a rolling signal, not a single noisy day.
"""
from __future__ import annotations

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# The extended finance lexicon (Innovation: custom finance sentiment tool)
# ---------------------------------------------------------------------------
# Terms and valences are our own curation, built from the Part A finance
# lexicon categories plus standard market vocabulary, on VADER's -4..+4 scale.
FINANCE_WORDS: dict[str, float] = {
    # Earnings / results
    "beat": 3.2, "miss": -3.0, "surprise": 0.8, "guidance": 0.0,
    "forecast": 0.4, "outlook": 0.4, "profit": 2.2, "earnings": 0.3,
    "revenue": 0.3, "loss": -2.4,
    # Ratings
    "upgrade": 2.6, "downgrade": -2.6, "overweight": 1.2, "underweight": -1.2,
    "outperform": 2.4, "underperform": -2.4, "buy": 1.6, "sell": -1.8,
    "hold": -0.2, "reiterate": 0.3, "initiate": 0.2, "target": 0.0,
    "rating": 0.0, "neutral": 0.0,
    # Corporate actions
    "merger": 0.6, "acquisition": 0.5, "acquire": 0.6, "deal": 0.4,
    "ipo": 0.8, "buyback": 1.8, "dividend": 1.5, "restructuring": -1.5,
    "layoffs": -2.4, "partnership": 1.2, "settlement": -1.2,
    # Macro
    "recession": -2.5, "inflation": -1.2, "tariff": -1.5, "sanctions": -2.2,
    "rate": -0.2, "yield": -0.2, "unemployment": -1.5, "hawkish": -1.8,
    "dovish": 1.5, "stimulus": 2.0,
    # Risk / legal
    "risk": -1.5, "debt": -1.5, "lawsuit": -2.0, "litigation": -2.0,
    "fraud": -3.4, "scandal": -3.0, "bankruptcy": -3.5, "default": -3.0,
    "violation": -2.0, "investigation": -1.8, "impairment": -2.0,
    "writedown": -2.3, "subpoena": -2.6, "probe": -1.5,
    # Market direction
    "surge": 3.0, "soar": 3.0, "rally": 2.6, "gain": 1.9, "rise": 1.2,
    "jump": 1.6, "crash": -3.2, "plunge": -3.0, "drop": -1.5, "fall": -1.5,
    "decline": -1.5, "slump": -2.2, "selloff": -2.5, "correction": -1.5,
    "bear": -2.0, "bull": 2.0, "bullish": 2.5, "bearish": -2.5,
    "momentum": 1.2, "breakout": 2.2, "volatility": -1.2,
    "recovery": 1.9, "robust": 2.0, "strong": 1.9, "weak": -1.9,
    "warning": -1.8, "milestone": 1.5, "concern": -1.2, "weakness": -1.6,
}

# Multi-word phrases, held whole by VADER's SPECIAL_CASES.
FINANCE_PHRASES: dict[str, float] = {
    "earnings beat": 3.0, "beats estimates": 2.8, "tops estimates": 2.6,
    "record revenue": 2.2, "record profit": 2.4, "above expectations": 1.8,
    "raised guidance": 2.5, "raises guidance": 2.5, "share buyback": 1.8,
    "stock buyback": 1.8, "cost cuts": 1.5, "earnings miss": -3.0,
    "misses estimates": -2.8, "below expectations": -1.8, "guidance cut": -2.5,
    "cuts guidance": -2.5, "cut guidance": -2.5, "profit warning": -3.0,
    "going concern": -3.5, "covenant breach": -3.0, "material weakness": -2.5,
    "credit downgrade": -3.0,
}

# Words that raise or lower the valence of the next word.
FINANCE_BOOSTERS: dict[str, float] = {
    "sharply": 0.293, "materially": 0.293, "steeply": 0.293,
    "significantly": 0.293, "dramatically": 0.293, "modestly": -0.293,
    "marginally": -0.293, "slightly": -0.293,
}


def build_analyzer() -> SentimentIntensityAnalyzer:
    """Build VADER with the finance extension installed in memory.

    The installed package is never modified; every run rebuilds the same
    extended dictionary from this module, which keeps the change reviewable.
    """
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update({w: float(v) for w, v in FINANCE_WORDS.items()})

    # NLTK 3.10 keeps the phrase/booster/negation tables on a VaderConstants
    # instance, but those dicts are shared class attributes: mutating them leaks
    # into every other analyzer in the process, so a "baseline" analyser built
    # after this call would silently see the extension (the same silent trap
    # the course documents for vaderSentiment 3.3.2's module-level dicts).
    # Copy the tables into instance-owned containers before editing them.
    analyzer.lexicon = dict(analyzer.lexicon)
    analyzer.constants.BOOSTER_DICT = dict(analyzer.constants.BOOSTER_DICT)
    analyzer.constants.SPECIAL_CASE_IDIOMS = dict(analyzer.constants.SPECIAL_CASE_IDIOMS)
    analyzer.constants.NEGATE = set(analyzer.constants.NEGATE)

    # Phrases need their head (last) word in the lexicon or VADER never looks
    # for them - it only checks phrase positions where the token is a hit.
    head_words: dict[str, float] = {}
    for phrase, valence in FINANCE_PHRASES.items():
        analyzer.constants.SPECIAL_CASE_IDIOMS[phrase] = float(valence)
        head = phrase.split()[-1]
        if head not in analyzer.lexicon:
            head_words[head] = 0.1 if valence > 0 else -0.1
    analyzer.lexicon.update(head_words)

    analyzer.constants.BOOSTER_DICT.update(FINANCE_BOOSTERS)
    return analyzer


def score_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """Score every headline with the extended VADER model.

    Input: the Part A daily headline panel (trade_date, ticker, sector, title,
    ...). Output: the same frame plus a compound score per headline.
    Text is scored raw - never pre-cleaned.
    """
    if "title" not in panel.columns:
        raise ValueError("panel needs a 'title' column")
    analyzer = build_analyzer()
    df = panel.copy()
    df["compound"] = df["title"].map(
        lambda t: analyzer.polarity_scores(str(t))["compound"]
    )
    return df


def ticker_daily_sentiment(scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate headline scores to one sentiment per ticker-trading-day.

    Ticker-days with no headlines are treated as neutral (0) later, when the
    panel is reindexed onto the trading calendar.
    """
    daily = (
        scores.groupby(["trade_date", "ticker", "sector"], as_index=False)
        .agg(sentiment=("compound", "mean"),
             n_headlines=("compound", "size"))
    )
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    return daily.sort_values(["trade_date", "ticker"]).reset_index(drop=True)


def _on_calendar(sentiment: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex per-ticker sentiment onto the full trading calendar, neutral fill."""
    daily = sentiment[["trade_date", "ticker", "sector", "sentiment"]].copy()
    tickers = daily["ticker"].unique()
    idx = pd.MultiIndex.from_product([calendar, tickers], names=["trade_date", "ticker"])
    wide = daily.set_index(["trade_date", "ticker"])["sentiment"].reindex(idx, fill_value=0.0)
    return wide.unstack("ticker")


def sector_sentiment_index(
    ticker_sentiment: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Equal-weight daily sector index, with a one-trading-day lag.

    Returns long format: trade_date, sector, sentiment (aligned) and
    sentiment_lag1 (first usable for trading on the NEXT day).
    """
    wide = _on_calendar(ticker_sentiment, calendar)
    sector_map = (
        ticker_sentiment[["ticker", "sector"]]
        .drop_duplicates()
        .set_index("ticker")["sector"]
    )
    sectors = sorted(sector_map.unique())
    rows: list[dict] = []
    for sector in sectors:
        tickers = sector_map[sector_map == sector].index
        sub = wide[tickers.intersection(wide.columns)]
        aligned = sub.mean(axis=1, skipna=True)
        aligned = aligned.reindex(calendar, fill_value=0.0)
        lagged = aligned.shift(1)  # tradable from the NEXT trading day
        rows.append(pd.DataFrame({
            "trade_date": aligned.index,
            "sector": sector,
            "sentiment": aligned.values,
            "sentiment_lag1": lagged.values,
        }))
    return pd.concat(rows).sort_values(["sector", "trade_date"]).reset_index(drop=True)


def ticker_signal_wide(
    ticker_sentiment: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    lag: int = 1,
    smooth: int = 21,
) -> pd.DataFrame:
    """Look-ahead-safe smoothed per-ticker sentiment for the fusion.

    raw aligned sentiment -> neutral fill -> shift(lag) -> trailing `smooth`-day
    mean. The result on date t uses only sentiment from dates < t, so it is
    safe to trade on at t. Returns a DataFrame indexed by date, one column per
    ticker.
    """
    wide = _on_calendar(ticker_sentiment, calendar)
    signal = wide.shift(lag).rolling(smooth, min_periods=min(5, smooth)).mean()
    return signal


def lexicon_coverage(scores: pd.DataFrame) -> pd.DataFrame:
    """Count how many headlines were 'woken up' by the finance extension.

    Diagnostic for the report: how many headlines score a non-zero compound
    with the extended model. (Input is the scored panel.)
    """
    total = len(scores)
    nonzero = int((scores["compound"] != 0.0).sum())
    return pd.DataFrame({
        "metric": ["headlines scored", "non-zero compound", "share non-zero"],
        "value": [total, nonzero, (nonzero / total) if total else 0.0],
    })


def lagged_attention(wide_counts: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Trailing `horizon`-day mean headline count, shifted one trading day.

    attention_t uses only counts from days t-horizon..t-1 - the day-t count is
    never part of its own attention, so the measure never looks ahead.
    """
    return wide_counts.shift(1).rolling(horizon, min_periods=1).mean()


def forward_realized_vol(
    returns_wide: pd.DataFrame,
    horizon: int,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Annualised standard deviation of the NEXT `horizon` daily returns.

    fwd_vol_t = std(returns over days t+1..t+horizon). Purely forward-looking:
    no day-t or earlier return enters the window, so nothing from the past
    leaks into the forward measure.
    """
    return returns_wide.rolling(horizon).std().shift(-horizon) * (periods_per_year ** 0.5)


def news_volume_volatility_crosscheck(
    ticker_daily: pd.DataFrame,
    returns_wide: pd.DataFrame,
    horizon: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """News attention vs forward realised volatility (research cross-check).

    The thesis the funds rely on: a ticker's news attention carries information
    about its *future* risk, not just its same-day price move. For every
    ticker-trading-day we pair a strictly-lagged attention measure
    (``lagged_attention``) with a strictly-forward volatility measure
    (``forward_realized_vol``). It is a diagnostic for the report - no trade
    ever uses it - so pairing past attention with future volatility is the
    correct, honest framing.

    Returns (buckets, rho):
      - buckets: long frame, one row per sector x attention-quintile, with
        n_days and the mean annualised forward volatility per bucket.
      - rho: one row per sector with the pooled Spearman correlation between
        attention and forward volatility (n_days, spearman_rho).
    """
    daily = ticker_daily[["trade_date", "ticker", "n_headlines"]].copy()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    calendar = pd.DatetimeIndex(sorted(returns_wide.index))
    wide_counts = (
        daily.pivot_table(index="trade_date", columns="ticker", values="n_headlines")
        .reindex(calendar, fill_value=0.0)
    )
    attention = lagged_attention(wide_counts, horizon)
    fwd_vol = forward_realized_vol(
        returns_wide.reindex(calendar).sort_index(), horizon
    )

    sector_map = (
        ticker_daily[["ticker", "sector"]].drop_duplicates().set_index("ticker")["sector"]
    )
    parts: list[pd.DataFrame] = []
    for ticker in wide_counts.columns:
        if ticker not in sector_map.index:
            continue
        pair = pd.DataFrame({
            "attention": attention[ticker],
            "fwd_vol": fwd_vol[ticker],
        }).dropna()
        if pair.empty:
            continue
        pair["ticker"] = ticker
        pair["sector"] = sector_map[ticker]
        parts.append(pair)
    if not parts:
        return pd.DataFrame(), pd.DataFrame()
    pooled = pd.concat(parts, ignore_index=True)

    labels = ["Q1 (least news)", "Q2", "Q3", "Q4", "Q5 (most news)"]
    pooled["attention_quintile"] = pd.qcut(
        pooled["attention"].rank(method="first"), 5, labels=labels
    )
    buckets = (
        pooled.groupby(["sector", "attention_quintile"], observed=True)
        .agg(n_days=("fwd_vol", "size"),
             mean_forward_vol_annualised=("fwd_vol", "mean"))
        .reset_index()
    )

    rho_rows: list[dict] = []
    for sector, g in pooled.groupby("sector", observed=True):
        rho_rows.append({
            "sector": sector,
            "n_days": len(g),
            "spearman_rho": round(float(
                g["attention"].corr(g["fwd_vol"], method="spearman")
            ), 4),
        })
    return buckets, pd.DataFrame(rho_rows)
