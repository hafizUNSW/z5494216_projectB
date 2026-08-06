"""Unit tests for src.sentiment (extended VADER + sector index).

Runnable standalone (`python tests/test_sentiment.py`) or via pytest.
Uses small synthetic headline frames; needs nltk (requirements-dev.txt).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from src import sentiment as s


def make_panel(n: int = 40, k: int = 3) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=n)
    tickers = [f"T{i}" for i in range(k)]
    rows = []
    for i, dt in enumerate(dates):
        for tk in tickers:
            rows.append({
                "trade_date": dt, "ticker": tk,
                "sector": "Technology" if tk == "T0" else "Energy",
                "title": "A routine market update for the session.",
            })
    return pd.DataFrame(rows)


def test_extension_does_not_leak_into_plain_analyzer():
    # A fresh VADER analyzer must never see the finance phrases, even after
    # build_analyzer() has run in this process (NLTK 3.10 class-level dicts).
    s.build_analyzer()
    plain = SentimentIntensityAnalyzer()
    assert "earnings beat" not in plain.constants.SPECIAL_CASE_IDIOMS
    assert "sharply" not in plain.constants.BOOSTER_DICT
    assert plain.polarity_scores("The company reported an earnings beat.")["compound"] == 0.0


def test_phrases_fire_in_extended_analyzer():
    a = s.build_analyzer()
    assert a.polarity_scores("The company reported an earnings beat.")["compound"] > 0.5
    assert a.polarity_scores("Analysts issued a downgrade.")["compound"] < 0.0


def test_head_word_needed_for_phrase_lookup():
    # The Week 8 trap: a phrase whose head word is not in the lexicon does
    # nothing. Every installed phrase must have its head word present.
    a = s.build_analyzer()
    for phrase in s.FINANCE_PHRASES:
        head = phrase.split()[-1]
        assert head in a.lexicon, f"missing head word for {phrase!r}"


def test_rejected_terms_unchanged():
    a = s.build_analyzer()
    before = SentimentIntensityAnalyzer().polarity_scores("The firm reduced its debt.")["compound"]
    after = a.polarity_scores("The firm reduced its debt.")["compound"]
    assert np.isclose(before, after)


def test_score_headlines_adds_compound():
    panel = make_panel()
    scored = s.score_headlines(panel)
    assert "compound" in scored.columns
    assert len(scored) == len(panel)
    assert scored["compound"].between(-1, 1).all()


def test_ticker_daily_sentiment():
    scored = s.score_headlines(make_panel())
    daily = s.ticker_daily_sentiment(scored)
    assert {"trade_date", "ticker", "sector", "sentiment", "n_headlines"}.issubset(daily.columns)
    assert len(daily) == len(scored)


def test_sector_sentiment_index_aligned_and_lagged():
    scored = s.score_headlines(make_panel())
    daily = s.ticker_daily_sentiment(scored)
    calendar = pd.DatetimeIndex(sorted(scored["trade_date"].unique()))
    index = s.sector_sentiment_index(daily, calendar)
    assert {"trade_date", "sector", "sentiment", "sentiment_lag1"}.issubset(index.columns)
    # lag1 must be the sentiment from the previous calendar day
    g = index[index["sector"] == "Technology"].sort_values("trade_date")
    assert np.allclose(g["sentiment_lag1"].iloc[1:].to_numpy(),
                       g["sentiment"].iloc[:-1].to_numpy())


def test_ticker_signal_wide_is_look_ahead_safe():
    scored = s.score_headlines(make_panel())
    daily = s.ticker_daily_sentiment(scored)
    calendar = pd.DatetimeIndex(sorted(scored["trade_date"].unique()))
    signal = s.ticker_signal_wide(daily, calendar, lag=1, smooth=5)
    # signal at t equals the mean of aligned sentiment over the window ending
    # at t-1 - nothing from t or later.
    aligned = (daily.set_index(["trade_date", "ticker"])["sentiment"]
               .unstack("ticker").reindex(calendar).ffill().fillna(0.0))
    for dt in signal.index[5:20]:
        t0 = calendar[calendar < dt][-5]
        expected = aligned.loc[t0].mean()
        assert np.isclose(signal.loc[dt, "T0"], expected, atol=1e-9), dt


def test_lexicon_coverage_counts():
    scored = s.score_headlines(make_panel())
    cov = s.lexicon_coverage(scored)
    assert cov[cov["metric"] == "headlines scored"]["value"].iloc[0] == len(scored)
    assert cov[cov["metric"] == "share non-zero"]["value"].iloc[0] == \
        (scored["compound"] != 0.0).mean()


def test_lagged_attention_never_uses_same_day_news():
    # A 100-headline day must not lift its own attention: only days BEFORE t
    # count. Day index 4 has all the news, horizon 2.
    dates = pd.bdate_range("2022-01-03", periods=8)
    counts = pd.DataFrame({"date": dates, "T0": [0, 0, 0, 0, 100, 0, 0, 0]})
    wide = counts.set_index("date")
    att = s.lagged_attention(wide, horizon=2)
    assert att["T0"].iloc[4] == 0.0            # day 4 news not in day-4 attention
    assert np.isclose(att["T0"].iloc[5], 50.0)  # mean of days 3 and 4


def test_forward_vol_uses_only_future_returns():
    # A +0.1 return spike at day index 4 must show up in fwd_vol ONLY for the
    # t that look FORWARD through it: t = 3 (horizon 2 window days 4-5), never
    # at t = 4 or later.
    dates = pd.bdate_range("2022-01-03", periods=10)
    rets = pd.DataFrame({"date": dates, "T0": [0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0]})
    vol = s.forward_realized_vol(rets.set_index("date"), horizon=2)
    assert vol["T0"].iloc[3] > 0.0
    assert vol["T0"].iloc[4] == 0.0
    assert vol["T0"].iloc[7] == 0.0


def test_news_volume_volatility_crosscheck():
    # News on days 10-14 (5 a day) makes attention high on days 15-19; returns
    # that alternate +/- 0.1 on days 16-20 make forward vol high exactly where
    # attention is high. The cross-check must see a positive relationship.
    dates = pd.bdate_range("2022-01-03", periods=28)
    ticker = "T0"
    counts = pd.Series(0.0, index=dates)
    counts.iloc[10:15] = 5.0
    daily = pd.DataFrame({
        "trade_date": dates,
        "ticker": ticker,
        "sector": "Technology",
        "sentiment": 0.0,
        "n_headlines": counts.to_numpy(),
    })
    rets = pd.Series(0.0, index=dates)
    rets.iloc[16:21] = [0.1, -0.1, 0.1, -0.1, 0.1]
    wide = pd.DataFrame({ticker: rets.to_numpy()}, index=dates)

    buckets, rho = s.news_volume_volatility_crosscheck(daily, wide, horizon=5)
    assert {"sector", "attention_quintile", "n_days",
            "mean_forward_vol_annualised"}.issubset(buckets.columns)
    assert buckets.groupby("sector")["attention_quintile"].nunique().eq(5).all()
    assert {"sector", "n_days", "spearman_rho"}.issubset(rho.columns)
    assert rho["spearman_rho"].between(-1, 1).all()
    q1 = buckets.loc[buckets["attention_quintile"] == "Q1 (least news)",
                     "mean_forward_vol_annualised"].iloc[0]
    q5 = buckets.loc[buckets["attention_quintile"] == "Q5 (most news)",
                     "mean_forward_vol_annualised"].iloc[0]
    assert q5 > q1
    assert rho["spearman_rho"].iloc[0] > 0.0


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  {t.__name__} OK")


if __name__ == "__main__":
    _run()
    print("all sentiment tests passed")
