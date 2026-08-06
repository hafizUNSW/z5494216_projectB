"""Quantis - your systematic multi-asset fund dashboard (Part B, Station 4).

The investor journey in one app:
  * Compare  - pick funds and compare out-of-sample risk/return.
  * Fact sheet - one fund's growth of $1, drawdown, metrics and current holdings.
  * Allocate - set an allocation across funds and see the blended result.
  * Sentiment - the sector sentiment index over time.

The app reads ONLY four precomputed artifacts (never raw data, never the
sentiment-scoring package, never recomputes backtests), so it stays light
enough for Streamlit Community Cloud:

  results/data/fund_returns.csv
  results/data/fund_weights.csv
  results/data/sector_sentiment_index.csv
  results/tables/performance_metrics.csv

Run locally with:

    streamlit run streamlit_app.py
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "results" / "data"
TABLES = ROOT / "results" / "tables"

# --- Quantis design system (the Economist-style palette carried from Part A) ---
RED = "#E3120B"
BLUE = "#0D5691"
GREY = "#666666"
PAL = [
    "#0D5691", "#E3120B", "#4E8C2C", "#D98B35",
    "#6C3483", "#1AABB8", "#BC4B52", "#4A5568",
    "#5B7553", "#8B6914",
]

FAMILY_LABEL = {"equity": "Equity", "crypto": "Crypto", "combined": "Equity + Crypto"}
METHOD_LABEL = {
    "equal_weight": "Equal Weight",
    "minimum_variance": "Minimum Variance",
    "maximum_sharpe": "Maximum Sharpe",
    "risk_parity": "Risk Parity",
}


def _method_label(method: str) -> str:
    if isinstance(method, str) and method.endswith(" + sentiment"):
        base = METHOD_LABEL.get(method[: -len(" + sentiment")], method)
        return f"{base} + Sentiment"
    return METHOD_LABEL.get(method, method)

st.set_page_config(page_title="Quantis", page_icon="Q", layout="wide")
st.title("Quantis")
st.caption(
    "Systematically managed multi-asset funds - out-of-sample performance, "
    "not in-sample fit. Every number is computed by scripts/run_part_b.py from "
    "the data described in the report."
)


# ---------------------------------------------------------------------------
# Data loading (precomputed results only, cached for the session)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=86_400, show_spinner="Loading fund returns...")
def load_fund_returns() -> pd.DataFrame:
    df = pd.read_csv(DATA / "fund_returns.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=86_400, show_spinner="Loading fund weights...")
def load_fund_weights() -> pd.DataFrame:
    df = pd.read_csv(DATA / "fund_weights.csv")
    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"])
    return df


@st.cache_data(ttl=86_400, show_spinner="Loading sector sentiment...")
def load_sector_sentiment() -> pd.DataFrame:
    df = pd.read_csv(DATA / "sector_sentiment_index.csv")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


@st.cache_data(ttl=86_400, show_spinner="Loading performance metrics...")
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(TABLES / "performance_metrics.csv")


@st.cache_data(ttl=86_400, show_spinner="Loading fund weights...")
def load_current_holdings(fund: str) -> pd.DataFrame:
    """Current holdings = target weights at the fund's most recent rebalance."""
    weights = load_fund_weights()
    sub = weights[weights["fund"] == fund]
    last = sub["rebalance_date"].max()
    top = (sub[sub["rebalance_date"] == last]
           .sort_values("weight", ascending=False)
           .head(10)[["ticker", "weight"]]
           .assign(weight_pct=lambda d: (d["weight"] * 100).round(2)))
    return top


# ---------------------------------------------------------------------------
# Small analysis helpers (all self-contained so the app stays light)
# ---------------------------------------------------------------------------
def fund_return_series(fund: str) -> pd.Series:
    sub = load_fund_returns()
    return sub.loc[sub["fund"] == fund, ["date", "return"]].set_index("date")["return"].sort_index()


def growth_of_one(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    growth = growth_of_one(returns)
    return growth / growth.cummax() - 1.0


def summary_metrics(returns: pd.Series, periods_per_year: int = 252) -> dict:
    growth = growth_of_one(returns)
    ann_ret = float(growth.iloc[-1]) ** (periods_per_year / len(returns)) - 1.0
    ann_vol = float(returns.std(ddof=1)) * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    mdd = float(drawdown_series(returns).min())
    return {
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "growth_of_1": float(growth.iloc[-1]),
    }


def _style(fig: plt.Figure) -> plt.Figure:
    """Apply the Quantis look to a freshly-made figure."""
    for ax in fig.get_axes():
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_facecolor("white")
        ax.grid(True, axis="y", alpha=0.4, color="#E0E0E0", linewidth=0.5)
    fig.patch.set_facecolor("white")
    return fig


def _fund_colors(funds: list[str]) -> dict:
    return {fund: PAL[i % len(PAL)] for i, fund in enumerate(funds)}


def _pct(x: float) -> str:
    return f"{x:+.1%}" if np.isfinite(x) else "n/a"


# ---------------------------------------------------------------------------
# Compare tab
# ---------------------------------------------------------------------------
def compare_tab() -> None:
    st.subheader("Compare the funds")
    metrics = load_metrics()
    metrics["family_label"] = metrics["family"].map(FAMILY_LABEL)
    metrics["method_label"] = metrics["method"].map(_method_label)
    metrics["label"] = metrics["fund"]

    funds = list(metrics["fund"])
    defaults = [f for f in funds if f.startswith("Equity")]
    selected = st.multiselect(
        "Funds to compare", funds, default=defaults, key="compare_funds"
    )

    if not selected:
        st.info("Pick at least one fund.")
        return

    view = metrics[metrics["fund"].isin(selected)].copy()
    view["return"] = view["annualized_return"].map(lambda v: f"{v:.2%}")
    view["volatility"] = view["annualized_volatility"].map(lambda v: f"{v:.2%}")
    view["Sharpe"] = view["sharpe_ratio"].map(lambda v: f"{v:.2f}")
    view["max drawdown"] = view["max_drawdown"].map(lambda v: f"{v:.1%}")
    view["growth of $1"] = view["growth_of_1"].map(lambda v: f"${v:.2f}")

    cols = st.columns(5)
    labels = ["return", "volatility", "Sharpe", "max drawdown", "growth of $1"]
    for col, lab in zip(cols, labels):
        col.metric(lab, view[lab].iloc[0])
    st.dataframe(
        view[["fund", "family_label", "method_label", "return", "volatility",
              "Sharpe", "max drawdown", "growth of $1", "oos_period"]]
        .rename(columns={
            "fund": "Fund", "family_label": "Universe",
            "method_label": "Method", "return": "Annualised return",
            "volatility": "Annualised vol", "Sharpe": "Sharpe",
            "max drawdown": "Max drawdown", "growth of $1": "Growth of $1",
            "oos_period": "Out-of-sample period",
        }),
        hide_index=True,
    )

    growth = pd.DataFrame({f: growth_of_one(fund_return_series(f)) for f in selected})
    fig, ax = plt.subplots(figsize=(11, 4.6))
    colors = _fund_colors(selected)
    for f in selected:
        ax.plot(growth.index, growth[f].to_numpy(), linewidth=1.4,
                color=colors[f], label=f)
    ax.set_title("Growth of $1 (out-of-sample)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value of $1")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    st.pyplot(_style(fig))

    fig2, ax2 = plt.subplots(figsize=(11, 4.6))
    for f in selected:
        m = metrics[metrics["fund"] == f].iloc[0]
        ax2.scatter(m["annualized_volatility"], m["annualized_return"],
                    color=colors[f], s=70, label=f)
    ax2.set_title("Return vs risk (out-of-sample)", fontweight="bold")
    ax2.set_xlabel("Annualised volatility")
    ax2.set_ylabel("Annualised return")
    ax2.legend(fontsize=8, loc="best")
    st.pyplot(_style(fig2))


# ---------------------------------------------------------------------------
# Fact sheet tab
# ---------------------------------------------------------------------------
def fact_sheet_tab() -> None:
    st.subheader("Fund fact sheet")
    fund = st.selectbox("Fund", list(load_metrics()["fund"]), key="fact_fund")

    r = fund_return_series(fund)
    m = summary_metrics(r)
    growth = growth_of_one(r)
    dd = drawdown_series(r)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Annualised return", _pct(m["annualized_return"]))
    c2.metric("Annualised vol", f"{m['annualized_volatility']:.1%}")
    c3.metric("Sharpe ratio", f"{m['sharpe_ratio']:.2f}")
    c4.metric("Max drawdown", f"{m['max_drawdown']:.1%}")
    c5.metric("Growth of $1", f"${m['growth_of_1']:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(growth.index, growth.to_numpy(), color=BLUE, linewidth=1.4)
    axes[0].set_title("Growth of $1", fontweight="bold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Value of $1")
    axes[1].plot(dd.index, dd.to_numpy(), color=RED, linewidth=1.4)
    axes[1].set_title("Drawdown", fontweight="bold")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Drawdown")
    fig.tight_layout()
    st.pyplot(_style(fig))

    top = load_current_holdings(fund)
    if len(top):
        st.markdown(f"**Current holdings** (top {len(top)} positions at the latest rebalance)")
        fig2, ax2 = plt.subplots(figsize=(11, 0.35 * len(top) + 1.6))
        ax2.barh(top["ticker"].to_numpy(), top["weight_pct"].to_numpy(),
                 color=BLUE, height=0.6)
        ax2.set_xlabel("Weight (%)")
        ax2.set_title("Current holdings", fontweight="bold")
        fig2.tight_layout()
        st.pyplot(_style(fig2))

    weights = load_fund_weights()
    wf = weights[weights["fund"] == fund]
    if len(wf):
        top_tickers = list(top["ticker"])
        pivot = (wf[wf["ticker"].isin(top_tickers)]
                 .pivot(index="rebalance_date", columns="ticker", values="weight")
                 .fillna(0.0).sort_index())
        if len(pivot):
            fig3, ax3 = plt.subplots(figsize=(11, 4.2))
            colors = _fund_colors(list(pivot.columns))
            for t in pivot.columns:
                ax3.plot(pivot.index, pivot[t].to_numpy(), linewidth=1.1,
                         color=colors[t], label=t)
            ax3.set_title("Weights over time (top current holdings)", fontweight="bold")
            ax3.set_xlabel("Rebalance date")
            ax3.set_ylabel("Weight")
            ax3.set_ylim(0, 1)
            ax3.legend(fontsize=8, ncol=3, loc="upper left")
            st.pyplot(_style(fig3))


# ---------------------------------------------------------------------------
# Allocate tab
# ---------------------------------------------------------------------------
def allocate_tab() -> None:
    st.subheader("Set your allocation")
    st.caption(
        "Split your money across the funds below. Weights are normalised to 100% "
        "and the blended portfolio is built from the funds' out-of-sample daily "
        "returns (a fund with no trading day that day counts as cash)."
    )

    all_funds = list(load_metrics()["fund"])
    chosen = st.multiselect("Funds", all_funds, default=all_funds[:2], key="alloc_funds")
    if not chosen:
        st.info("Pick at least one fund.")
        return

    weights_input: dict[str, float] = {}
    cols = st.columns(min(len(chosen), 4))
    for i, fund in enumerate(chosen):
        with cols[i % len(cols)]:
            weights_input[fund] = st.slider(
                fund, 0, 100, int(100 / len(chosen)), step=5, key=f"alloc_{fund}"
            )
    total = sum(weights_input.values())
    if total <= 0:
        st.info("Allocate something.")
        return

    alloc = {f: w / total for f, w in weights_input.items()}

    daily = load_fund_returns()
    wide = daily.pivot(index="date", columns="fund", values="return")
    parts = []
    for fund, w in alloc.items():
        r = wide[fund].reindex(wide.index).fillna(0.0)
        parts.append(w * r)
    blended = sum(parts)

    m = summary_metrics(blended)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Blended annualised return", _pct(m["annualized_return"]))
    c2.metric("Blended annualised vol", f"{m['annualized_volatility']:.1%}")
    c3.metric("Blended Sharpe", f"{m['sharpe_ratio']:.2f}")
    c4.metric("Blended max drawdown", f"{m['max_drawdown']:.1%}")
    c5.metric("Blended growth of $1", f"${m['growth_of_1']:.2f}")

    growth = growth_of_one(blended)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    colors = _fund_colors(chosen)
    for fund in chosen:
        g = growth_of_one(wide[fund].dropna())
        ax.plot(g.index, g.to_numpy(), color=colors[fund], linewidth=1.0,
                alpha=0.6, label=fund)
    ax.plot(growth.index, growth.to_numpy(), color=RED, linewidth=2.0,
            label=f"Blended (${total} split {', '.join(f'{w:.0%}' for w in alloc.values())})")
    ax.set_title("Your allocation vs the funds", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value of $1")
    ax.legend(fontsize=8, loc="upper left")
    st.pyplot(_style(fig))


# ---------------------------------------------------------------------------
# Sentiment tab
# ---------------------------------------------------------------------------
def sentiment_tab() -> None:
    st.subheader("Sector sentiment index")
    st.caption(
        "Daily average VADER compound score of finance-lexicon-extended sentiment, "
        "equal-weighted across tickers within each sector. The signal is lagged one "
        "trading day before it is used, so no decision looks ahead."
    )

    sector_df = load_sector_sentiment()
    sector = st.selectbox("Sector", list(sector_df["sector"].unique()), key="sent_sector")
    sub = sector_df[sector_df["sector"] == sector].set_index("trade_date")

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(sub.index, sub["sentiment"].to_numpy(), color=BLUE, linewidth=1.2,
            label="Sentiment (same day)")
    lag = sub["sentiment_lag1"].dropna()
    ax.plot(lag.index, lag.to_numpy(), color=RED, linewidth=1.0, alpha=0.8,
            label="Sentiment (lagged 1 trading day - what trades can use)")
    ax.axhline(0.0, color=GREY, linewidth=0.8, linestyle="--")
    ax.set_title(f"{sector} sector sentiment over time", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment (compound)")
    ax.legend(fontsize=8, loc="best")
    st.pyplot(_style(fig))

    monthly = (sub["sentiment"].resample("ME").mean().dropna()
               .reset_index().rename(columns={"trade_date": "Month", "sentiment": "Sentiment"}))
    monthly["Month"] = monthly["Month"].dt.strftime("%Y-%m")
    st.markdown("**Monthly average** (all sectors)")
    monthly_all = (sector_df.groupby([pd.Grouper(key="trade_date", freq="ME"), "sector"])
                   ["sentiment"].mean().reset_index())
    monthly_all["Month"] = monthly_all["trade_date"].dt.strftime("%Y-%m")
    monthly_wide = monthly_all.pivot(index="Month", columns="sector", values="sentiment")
    st.dataframe(monthly_wide.round(3), hide_index=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Selected sector: mean", f"{sub['sentiment'].mean():+.4f}")
    c2.metric("Selected sector: std", f"{sub['sentiment'].std():.4f}")
    c3.metric("Days above neutral", f"{(sub['sentiment'] > 0).mean():.0%}")
    st.caption(
        "The sentiment index feeds the equity-fund tilt (fusion); the "
        "before-vs-after evidence is a report exhibit (results/tables/"
        "fusion_comparison.csv), not recomputed here."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    tab_compare, tab_fact, tab_alloc, tab_sent = st.tabs(
        ["Compare", "Fact sheet", "Allocate", "Sentiment"]
    )
    with tab_compare:
        compare_tab()
    with tab_fact:
        fact_sheet_tab()
    with tab_alloc:
        allocate_tab()
    with tab_sent:
        sentiment_tab()


main()
