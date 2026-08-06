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

Charts are interactive Plotly figures in a dark, fintech-style design system
(soft area fills, donuts for holdings, range sliders, unified hover, sign-coded
green/red tiles).

Run locally with:

    streamlit run streamlit_app.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "results" / "data"
TABLES = ROOT / "results" / "tables"

# --- Quantis dark design system (clean fintech look, kept self-contained) ---
BG = "#0A0A0B"
PANEL = "#141416"
PANEL_HI = "#1C1C1F"
BORDER = "#26262B"
MUTED = "#8A8A90"
TEXT = "#F5F5F6"
GRID = "#202026"
GREEN = "#00C805"
RED = "#FF4D4D"
BLUE = "#58A6FF"
GOLD = "#D29922"
PAL = [
    "#58A6FF", "#FF7B72", "#3FB950", "#D29922", "#BC8CFF",
    "#39C5CF", "#FFA198", "#8B949E", "#7EE787", "#F2CC60",
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
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont,
        'Segoe UI', Roboto, sans-serif; }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }
    .stApp { background-color: #0A0A0B; }
    header[data-testid="stHeader"] { background: transparent; }
    .qhdr { display: flex; align-items: center; gap: 14px; margin-bottom: 6px; }
    .qlogo { width: 44px; height: 44px; border-radius: 12px; flex: 0 0 auto;
        background: linear-gradient(135deg, #00C805, #0093E0); display: flex;
        align-items: center; justify-content: center; font-weight: 800; font-size: 24px;
        color: #FFFFFF; box-shadow: 0 2px 14px rgba(0, 200, 5, 0.35); }
    .qtitle { font-size: 30px; font-weight: 800; color: #F5F5F6;
        letter-spacing: -0.5px; line-height: 1.05; }
    .qsub { color: #8A8A90; font-size: 14px; margin-top: 2px; }
    .qtile { background: #141416; border: 1px solid #26262B; border-radius: 14px;
        padding: 12px 14px; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.35); }
    .qtil-label { font-size: 11px; font-weight: 600; letter-spacing: 0.07em;
        text-transform: uppercase; color: #8A8A90; }
    .qtil-val { font-size: 24px; font-weight: 700; color: #F5F5F6;
        line-height: 1.3; margin-top: 2px; }
    .qtil-val.pos { color: #00C805; }
    .qtil-val.neg { color: #FF4D4D; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { background: #141416; border: 1px solid #26262B;
        border-radius: 10px; padding: 4px 16px; }
    .stTabs [aria-selected="true"] { background: #1C1C1F; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="qhdr"><div class="qlogo">Q</div>'
    '<div><div class="qtitle">Quantis</div>'
    '<div class="qsub">Systematic multi-asset funds &bull; out-of-sample performance</div>'
    "</div></div>",
    unsafe_allow_html=True,
)
st.caption(
    "Every number is computed by scripts/run_part_b.py from the data described in "
    "the report - out-of-sample, not in-sample fit."
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


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _fund_colors(funds: list[str]) -> dict:
    return {fund: PAL[i % len(PAL)] for i, fund in enumerate(funds)}


def _pct(x: float) -> str:
    return f"{x:+.1%}" if np.isfinite(x) else "n/a"


def _sign_tone(value: float) -> str:
    return "pos" if value > 0 else ("neg" if value < 0 else "")


def _theme(
    fig: go.Figure,
    *,
    height: int = 430,
    title: str | None = None,
    ytitle: str | None = None,
    slider: bool = False,
    hovermode: str = "x unified",
    showlegend: bool = True,
    legend_y: float = 1.12,
    legend_x: float = 1,
) -> go.Figure:
    """Apply the dark Quantis chart style to a freshly-built figure."""
    legend_below = legend_y < 0
    top_margin = 52 if showlegend and not legend_below else 36
    bottom_margin = (
        88 if legend_below and slider
        else (66 if slider else (52 if legend_below else 34))
    )
    fig.update_layout(
        template=None,
        height=height,
        margin={"l": 12, "r": 12, "t": top_margin, "b": bottom_margin},
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font={"family": "Inter, -apple-system, 'Segoe UI', Roboto, sans-serif",
              "size": 13, "color": TEXT},
        hovermode=hovermode,
        hoverlabel={"bgcolor": PANEL_HI, "bordercolor": BORDER, "font": {"color": TEXT}},
        showlegend=showlegend,
        legend={
            "orientation": "h", "y": legend_y, "x": legend_x,
            "xanchor": "left" if legend_below else "right",
            "yanchor": "top" if legend_below else "bottom",
            "title": None, "font": {"color": MUTED, "size": 12},
        },
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 16, "color": TEXT}}
        if title else None,
    )
    fig.update_xaxes(
        showgrid=False, showline=False, zeroline=False, tickfont={"color": MUTED},
        rangeslider={
            "visible": True, "thickness": 0.07,
            "bordercolor": BORDER, "bgcolor": PANEL,
        } if slider else None,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID, zeroline=False, tickfont={"color": MUTED},
        title=ytitle, title_font={"size": 12, "color": MUTED},
    )
    return fig


def _plot(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _tile(label: str, value: str, tone: str = "") -> None:
    st.markdown(
        f'<div class="qtile">'
        f'<div class="qtil-label">{label}</div>'
        f'<div class="qtil-val {tone}">{value}</div></div>',
        unsafe_allow_html=True,
    )


def _fund_growth_trace(
    fund: str, color: str, *, width: float = 2.2, alpha: float = 0.06
) -> go.Scatter:
    g = growth_of_one(fund_return_series(fund))
    return go.Scatter(
        x=g.index, y=g, mode="lines", name=fund,
        line={"color": color, "width": width},
        fill="tozeroy", fillcolor=_rgba(color, alpha),
        hovertemplate=(
            "%{x|%d %b %Y}<br><b>%{fullData.name}</b><br>"
            "Value of $1: %{y:.2f}<extra></extra>"
        ),
    )


# ---------------------------------------------------------------------------
# Compare tab
# ---------------------------------------------------------------------------
def compare_tab() -> None:
    st.subheader("Compare the funds")
    metrics = load_metrics()
    metrics["family_label"] = metrics["family"].map(FAMILY_LABEL)
    metrics["method_label"] = metrics["method"].map(_method_label)

    funds = list(metrics["fund"])
    defaults = [f for f in funds if f.startswith("Equity")]
    selected = st.multiselect(
        "Funds to compare", funds, default=defaults, key="compare_funds"
    )

    if not selected:
        st.info("Pick at least one fund.")
        return

    view = metrics[metrics["fund"].isin(selected)].copy()
    first = view.iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _tile("Annualised return", _pct(first["annualized_return"]),
              _sign_tone(first["annualized_return"]))
    with c2:
        _tile("Annualised vol", f"{first['annualized_volatility']:.1%}")
    with c3:
        _tile("Sharpe", f"{first['sharpe_ratio']:.2f}", _sign_tone(first["sharpe_ratio"]))
    with c4:
        _tile("Max drawdown", f"{first['max_drawdown']:.1%}", "neg")
    with c5:
        _tile("Growth of $1", f"${first['growth_of_1']:.2f}",
              "pos" if first["growth_of_1"] >= 1 else "neg")
    st.caption(
        f"Headline numbers shown for **{first['fund']}** - all selected funds "
        "are in the table below."
    )

    view["return"] = view["annualized_return"].map(lambda v: f"{v:.2%}")
    view["volatility"] = view["annualized_volatility"].map(lambda v: f"{v:.2%}")
    view["Sharpe"] = view["sharpe_ratio"].map(lambda v: f"{v:.2f}")
    view["max drawdown"] = view["max_drawdown"].map(lambda v: f"{v:.1%}")
    view["growth of $1"] = view["growth_of_1"].map(lambda v: f"${v:.2f}")
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
        use_container_width=True,
    )

    colors = _fund_colors(selected)
    fig = go.Figure([_fund_growth_trace(f, colors[f]) for f in selected])
    _theme(fig, height=460, title="Growth of $1 - out of sample",
           ytitle="Value of $1", slider=True, legend_y=-0.2, legend_x=0)
    _plot(fig)

    fig2 = go.Figure()
    for f in selected:
        m = metrics[metrics["fund"] == f].iloc[0]
        fig2.add_trace(go.Scatter(
            x=[m["annualized_volatility"]], y=[m["annualized_return"]],
            mode="markers", name=f,
            marker={"size": 17, "color": colors[f],
                    "line": {"color": "white", "width": 2}},
            hovertemplate="<b>%{fullData.name}</b><br>Volatility: %{x:.1%}"
                          "<br>Return: %{y:.1%}<extra></extra>",
        ))
    _theme(fig2, height=420, title="Return vs risk - out of sample",
           ytitle="Annualised return", hovermode="closest",
           legend_y=-0.2, legend_x=0)
    fig2.update_xaxes(tickformat=".0%", title="Annualised volatility",
                      title_font={"size": 12, "color": MUTED})
    _plot(fig2)


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
    with c1:
        _tile("Annualised return", _pct(m["annualized_return"]),
              _sign_tone(m["annualized_return"]))
    with c2:
        _tile("Annualised vol", f"{m['annualized_volatility']:.1%}")
    with c3:
        _tile("Sharpe ratio", f"{m['sharpe_ratio']:.2f}", _sign_tone(m["sharpe_ratio"]))
    with c4:
        _tile("Max drawdown", f"{m['max_drawdown']:.1%}", "neg")
    with c5:
        _tile("Growth of $1", f"${m['growth_of_1']:.2f}",
              "pos" if m["growth_of_1"] >= 1 else "neg")

    col_l, col_r = st.columns(2)
    with col_l:
        fig = go.Figure(go.Scatter(
            x=growth.index, y=growth, mode="lines",
            line={"color": GREEN, "width": 2.4},
            fill="tozeroy", fillcolor=_rgba(GREEN, 0.08),
            hovertemplate="%{x|%d %b %Y}<br>Value of $1: %{y:.2f}<extra></extra>",
        ))
        _theme(fig, height=400, title="Growth of $1", ytitle="Value of $1",
               slider=True, showlegend=False)
        _plot(fig)
    with col_r:
        fig = go.Figure(go.Scatter(
            x=dd.index, y=dd, mode="lines",
            line={"color": RED, "width": 2.0},
            fill="tozeroy", fillcolor=_rgba(RED, 0.12),
            hovertemplate="%{x|%d %b %Y}<br>Drawdown: %{y:.1%}<extra></extra>",
        ))
        _theme(fig, height=400, title="Drawdown", ytitle="Drawdown", slider=True, showlegend=False)
        fig.update_yaxes(tickformat=".0%")
        _plot(fig)

    top = load_current_holdings(fund)
    if len(top):
        st.markdown(f"**Current holdings** (top {len(top)} positions at the latest rebalance)")
        col_d, col_w = st.columns(2)
        with col_d:
            fig = go.Figure(go.Pie(
                labels=top["ticker"], values=top["weight_pct"],
                hole=0.68, sort=False, direction="clockwise",
                marker={"colors": [PAL[i % len(PAL)] for i in range(len(top))],
                        "line": {"color": "white", "width": 2}},
                textinfo="none",
                customdata=top["weight_pct"],
                hovertemplate="<b>%{label}</b><br>%{customdata:.2f}% of the fund<extra></extra>",
            ))
            fig.add_annotation(
                text=(f"<b>{len(top)} positions</b>"
                      "<br><span style='font-size:12px'>latest rebalance</span>"),
                showarrow=False, x=0.5, y=0.5, font={"color": TEXT, "size": 14},
            )
            _theme(fig, height=380, title="Portfolio split", hovermode="closest",
                   legend_y=-0.2, legend_x=0)
            _plot(fig)
        with col_w:
            weights = load_fund_weights()
            wf = weights[weights["fund"] == fund]
            pivot = (wf[wf["ticker"].isin(top["ticker"])]
                     .pivot(index="rebalance_date", columns="ticker", values="weight")
                     .fillna(0.0).sort_index())
            if len(pivot):
                fig = go.Figure()
                for i, t in enumerate(pivot.columns):
                    fig.add_trace(go.Scatter(
                        x=pivot.index, y=pivot[t], mode="lines", stackgroup="one",
                        name=t, line={"width": 0.4, "color": PAL[i % len(PAL)]},
                        fillcolor=PAL[i % len(PAL)],
                        hovertemplate="%{x|%d %b %Y}<br>%{fullData.name}: %{y:.1%}<extra></extra>",
                    ))
                _theme(fig, height=380, title="Allocation over time",
                       ytitle="Weight", slider=True, legend_y=-0.2, legend_x=0)
                fig.update_yaxes(tickformat=".0%", range=[0, 1])
                _plot(fig)


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
    with c1:
        _tile("Blended return", _pct(m["annualized_return"]),
              _sign_tone(m["annualized_return"]))
    with c2:
        _tile("Blended vol", f"{m['annualized_volatility']:.1%}")
    with c3:
        _tile("Blended Sharpe", f"{m['sharpe_ratio']:.2f}", _sign_tone(m["sharpe_ratio"]))
    with c4:
        _tile("Blended max drawdown", f"{m['max_drawdown']:.1%}", "neg")
    with c5:
        _tile("Blended growth of $1", f"${m['growth_of_1']:.2f}",
              "pos" if m["growth_of_1"] >= 1 else "neg")

    col_p, col_g = st.columns(2)
    with col_p:
        fig = go.Figure(go.Pie(
            labels=list(alloc.keys()), values=[w * 100 for w in alloc.values()],
            hole=0.68, sort=False, direction="clockwise",
            marker={"colors": [_fund_colors(chosen)[f] for f in chosen],
                    "line": {"color": "white", "width": 2}},
            textinfo="percent", textposition="inside", textfont={"color": "white", "size": 13},
            customdata=[alloc[f] for f in chosen],
            hovertemplate="<b>%{label}</b><br>%{customdata:.0%} of portfolio<extra></extra>",
        ))
        fig.add_annotation(
            text=f"<b>{total:.0f}%</b><br><span style='font-size:12px'>allocated</span>",
            showarrow=False, x=0.5, y=0.5, font={"color": TEXT, "size": 14},
        )
        _theme(fig, height=400, title="Your split", hovermode="closest", showlegend=False)
        _plot(fig)
    with col_g:
        growth = growth_of_one(blended)
        fig = go.Figure()
        colors = _fund_colors(chosen)
        for fund in chosen:
            g = growth_of_one(wide[fund].dropna())
            fig.add_trace(go.Scatter(
                x=g.index, y=g, mode="lines", name=fund,
                line={"color": colors[fund], "width": 1.3},
                opacity=0.55,
                hovertemplate="%{x|%d %b %Y}<br>%{fullData.name}: %{y:.2f}<extra></extra>",
            ))
        fig.add_trace(go.Scatter(
            x=growth.index, y=growth, mode="lines", name="Your portfolio",
            line={"color": GREEN, "width": 2.6},
            fill="tozeroy", fillcolor=_rgba(GREEN, 0.08),
            hovertemplate="%{x|%d %b %Y}<br>%{fullData.name}: %{y:.2f}<extra></extra>",
        ))
        _theme(fig, height=400, title="Your allocation vs the funds",
               ytitle="Value of $1", slider=True, legend_y=-0.2, legend_x=0)
        _plot(fig)


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

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub.index, y=sub["sentiment"], mode="lines", name="Sentiment (same day)",
        line={"color": BLUE, "width": 1.6},
        hovertemplate="%{x|%d %b %Y}<br>%{fullData.name}: %{y:.3f}<extra></extra>",
    ))
    lag = sub["sentiment_lag1"].dropna()
    fig.add_trace(go.Scatter(
        x=lag.index, y=lag, mode="lines", name="Lagged 1 day (what trades can use)",
        line={"color": GREEN, "width": 1.4},
        fill="tozeroy", fillcolor=_rgba(GREEN, 0.05),
        hovertemplate="%{x|%d %b %Y}<br>%{fullData.name}: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0.0, line_dash="dot", line_color=MUTED, line_width=1)
    _theme(fig, height=430, title=f"{sector} sector sentiment over time",
           ytitle="Sentiment (compound)", slider=True, legend_y=-0.2, legend_x=0)
    fig.update_yaxes(tickformat=".3f")
    _plot(fig)

    monthly_all = (sector_df.groupby([pd.Grouper(key="trade_date", freq="ME"), "sector"])
                   ["sentiment"].mean().reset_index())
    monthly_all["Month"] = monthly_all["trade_date"].dt.strftime("%Y-%m")
    monthly_wide = monthly_all.pivot(index="Month", columns="sector", values="sentiment")
    st.markdown("**Monthly average** (all sectors)")
    st.dataframe(monthly_wide.round(3), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        _tile("Selected sector: mean", f"{sub['sentiment'].mean():+.4f}",
              _sign_tone(sub["sentiment"].mean()))
    with c2:
        _tile("Selected sector: std", f"{sub['sentiment'].std():.4f}")
    with c3:
        _tile("Days above neutral", f"{(sub['sentiment'] > 0).mean():.0%}")
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
