"""Reproduce your Part B results. Run from the project root:

    python scripts/run_part_b.py

Pipeline (DFF Stations 3-4):
  1. Load raw data through src/data_access.py and rebuild the Part A
     foundation (ETL integrity checks, returns, combined panel, headline
     panel) inside this folder - nothing is copied from Part A's outputs.
  2. Funds: walk-forward out-of-sample backtests for equity-only, crypto-only,
     and combined families across four methods (equal weight, minimum
     variance, maximum Sharpe, risk parity).
  3. Sentiment: score headlines with VADER extended by a finance lexicon,
     build the sector index, and produce a look-ahead-safe per-ticker signal.
  4. Fusion: tilt the equity funds with sentiment and measure before-vs-after.
  5. App artifacts + report exhibits are written to results/.

Required output filenames (the app reads these, markers check these):
  results/data/fund_returns.csv
  results/data/fund_weights.csv
  results/data/sector_sentiment_index.csv
  results/tables/performance_metrics.csv
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from src import sentiment as sentiment_mod
from src.etl import IntegrityReport, load_clean_crypto, load_clean_equities, load_clean_headlines
from src.features import assemble_headline_panel, build_combined_returns_panel, compute_returns
from src.fusion import apply_sentiment_tilt, apply_weights
from src.portfolios import (
    MAX_WEIGHT,
    METHOD_LABELS,
    METHODS,
    drawdown,
    fact_sheet,
    growth_of_one,
    performance_metrics,
    walk_forward_backtest,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Backtest configuration (state every choice in the report)
# ---------------------------------------------------------------------------
WINDOW = 504          # trailing estimation window (2 years of trading periods)
REBALANCE = 21        # rebalance every 21 periods (~ monthly)
KAPPA = 1.0           # sentiment tilt strength
COST_BPS = 10         # one-sided transaction cost model (0.10%)

# ---------------------------------------------------------------------------
# The Economist-style design system (carried over from Part A)
# ---------------------------------------------------------------------------
ECONOMIST_RED = "#E3120B"
ECONOMIST_BLUE = "#0D5691"
ECONOMIST_GREY = "#666666"
ECONOMIST_PAL = [
    "#0D5691", "#E3120B", "#4E8C2C", "#D98B35",
    "#6C3483", "#1AABB8", "#BC4B52", "#4A5568",
    "#5B7553", "#8B6914",
]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": ECONOMIST_GREY,
    "axes.labelcolor": ECONOMIST_GREY,
    "xtick.color": ECONOMIST_GREY,
    "ytick.color": ECONOMIST_GREY,
    "text.color": ECONOMIST_GREY,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.5,
})

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
DATA = RESULTS / "data"
for d in [TABLES, FIGURES, DATA]:
    d.mkdir(parents=True, exist_ok=True)

FAMILY_META = {
    "equity":   {"label": "Equity",   "periods": 252, "window": WINDOW},
    "crypto":   {"label": "Crypto",   "periods": 365, "window": WINDOW},
    "combined": {"label": "Combined", "periods": 252, "window": WINDOW},
}


def fund_label(family: str, method: str) -> str:
    return f"{FAMILY_META[family]['label']} {METHOD_LABELS[method]}"


def clean_wide(returns_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot returns to (date x ticker), forward-fill small gaps, drop NaN rows."""
    wide = (
        returns_long[["date", "ticker", "return"]]
        .pivot_table(index="date", columns="ticker", values="return")
        .sort_index()
    )
    wide = wide.ffill()
    wide = wide.dropna(how="any")
    return wide


# ===================================================================
# Station 1+2 (from Part A) - rebuild the clean foundation
# ===================================================================

print("[1/6] Loading and cleaning data through src/data_access.py...")
report = IntegrityReport()
eq_prices, report = load_clean_equities()
cr_prices = load_clean_crypto(report)
headlines = load_clean_headlines(report)

print("[2/6] Computing returns and assembling the panels...")
eq_returns = compute_returns(eq_prices)
cr_returns = compute_returns(cr_prices)
combined_panel = build_combined_returns_panel(eq_returns, cr_returns)
equity_dates = pd.DatetimeIndex(sorted(eq_prices["date"].unique()))
headline_panel = assemble_headline_panel(headlines, equity_dates=equity_dates)

print(report.summary())
print()

equity_wide = clean_wide(eq_returns)
crypto_wide = clean_wide(cr_returns)
combined_wide = clean_wide(combined_panel)

print(f"  equity wide: {equity_wide.shape}  "
      f"({equity_wide.index.min().date()} to {equity_wide.index.max().date()})")
print(f"  crypto wide: {crypto_wide.shape}  "
      f"({crypto_wide.index.min().date()} to {crypto_wide.index.max().date()})")
print(f"  combined wide: {combined_wide.shape}")
print(f"  headline panel: {headline_panel.shape}")


# ===================================================================
# Station 3 - funds: walk-forward out-of-sample backtests
# ===================================================================

print("[3/6] Building funds with walk-forward out-of-sample backtests...")

family_wide = {"equity": equity_wide, "crypto": crypto_wide, "combined": combined_wide}

fund_returns_rows: list[dict] = []
fund_weights_rows: list[dict] = []
metrics_rows: list[dict] = []
fact_rows: list[dict] = []
holdings_rows: list[dict] = []
backtest_params: dict[str, dict] = {}

for family in FAMILY_META:
    wide = family_wide[family]
    params = FAMILY_META[family]
    first_live = wide.index[params["window"] - 1]
    backtest_params[family] = {
        "family": family,
        "n_assets": wide.shape[1],
        "window": params["window"],
        "rebalance": REBALANCE,
        "periods_per_year": params["periods"],
        "first_live_date": str(first_live.date()),
        "oos_days": len(wide) - params["window"] + 1,
        "cap": MAX_WEIGHT[family],
    }
    for method in METHODS:
        returns, weights = walk_forward_backtest(
            wide, method, window=params["window"], rebalance=REBALANCE, family=family
        )
        label = fund_label(family, method)
        for dt, r in returns.items():
            fund_returns_rows.append({"date": dt, "fund": label, "return": round(float(r), 8)})
        for _, row in weights.iterrows():
            fund_weights_rows.append({
                "rebalance_date": row["rebalance_date"], "fund": label,
                "ticker": row["ticker"], "weight": row["weight"],
            })
        metrics = performance_metrics(returns, periods_per_year=params["periods"])
        metrics_rows.append({
            "fund": label, "family": family, "method": method,
            "annualized_return": round(metrics["annualized_return"], 6),
            "annualized_volatility": round(metrics["annualized_volatility"], 6),
            "sharpe_ratio": round(metrics["sharpe_ratio"], 4),
            "max_drawdown": round(metrics["max_drawdown"], 6),
            "growth_of_1": round(metrics["growth_of_1"], 4),
            "oos_period": f"{first_live.date()!s} to {returns.index.max().date()!s}",
        })
        sheet = fact_sheet(label, family, returns, weights, periods_per_year=params["periods"])
        fact_rows.append({k: v for k, v in sheet.items() if k != "holdings"})
        for h in sheet["holdings"]:
            holdings_rows.append({"fund": label, **h})

    print(f"  {family}: {', '.join(METHOD_LABELS[m] for m in METHODS)} done; "
          f"first live date {backtest_params[family]['first_live_date']}, "
          f"OOS {backtest_params[family]['oos_days']} days")

fund_returns = pd.DataFrame(fund_returns_rows)
fund_weights = pd.DataFrame(fund_weights_rows)
performance_metrics_df = pd.DataFrame(metrics_rows)
fact_sheets = pd.DataFrame(fact_rows)
fund_holdings = pd.DataFrame(holdings_rows)


# ===================================================================
# Turnover & transaction-cost model (Innovation)
# ===================================================================

print("[4/6] Turnover and transaction-cost model...")

turnover_rows = []
for label, g in fund_weights.groupby("fund"):
    pivot = g.pivot(index="rebalance_date", columns="ticker", values="weight").sort_index()
    if len(pivot) < 2:
        continue
    per_rebalance = pivot.diff().abs().sum(axis=1).dropna()
    avg_turnover = float(per_rebalance.mean())
    annualised_turnover = avg_turnover * (252 / REBALANCE)
    cost = annualised_turnover * (COST_BPS / 100.0)
    # Use the same compounded annualised return as the metrics table so the
    # net figure is consistent with results/tables/performance_metrics.csv.
    gross_ann = float(performance_metrics_df.loc[
        performance_metrics_df["fund"] == label, "annualized_return"].iloc[0])
    turnover_rows.append({
        "fund": label,
        "n_rebalances": len(pivot),
        "avg_turnover_per_rebalance": round(avg_turnover, 4),
        "annualised_turnover": round(annualised_turnover, 4),
        "estimated_cost_annualised": round(cost, 6),
        "gross_annualised_return": round(gross_ann, 6),
        "net_annualised_return": round(gross_ann - cost, 6),
    })
turnover_df = pd.DataFrame(turnover_rows)
turnover_df.to_csv(TABLES / "turnover.csv", index=False)


# ===================================================================
# Station 3 - sentiment model, sector index, fusion signal
# ===================================================================

print("[5/6] Scoring headlines with extended VADER and building the sector index...")

# One-time VADER lexicon download is a build step, never a deployed-app step.
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

# Innovation evidence: before/after on finance sentences (Week 8 method).
# The last two sentences use terms the finance lexicon deliberately rejects;
# their scores must not move, which is the check that the approval step works.
demo_sentences = [
    "The company reported an earnings beat.",
    "The stock surged on strong guidance.",
    "Management issued a guidance cut.",
    "Analysts issued a downgrade.",
    "The firm announced a buyback.",
    "Revenue was sharply lower than expected.",
    "The filing disclosed a material weakness.",
    "The auditor raised a going concern warning.",
    "The firm reduced its debt.",
    "The stock is volatile.",
]
demo_rows = []
for s in demo_sentences:
    before = sentiment_mod.SentimentIntensityAnalyzer().polarity_scores(s)["compound"]
    after = sentiment_mod.build_analyzer().polarity_scores(s)["compound"]
    demo_rows.append({
        "sentence": s,
        "vader_before": round(before, 4),
        "extended_after": round(after, 4),
        "change": round(after - before, 4),
        "woke_up": int(before == 0.0 and after != 0.0),
    })
pd.DataFrame(demo_rows).to_csv(TABLES / "vader_extension_before_after.csv", index=False)

scores = sentiment_mod.score_headlines(headline_panel)
ticker_daily = sentiment_mod.ticker_daily_sentiment(scores)
ticker_daily.to_csv(DATA / "ticker_daily_sentiment.csv", index=False)

sector_index = sentiment_mod.sector_sentiment_index(ticker_daily, equity_dates)
sector_index.to_csv(DATA / "sector_sentiment_index.csv", index=False)

ticker_signal = sentiment_mod.ticker_signal_wide(ticker_daily, equity_dates, lag=1, smooth=21)

lexicon_coverage = sentiment_mod.lexicon_coverage(scores)
lexicon_coverage.to_csv(TABLES / "sentiment_coverage.csv", index=False)

print(f"  headlines scored: {len(scores):,}; ticker-days: {len(ticker_daily):,}")
share_nonzero = lexicon_coverage.loc[
    lexicon_coverage["metric"] == "share non-zero", "value"
].iloc[0]
print(f"  sector index rows: {len(sector_index):,}; "
      f"non-zero compound share: {share_nonzero:.1%}")

# Innovation: news attention vs forward return volatility cross-check. The
# attention measure is strictly lagged and the volatility strictly forward, so
# the pair asks "does what you already know anticipate how much the stock will
# move" - the risk information the funds are built to control.
volume_vol_buckets, volume_vol_rho = sentiment_mod.news_volume_volatility_crosscheck(
    ticker_daily, equity_wide, horizon=5
)
volume_vol_buckets.to_csv(TABLES / "news_volume_volatility.csv", index=False)
volume_vol_rho.to_csv(TABLES / "news_volume_volatility_correlation.csv", index=False)
if len(volume_vol_rho):
    print("  news-volume vs forward-vol cross-check (Spearman rho by sector):")
    for _, row in volume_vol_rho.iterrows():
        print(f"    {row['sector']}: {row['spearman_rho']:+.3f} "
              f"({row['n_days']:,} ticker-days)")


# ===================================================================
# Station 3 - fusion: sentiment tilt on the equity funds
# ===================================================================

print("[6/6] Fusing sentiment into the equity funds (before vs after)...")

fused_funds = []
fusion_compare_rows = []
fusion_kappa_rows = []
KAPPAS = (0.5, 1.0, 2.0, 3.0)
for method in ("equal_weight", "minimum_variance"):
    base_label = fund_label("equity", method)
    base_r, base_w = walk_forward_backtest(
        equity_wide, method,
        window=FAMILY_META["equity"]["window"], rebalance=REBALANCE, family="equity",
    )
    for kappa in KAPPAS:
        fused_w = apply_sentiment_tilt(base_w, ticker_signal, kappa=kappa, cap=0.20)
        fused_r = apply_weights(equity_wide, fused_w)
        base_metrics = performance_metrics(base_r, periods_per_year=252)
        fused_metrics = performance_metrics(fused_r, periods_per_year=252)
        fused_label = f"{base_label} + Sentiment"
        if kappa == KAPPA:
            fused_funds.append((base_label, fused_label, base_r, fused_r, fused_w))
            for dt, r in fused_r.items():
                fund_returns_rows.append(
                    {"date": dt, "fund": fused_label, "return": round(float(r), 8)}
                )
            for _, row in fused_w.iterrows():
                fund_weights_rows.append({
                    "rebalance_date": row["rebalance_date"], "fund": fused_label,
                    "ticker": row["ticker"], "weight": row["weight"],
                })
            fusion_compare_rows.append({
                "fund": fused_label,
                "base_fund": base_label,
                "tilt_strength_kappa": KAPPA,
                "base_annualized_return": round(base_metrics["annualized_return"], 6),
                "fused_annualized_return": round(fused_metrics["annualized_return"], 6),
                "base_annualized_volatility": round(base_metrics["annualized_volatility"], 6),
                "fused_annualized_volatility": round(fused_metrics["annualized_volatility"], 6),
                "base_sharpe_ratio": round(base_metrics["sharpe_ratio"], 4),
                "fused_sharpe_ratio": round(fused_metrics["sharpe_ratio"], 4),
                "base_max_drawdown": round(base_metrics["max_drawdown"], 6),
                "fused_max_drawdown": round(fused_metrics["max_drawdown"], 6),
                "base_growth_of_1": round(base_metrics["growth_of_1"], 4),
                "fused_growth_of_1": round(fused_metrics["growth_of_1"], 4),
                "sharpe_change": round(
                    fused_metrics["sharpe_ratio"] - base_metrics["sharpe_ratio"], 4
                ),
            })
            # Give the fused funds a full fact-sheet entry too, so the app can
            # offer them side-by-side with the base funds.
            metrics_rows.append({
                "fund": fused_label, "family": "equity", "method": f"{method} + sentiment",
                "annualized_return": round(fused_metrics["annualized_return"], 6),
                "annualized_volatility": round(fused_metrics["annualized_volatility"], 6),
                "sharpe_ratio": round(fused_metrics["sharpe_ratio"], 4),
                "max_drawdown": round(fused_metrics["max_drawdown"], 6),
                "growth_of_1": round(fused_metrics["growth_of_1"], 4),
                "oos_period": (
                    f"{fused_r.index.min().date()!s} to {fused_r.index.max().date()!s}"
                ),
            })
            sheet = fact_sheet(fused_label, "equity", fused_r, fused_w, periods_per_year=252)
            fact_rows.append({k: v for k, v in sheet.items() if k != "holdings"})
            for h in sheet["holdings"]:
                holdings_rows.append({"fund": fused_label, **h})
        fusion_kappa_rows.append({
            "base_fund": base_label,
            "kappa": kappa,
            "sharpe_ratio": round(fused_metrics["sharpe_ratio"], 4),
            "annualized_return": round(fused_metrics["annualized_return"], 6),
            "annualized_volatility": round(fused_metrics["annualized_volatility"], 6),
            "max_drawdown": round(fused_metrics["max_drawdown"], 6),
        })
    kappa_scores = {k: round(m["sharpe_ratio"], 3) for k, m in
                    ((r["kappa"], r) for r in fusion_kappa_rows
                     if r["base_fund"] == base_label)}
    print(f"  {base_label}: base Sharpe {base_metrics['sharpe_ratio']:.3f}; "
          f"fused at kappa {KAPPAS}: {kappa_scores}")

fund_returns = pd.DataFrame(fund_returns_rows)
fund_weights = pd.DataFrame(fund_weights_rows)
performance_metrics_df = pd.DataFrame(metrics_rows)
fact_sheets = pd.DataFrame(fact_rows)
fund_holdings = pd.DataFrame(holdings_rows)
fusion_compare = pd.DataFrame(fusion_compare_rows)
fusion_compare.to_csv(TABLES / "fusion_comparison.csv", index=False)
fusion_kappa = pd.DataFrame(fusion_kappa_rows)
fusion_kappa.to_csv(TABLES / "fusion_kappa_sensitivity.csv", index=False)


# ===================================================================
# Save required app artifacts (exact filenames for the app and markers)
# ===================================================================

fund_returns.to_csv(DATA / "fund_returns.csv", index=False)
fund_weights.to_csv(DATA / "fund_weights.csv", index=False)
performance_metrics_df.to_csv(TABLES / "performance_metrics.csv", index=False)
fact_sheets.to_csv(TABLES / "fact_sheets.csv", index=False)
fund_holdings.to_csv(DATA / "fund_holdings.csv", index=False)

# Sanity check: optimisers can silently stall on tiny daily covariances and
# return near-equal-weight vectors. Print per-family concentration so the log
# shows the methods genuinely differ (EW ~= 1/n top-3 share, MV/MS concentrated,
# RP in between).
print("  weight sanity check (top-3 share at latest rebalance):")
for family in ("equity", "crypto", "combined"):
    label = FAMILY_META[family]["label"]
    shares = {}
    for method in METHODS:
        fname = fund_label(family, method)
        sub = fund_weights[fund_weights["fund"] == fname]
        last = sub["rebalance_date"].max()
        top3 = sub[sub["rebalance_date"] == last]["weight"].nlargest(3).sum()
        shares[METHOD_LABELS[method]] = top3
    print(f"    {label}: " + ", ".join(f"{k} {v:.1%}" for k, v in shares.items()))
param_rows = [
    {"parameter": "estimation_window", "value": WINDOW},
    {"parameter": "rebalance_frequency", "value": f"every {REBALANCE} periods (~monthly)"},
    {"parameter": "risk_free_rate", "value": 0.0},
    {"parameter": "transaction_costs", "value": f"{COST_BPS} bps one-sided in turnover model"},
    {"parameter": "long_only", "value": True},
    {"parameter": "covariance", "value": "sample covariance shrunk 10% toward diagonal"},
]
for family, p in backtest_params.items():
    for key, val in p.items():
        param_rows.append({"parameter": f"{family}_{key}", "value": val})
pd.DataFrame(param_rows).to_csv(TABLES / "backtest_parameters.csv", index=False)

print(f"  fund_returns.csv: {len(fund_returns):,} rows, {fund_returns['fund'].nunique()} funds")
print(f"  fund_weights.csv: {len(fund_weights):,} rows")


# ===================================================================
# Figures
# ===================================================================

print("  Building figures...")


def _month_end(s: pd.Series) -> pd.Series:
    return s.resample("ME").last()


def fig_growth():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for i, family in enumerate(("equity", "crypto", "combined")):
        ax = axes[i]
        for m, method in enumerate(METHODS):
            label = fund_label(family, method)
            sub = (fund_returns[fund_returns["fund"] == label]
                   .set_index("date")["return"].sort_index())
            g = growth_of_one(sub)
            ax.plot(g.index, g.values, linewidth=1.4,
                    color=ECONOMIST_PAL[m % len(ECONOMIST_PAL)], label=METHOD_LABELS[method])
        ax.set_title(f"{FAMILY_META[family]['label']} Funds")
        ax.set_xlabel("Date")
        ax.grid(True, axis="y", alpha=0.4)
        if i == 0:
            ax.set_ylabel("Growth of $1")
        ax.legend(fontsize=8, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("Growth of $1, Out-of-Sample (2022\u20132023)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "growth_of_1.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_drawdown():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for m, method in enumerate(METHODS):
        label = fund_label("combined", method)
        sub = fund_returns[fund_returns["fund"] == label].set_index("date")["return"].sort_index()
        dd = drawdown(sub)
        ax.plot(dd.index, dd.values, linewidth=1.3,
                color=ECONOMIST_PAL[m % len(ECONOMIST_PAL)], label=METHOD_LABELS[method])
    ax.set_title("Drawdown, Combined Funds (Out-of-Sample)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, axis="y", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(FIGURES / "drawdown.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_weights():
    label = fund_label("combined", "minimum_variance")
    g = fund_weights[fund_weights["fund"] == label]
    pivot = g.pivot(index="rebalance_date", columns="ticker", values="weight")
    top = pivot.mean().nlargest(8).index
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, t in enumerate(top):
        ax.plot(pivot.index, pivot[t].values, linewidth=1.2,
                color=ECONOMIST_PAL[i % len(ECONOMIST_PAL)], label=t)
    ax.set_title("Portfolio Weights Over Time, Combined Minimum Variance")
    ax.set_xlabel("Rebalance Date")
    ax.set_ylabel("Weight")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, axis="y", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(FIGURES / "weights_over_time.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_sharpe():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    perf = performance_metrics_df.copy()
    perf["x"] = perf["family"] + "\n" + perf["method"].replace(
        {"equal_weight": "EW", "minimum_variance": "MinVar",
         "max_sharpe": "MaxSharpe", "risk_parity": "RiskParity"})
    x = np.arange(len(perf))
    colors = {"equity": ECONOMIST_PAL[0], "crypto": ECONOMIST_PAL[1], "combined": ECONOMIST_PAL[2]}
    ax.bar(x, perf["sharpe_ratio"], color=[colors[f] for f in perf["family"]], edgecolor="white")
    ax.set_title("Sharpe Ratio by Fund and Method (Out-of-Sample)")
    ax.set_ylabel("Sharpe Ratio (rf = 0)")
    ax.set_xticks(x)
    ax.set_xticklabels(perf["x"], fontsize=7.5, rotation=0)
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES / "sharpe_barplot.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_sentiment():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    idx = sector_index.copy()
    idx["month"] = idx["trade_date"].dt.to_period("M")
    monthly = idx.groupby(["month", "sector"])["sentiment"].mean().unstack("sector")
    for i, sector in enumerate(monthly.columns):
        ax.plot(monthly.index.astype(str), monthly[sector].values, linewidth=1.1,
                color=ECONOMIST_PAL[i % len(ECONOMIST_PAL)], label=sector)
    ax.set_title("Sector Sentiment Index (monthly average, extended VADER)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sentiment (compound)")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES / "sector_sentiment_index.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_fusion():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, (base_label, fused_label, base_r, fused_r, _) in enumerate(fused_funds):
        ax = axes[i]
        gb = growth_of_one(base_r.sort_index())
        gf = growth_of_one(fused_r.sort_index())
        ax.plot(gb.index, gb.values, color=ECONOMIST_GREY, linewidth=1.3, label="Base fund")
        ax.plot(gf.index, gf.values, color=ECONOMIST_RED, linewidth=1.3, label="+ Sentiment tilt")
        ax.set_title(f"{base_label} vs {fused_label}")
        ax.set_xlabel("Date")
        ax.grid(True, axis="y", alpha=0.4)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        if i == 0:
            ax.set_ylabel("Growth of $1")
    fig.suptitle("Sentiment Fusion: Before vs After", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fusion_before_after.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_news_volume_volatility():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.15
    sectors = sorted(volume_vol_buckets["sector"].unique())
    for i, quintile in enumerate(["Q1 (least news)", "Q2", "Q3", "Q4", "Q5 (most news)"]):
        vals = [
            volume_vol_buckets.loc[
                (volume_vol_buckets["sector"] == sec)
                & (volume_vol_buckets["attention_quintile"] == quintile),
                "mean_forward_vol_annualised",
            ].iloc[0]
            for sec in sectors
        ]
        ax.bar(np.arange(len(sectors)) + (i - 2) * width, vals, width=width,
               color=ECONOMIST_PAL[i % len(ECONOMIST_PAL)], label=quintile)
    ax.set_xticks(range(len(sectors)))
    ax.set_xticklabels(sectors, fontsize=8)
    ax.set_title("News Attention vs Forward Return Volatility (next 5 days, annualised)")
    ax.set_ylabel("Forward volatility")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES / "news_volume_volatility.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


fig_growth()
fig_drawdown()
fig_weights()
fig_sharpe()
fig_sentiment()
fig_fusion()
fig_news_volume_volatility()

print("\nPart B complete. All outputs saved under results/.")
print("  required: results/data/fund_returns.csv, fund_weights.csv, sector_sentiment_index.csv")
print("            results/tables/performance_metrics.csv")
print("\nBacktest summary:")
print(pd.DataFrame([{"family": k, **v} for k, v in backtest_params.items()]).to_string(index=False))
