# Prompt log 06 - Station 3 audit against the brief (funds, backtests, sentiment)

## What I wanted
Audit Station 3 against PROJECT_BRIEF.md Section 5 line by line: distinct
(asset family, method) funds, walk-forward OOS backtest with stated parameters
(first live date, window, risk-free, transaction cost), per-fund fact sheets,
the required exhibits, the sentiment model + lagged sector index, the fusion
before/after, exact artifact filenames, and a sanity check that the optimisers
actually produce different weights.

## Prompt(s)
"Verify every Station 3 requirement is met in code and artifacts: funds
(combined + equity + crypto, 4 methods each), no-look-ahead walk-forward
backtest with explicit parameters, fact sheets, the full exhibit set, lagged
sector sentiment index, fusion before/after, exact filenames, and a numeric
sanity check that weights differ meaningfully across methods."

## What the assistant produced
The audit report (below), and three corrections it found along the way.

## What was wrong or risky
1. `results/tables/backtest_parameters.csv` stored each family's parameters as
   one row with a Python dict in the `value` column (e.g. `{'family': ...,
   'window': ...}`) - a marker reading the CSV would see an unreadable blob
   instead of the explicit first-live-date / window / risk-free statement the
   brief demands. Fixed: `run_part_b.py` now flattens each family into tidy
   rows (`equity_first_live_date`, `equity_window`, ...).
2. The report outline said ticker-days with no headlines are "carried forward".
   The implemented policy is neutral-0 fill (`_on_calendar` reindexes with
   fill_value=0.0) - a deliberate, documented choice. Fixed the outline to say
   neutral, matching the code and the module docstring.
3. The weight-diversity sanity check only existed as a one-off command. Added
   a reproducible diagnostic to `run_part_b.py` that prints the top-3 weight
   share per method at each fund's latest rebalance, so the run log itself
   shows the optimisers did not stall.

## What I changed and why
All three fixes above, then a full re-run of `python scripts/run_part_b.py`.
The audit result follows.

## Audit result (after fixes)

### Funds and backtests - PASS
- 12 base funds = 4 methods (Equal Weight, Minimum Variance, Maximum Sharpe,
  Risk Parity) x 3 universes (equity 50, crypto 10, combined 60); plus 2
  sentiment-tilted equity funds. Each (family, method) pair is one fund.
- Walk-forward OOS: 504-day window, rebalanced every 21 periods (~monthly),
  weights applied from the next period, long-only, caps 20%/35%. Stated in
  `backtest_parameters.csv`: first live equity/combined 2021-12-31 (502 OOS
  days), crypto 2021-05-19 (957 OOS days), risk-free 0, transaction cost 0 in
  the base backtest with a 10 bps one-sided cost model as the innovation
  (`turnover.csv`).
- No look-ahead: pinned by `test_walk_forward_no_look_ahead_first_weight` and
  the lagged-signal test.
- Fact sheets: `fact_sheets.csv` (growth of $1, annualised return/vol, Sharpe,
  max drawdown) + `fund_holdings.csv` (top-10 target weights at the most
  recent rebalance) for all 14 funds.
- Compare funds: `performance_metrics.csv`, `growth_of_1.png`,
  `sharpe_barplot.png`.

### Weight diversity sanity check - PASS
Top-3 weight share at the latest rebalance: Equal Weight 6%/30%/5% (= 3/n),
Minimum Variance 39%/83%/36%, Maximum Sharpe 55%/85%/53%, Risk Parity
11%/39%/11%. Mean |dW| across method pairs up to ~0.027. The optimisers are
clearly not stalling on tiny covariances.

### Sentiment - PASS
- Headlines scored with extended VADER (146,836 headlines, 37,977 ticker-days,
  66.4% non-zero compound).
- Sector index equal-weights tickers within each sector; no-headline days
  treated as neutral (0), documented.
- Signal lagged 1 trading day (`sentiment_lag1`, `ticker_signal` shift(1)),
  tradable only from the next day.

### Fusion - PASS
- Tilt `weight_i *= (1 + kappa * signal_i)` on the equity funds, same cap,
  renormalised by `normalize_capped`. Before/after in `fusion_comparison.csv`
  and `fusion_before_after.png`; kappa sweep in
  `fusion_kappa_sensitivity.csv`. Honest read: Equal Weight 0.416 -> 0.409
  (slight decline), Minimum Variance 0.257 -> 0.298 (improvement).

### Required artifacts and exhibits - PASS
- Exact filenames present: `results/data/fund_returns.csv`,
  `results/data/fund_weights.csv`, `results/data/sector_sentiment_index.csv`,
  `results/tables/performance_metrics.csv`.
- All seven exhibit types produced (performance table, growth of $1,
  drawdown, weights over time, Sharpe barplot, sector sentiment index, fusion
  table + figure); everything under `results/data/`, `results/tables/`,
  `results/figures/`.

### Verification commands
`python scripts/run_part_b.py` - clean end-to-end; `python -m pytest -q` - 31
passed; `python -m ruff check scripts src tests streamlit_app.py` - clean;
`python scripts/check_handin.py` - 21 checks pass.
