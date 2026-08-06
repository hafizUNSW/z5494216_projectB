# Report outline (Part B) - Quantis

Author the report in Word (`report.docx` is the editable source; this OUTLINE.md
is only a planning aid) and submit it as `report.pdf`. Max 10 pages of written
narrative (excluding appendix and references) - required exhibits may go in an
appendix. Every exhibit must be self-contained (caption, labelled axes, units,
sample period) and referenced and interpreted in the text. The canonical list of
required exhibits is in PROJECT_BRIEF.md, Section 5.

## Build outline mapped to the brief (Section 5)

The plan below follows PROJECT_BRIEF.md Section 5 exactly; each item shows its
status.

### 1. Funds (required minimum + higher band)
- [x] Combined equity+crypto fund with >= 2 optimisation methods (we ship 4
      methods: Equal Weight, Minimum Variance, Maximum Sharpe, Risk Parity).
- [x] Higher band extras: equity-only and crypto-only universes (12 base funds
      in total: 4 methods x 3 universes), a novel method (true ERC risk
      parity), a sentiment fusion with a measured before-vs-after effect, and
      standout app features (allocation builder, weights-over-time view).
- Evidence: `results/tables/backtest_parameters.csv`,
  `results/tables/performance_metrics.csv`, `results/tables/fact_sheets.csv`,
  `results/figures/weights_over_time.png`.

### 2. Build it (reuse my own Part A foundation)
- [x] Rebuild the Part A foundation inside this folder from raw data
      (`src/data_access.py`, `src/etl.py`, `src/features.py`) - no Part A
      output files are copied.
- [x] Out-of-sample walk-forward funds and fact sheets (`src/portfolios.py`).
- [x] Sentiment model + sector index (`src/sentiment.py`, Week 8 extended
      VADER method) and the fusion (`src/fusion.py`).
- [x] Streamlit app (`streamlit_app.py`, Quantis).
- Entry point: `scripts/run_part_b.py` reproduces every artifact.

### 3. Artifacts (commit precomputed; never raw)
- [x] App-readable CSVs under `results/data/` (committed - the app reads them):
      fund_returns, fund_weights, sector_sentiment_index, fund_holdings.
- [x] Report tables under `results/tables/`: performance_metrics,
      backtest_parameters, fact_sheets, turnover, fusion_comparison,
      fusion_kappa_sensitivity, vader_extension_before_after,
      sentiment_coverage, news_volume_volatility,
      news_volume_volatility_correlation.
- [x] Figures under `results/figures/`: growth_of_1, drawdown,
      weights_over_time, sharpe_barplot, sector_sentiment_index,
      fusion_before_after, news_volume_volatility.
- [x] Raw .parquet / source data are never committed (`.gitignore` blocks
      `*.parquet`, `*.csv` outside `results/`; data loads via `data_access`).

### 4. Run order (verified end to end)
- [x] `python scripts/run_part_b.py` - full pipeline, 6 stages, reproducible.
- [x] `streamlit run streamlit_app.py` - boots; all four tabs load (checked
      with the Streamlit testing harness + headless HTTP 200).
- [x] `python scripts/check_handin.py` - 21 checks pass, no [FAIL].
- [ ] `git status` - the folder becomes its own GitHub repo (see Deploy below).

### Two mistakes to avoid (both actively guarded)
- [x] **Deployed app recomputing backtests or running VADER.** The app reads
      only precomputed `results/`; it never imports the sentiment-scoring
      package and never reruns an optimisation. `check_handin.py` warns if the
      app text references the VADER package (guard passes). nltk lives in
      requirements-dev.txt only.
- [ ] **Private repo at hand-in.** Keep the repo private while building, but
      make it PUBLIC and confirm the live app still loads before submitting the
      URL. This is the student's browser step (it needs the GitHub/Streamlit
      login).

### Remaining to hand in
- [x] First draft of `report/report.docx` authored (AI-drafted prose tagged
      `[DRAFT]` in red for the student's rewrite) and `report/report.pdf`
      exported (narrative pages 2-10, within the 10-page limit). Student must
      rewrite every red `[DRAFT]` paragraph in their own words, insert app
      screenshots in section 5, convert figure/table references to Word
      cross-references, then re-export the PDF.
- [ ] `git init` this folder, commit (including `results/`), push to a new
      GitHub repo, deploy on share.streamlit.io (entrypoint
      `streamlit_app.py`), make the repo public at hand-in. (Student's
      browser step.)
- [ ] Delete `__pycache__/` and `*.pyc` before zipping; zip the folder to
      Moodle; submit the live URL + repo link.

## Suggested structure

### 1. The funds and the backtest design (~1.5 pages)
- What Quantis offers: 12 base funds across three universes - equity (50 US
  large-caps), crypto (10 coins), and combined (60) - plus two sentiment-tilted
  variants of the equity funds.
- Methods: Equal Weight, Minimum Variance, Maximum Sharpe, Risk Parity (true
  equal risk contribution, Newton solve + recursive cap pinning).
- Backtest design: 504-trading-day (2-year) rolling estimation window,
  rebalanced every 21 periods (~monthly), weights applied from the next period
  (no look-ahead), long-only, caps 20% (equity/combined) / 35% (crypto),
  covariance shrunk 10% toward the diagonal, risk-free 0.
  Exhibit: `backtest_parameters.csv`; figure `weights_over_time.png`.

### 2. Out-of-sample results and fund fact sheets (~2 pages)
- Out-of-sample spans: equity/combined 2021-12-31 to 2023-12-29 (502 days);
  crypto 2021-05-19 to 2023-12-31 (957 days).
- Tables: `performance_metrics.csv` (annualised return, vol, Sharpe, max
  drawdown, growth of $1), `turnover.csv` (gross vs net at 10 bps one-sided).
- Figures: `growth_of_1.png`, `drawdown.png`, `sharpe_barplot.png`.
- Fact-sheet template (growth of $1, drawdown, Sharpe, current top-10 holdings)
  illustrated on the combined funds; full fact sheets in the appendix.

### 3. The sentiment index (~1.5 pages)
- Data: 146,836 headlines after dedup. Method: VADER extended by a finance
  lexicon (Week 8 build-and-test approach), kept isolated from the plain
  analyser; rejected-term controls.
- Sector index: equal-weighted ticker-day sentiment within each sector; the
  no-headline policy (treated as neutral 0 - absence of news is no signal,
  keeps the panel dense) justified.
- Look-ahead: signal lagged one trading day (Saturday/Monday headline aligned
  to Monday is first usable Tuesday).
  Exhibits: `sector_sentiment_index.png`, `ticker_daily_sentiment.csv`,
  `vader_extension_before_after.csv`, `sentiment_coverage.csv` (66.4% of
  ticker-days have non-zero compound).

### 4. Extensions and innovations: does attention add value? (~1.5 pages)
- Custom finance lexicon (Week 8 build-and-test method, isolated from the
  plain VADER analyser; rejected-term controls) - exhibit
  `vader_extension_before_after.csv`.
- Fusion tilt: `weight_i *= (1 + kappa * signal_i)`, renormalised with the same
  cap.
- Fusion results: Equity Equal Weight Sharpe 0.416 -> 0.409 (kappa 3), slight
  decline; Equity Minimum Variance 0.257 -> 0.298 (kappa 2), improvement.
  Honest read: the tilt is value-neutral to mildly positive depending on the
  fund - report the full kappa sweep, not a cherry-pick.
  Exhibits: `fusion_comparison.csv`, `fusion_kappa_sensitivity.csv`,
  `fusion_before_after.png`.
- News-volume-vs-forward-volatility cross-check: for each equity ticker-day,
  strictly-lagged news attention (trailing 5-day headline count, shift 1) vs
  strictly-forward realised volatility (next-5-day annualised return std).
  Result: positive Spearman rho in 6 of 10 sectors (Utilities +0.195,
  Industrials +0.175, Tech +0.157, Materials +0.127, RealEstate +0.110),
  near-zero Comm/Financials, and a notable NEGATIVE Energy reading (-0.204) -
  report as an honest, mixed finding: attention anticipates volatility in most
  sectors but not all, so the signal is sector-conditional.
  Exhibits: `news_volume_volatility.csv`, `news_volume_volatility_correlation.csv`,
  `news_volume_volatility.png`.

### 5. The app and the investor journey (~1.5 pages)
- Four tabs: Compare, Fact sheet, Allocate (blended portfolio), Sentiment.
- Screenshots; explain that the app reads only precomputed `results/` so it
  stays light on Streamlit Cloud. Live URL + public repo link.
- Innovation: ERC risk parity, water-filling cap projection, extended VADER
  with isolation, kappa sweep, turnover model, custom design system, and the
  news-volume-vs-forward-volatility cross-check (see section 4 note below).

### 6. Critical reflection and three recommendations (~1 page)
- What held back performance (e.g., buy-only max-Sharpe underperforms;
  sentiment signal is same-day-aligned and diluted by equal weighting).
- Three concrete recommendations (e.g., short-leg or relaxed caps, per-ticker
  signal with volatility scaling, rebalancing less often to cut turnover).
