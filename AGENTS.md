# AGENTS.md - agent instructions for Quantis (Part B)

Read PROJECT_BRIEF.md first, then context/ (DATA_GUIDE.md, project_context.md,
verify_ai_output.md). Those override any preference in this file.

## What this project is

I am building **Quantis**, my Part B FinTech product: an app that offers
systematically managed multi-asset funds. Raw market prices and news headlines
are turned into out-of-sample backtested funds (equity-only, crypto-only, and
combined), a news-sentiment sector index, and a sentiment-fusion extension. The
deliverable is this folder (the Moodle zip), a Streamlit app deployed from a
public GitHub repo, and the Part B report. This folder is my own GitHub
repository and the app entrypoint is `streamlit_app.py` at the folder root.

Everything runs from this folder. Reuse my own Part A approach but rebuild the
Part B outputs here from raw data - never copy Part A output files.

## Data and reproducibility rules (non-negotiable)

- Load ALL raw data through `src/data_access.py` (hosted ZIP, cached). Never
  read raw files directly and NEVER commit raw data or secrets.
- The full pipeline is one script: `python scripts/run_part_b.py`. It must
  reproduce every artifact from scratch: results/data/*.csv,
  results/tables/*.csv, results/figures/*.png. Do not hand-edit those files.
- The deployed Streamlit app reads ONLY precomputed `results/` artifacts. It
  must NOT import nltk, must NOT score headlines, and must NOT recompute
  backtests (the free Streamlit tier cannot). nltk stays in
  requirements-dev.txt; requirements.txt stays slim for the deployed app.
- Required artifact filenames (markers check these exact names):
  `results/data/fund_returns.csv`, `results/data/fund_weights.csv`,
  `results/data/sector_sentiment_index.csv`,
  `results/tables/performance_metrics.csv`.

## No look-ahead in backtests

- Fund weights for a rebalance date use only the trailing estimation window up
  to and including that date - nothing after. The first rebalance must equal an
  `optimize_weights` call on exactly the first `window` rows.
- Sentiment aligned to trading day t is usable only from day t+1: the signal is
  lagged at least one trading day before it drives any trade. A Saturday or
  Monday headline aligned to Monday is first usable for Tuesday's trade.
- Turnover/cost models must not let a future observation leak into a decision.
- I enforce these with `tests/` - if you change backtest or sentiment code, the
  look-ahead tests must still pass.

## Coding conventions

- Python 3.13, repo-local `.venv`; use `.\.venv\Scripts\python.exe` on Windows.
- Keep reusable logic in `src/` modules (etl, features, portfolios, sentiment,
  fusion, data_access); keep the orchestration in `scripts/run_part_b.py`;
  keep unit tests in `tests/`; keep the app in `streamlit_app.py`.
- Figures follow the Economist-style design system I carry from Part A (red
  #E3120B, blue #0D5691, grey #666666; no top/right spines; light grid).
- Ruff, line length 100: `python -m ruff check scripts src tests
  streamlit_app.py`. Match the config in the fins-agent repo pyproject.toml.
- No unused imports, no dead code, no placeholder comments. Keep the app light
  so it runs on a basic machine.

## Folder layout

- `src/` - importable library code (the engine).
- `scripts/` - `run_part_b.py` (pipeline) and `check_handin.py` (marker check).
- `tests/` - unit tests, synthetic data only, no network.
- `results/data/`, `results/tables/`, `results/figures/` - committed artifacts.
- `report/` - the Word-first report (report.docx -> report.pdf).
- `ai/` - prompt logs and AI_NOTES.md (graded evidence of how AI was used).
- `context/` - provided briefs (do not edit).
- `docs/` - provided helpers.

## How I check your work

1. `python scripts/run_part_b.py` - must run end-to-end with no exceptions and
   show the same headline numbers each run.
2. `python -m pytest -q` - all tests pass (currently 31).
3. `python scripts/check_handin.py` - no [FAIL] before hand-in.
4. `streamlit run streamlit_app.py` - must boot and load all tabs locally.
5. `python -m ruff check scripts src tests streamlit_app.py` - clean on my own
   files.
6. I read diffs for look-ahead: rebalance windows, signal lags, and any place
   a future row could feed a past decision. If you are unsure whether a change
   could leak, flag it instead of shipping it.

## Task routing

- Setup/tooling problems: sort the environment first, then the code.
- Backtest/fund changes: edit `src/portfolios.py` + tests, then rerun the
  pipeline so results/ matches.
- Sentiment changes: edit `src/sentiment.py` + `src/fusion.py` + tests, rerun
  pipeline, confirm `results/data/sector_sentiment_index.csv` and the fusion
  tables refresh.
- App changes: edit `streamlit_app.py` only; keep it reading `results/`.
- Report writing: follow `docs/ai/rules/*` and context/verify_ai_output.md.
- Never report a statistic I did not produce in this folder - every number in
  the report and app must trace back to an artifact or a test.

## Honesty

The AI workflow pack is graded on honesty. Where an AI suggestion was wrong
(for example the risk-parity objective or the VADER lexicon leak), the prompt
log must record what was wrong, how I caught it, and what I changed. Do not
paper over mistakes.
