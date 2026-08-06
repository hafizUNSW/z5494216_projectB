# Quantis - FinTech Project Part B

Part B: funds, sentiment, and the app (DFF Stations 3-4). **Quantis** is my
prototype investment product: an app that offers systematically managed
multi-asset funds (equity-only, crypto-only, and combined) built from raw
prices and news headlines, evaluated out-of-sample, plus a news-sentiment
sector index and a sentiment-fusion extension. This folder is also my public
GitHub repository; the app entrypoint is `streamlit_app.py` at the root.

## How to run

    pip install -r requirements.txt -r requirements-dev.txt   # dev adds nltk (VADER) for the build
    python scripts/run_part_b.py            # reproduces every result into results/
    python -m pytest -q                     # 31 unit tests, incl. look-ahead guards
    streamlit run streamlit_app.py          # runs the app locally

`python scripts/check_handin.py` verifies the hand-in requirements (run before
zipping/deploying).

Raw data loads through `src/data_access.py` (hosted ZIP, cached) and is never
committed. The deployed app reads the precomputed artifacts under `results/` -
those ARE committed.

## What is here

- `streamlit_app.py`   the Quantis app: Compare, Fact sheet, Allocate, Sentiment
- `.streamlit/`        app config
- `src/`               library code: data_access (provided), etl, features,
                       portfolios (backtests + ERC risk parity), sentiment
                       (extended VADER, sector index), fusion (sentiment tilt)
- `scripts/`           run_part_b.py (pipeline), check_handin.py (marker check)
- `tests/`             unit tests (synthetic data, no network)
- `results/data/`      app CSVs: fund_returns, fund_weights,
                       sector_sentiment_index, fund_holdings
- `results/tables/`    report tables: performance_metrics, fact_sheets,
                       turnover, fusion_comparison, fusion_kappa_sensitivity,
                       vader_extension_before_after, sentiment_coverage
- `results/figures/`   6 report figures (Economist-style design system)
- `context/`           provided data guide and project context (do not edit)
- `report/`            the report (see report/OUTLINE.md; author in Word)
- `ai/`                prompt logs and AI notes (graded evidence)
- `requirements-dev.txt`  build/repro-only deps (nltk); not deployed
- `AGENTS.md` / `CLAUDE.md`  my agent instructions (both edited - not stubs)

## Deploy + hand in

This folder is its own GitHub repo, independent of fins-agent. My assistant can
run the checks and push the repo; the browser deploy is mine (it needs my
login). See PROJECT_BRIEF.md Appendix D. In short:

    python scripts/check_handin.py        # must show no [FAIL]
    # commit precomputed app artifacts under results/ (the app reads them)
    # git init in this folder, push to a NEW private GitHub repo

Then connect the repo on share.streamlit.io (entrypoint streamlit_app.py). At
hand-in: make the repo PUBLIC, confirm the live app loads, submit the live URL
+ repo link, and zip this whole folder to Moodle.
