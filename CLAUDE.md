# CLAUDE.md - agent instructions (mirror of AGENTS.md)

I work with an opencode assistant, which reads AGENTS.md at this folder's root.
That file is the authoritative set of instructions for this project and is
graded as my AI workflow evidence. This CLAUDE.md mirrors it so the same rules
apply if this repo is ever opened in Claude Code.

Read PROJECT_BRIEF.md first, then context/ (DATA_GUIDE.md, project_context.md,
verify_ai_output.md). Those override anything below.

## Project

**Quantis** - my Part B FinTech product: an app offering systematically
managed multi-asset funds built from raw prices and news headlines via
out-of-sample backtests, a news-sentiment sector index, and a sentiment-fusion
extension. The folder is my own GitHub repo; the app entrypoint is
`streamlit_app.py` at the folder root.

## Non-negotiable rules

- Load raw data only through `src/data_access.py`; never commit raw data or
  secrets.
- Reproduce everything with `python scripts/run_part_b.py` - never hand-edit
  files under `results/`.
- The deployed app reads only precomputed `results/` artifacts; it must not
  import nltk, score headlines, or recompute backtests.
- Keep exact artifact filenames: `results/data/fund_returns.csv`,
  `results/data/fund_weights.csv`, `results/data/sector_sentiment_index.csv`,
  `results/tables/performance_metrics.csv`.
- No look-ahead: rebalance weights use only the trailing window; sentiment
  aligned to day t is tradable only from day t+1. Tests in `tests/` enforce
  this - never break them.
- Keep reusable logic in `src/`, orchestration in `scripts/`, tests in
  `tests/`, the app in `streamlit_app.py`.
- Ruff, line length 100; Economist-style figures (red #E3120B, blue #0D5691);
  match the fins-agent repo pyproject.toml config.

## How I check your work

`python scripts/run_part_b.py` runs end-to-end with the same headline numbers;
`python -m pytest -q` is green (currently 31 tests); `python
scripts/check_handin.py` shows no [FAIL]; `streamlit run streamlit_app.py`
boots and loads every tab; `python -m ruff check scripts src tests
streamlit_app.py` is clean on my own files. I read diffs for look-ahead leaks
and flag uncertainty rather than shipping a possible leak.

## Honesty

The AI workflow pack is graded on honesty. Where an AI suggestion was wrong,
record it in `ai/` prompt logs: what was wrong, how I caught it, what I
changed.
