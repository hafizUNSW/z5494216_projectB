# Prompt log 05 - the Quantis Streamlit app

## What I wanted
A deployed-ready Streamlit app for the full investor journey: compare the
funds, read a fund's fact sheet (growth of $1, drawdown, Sharpe, holdings),
set an allocation, and see the sentiment analytics. It must read only
precomputed `results/` and run on a basic machine.

## Prompt(s)
"Replace the starter streamlit_app.py with a Quantis dashboard: Compare /
Fact sheet / Allocate / Sentiment tabs. Read only results/data and
results/tables with st.cache_data. No sentiment scoring, no backtest
recomputation. Keep the Economist-style design system."

## What the assistant produced
A four-tab app: compare table + growth and risk/return charts; a fact-sheet
tab with metric cards, growth and drawdown, current top-10 holdings, and a
weights-over-time view; an allocation tab with per-fund sliders and a blended
portfolio; and a sentiment tab with the sector index, the lagged signal, and
the fusion before/after table.

## What was wrong or risky
- The fused "+ Sentiment" funds were in fund_returns.csv but not in
  performance_metrics.csv, so the app could not offer them in Compare or Fact
  sheet. Fixed in run_part_b.py: fused funds now get full metrics, fact-sheet
  and holdings rows, so the app lists all 14 funds.
- The hand-in checker warns if the app text contains "nltk" (deploy guard).
  A docstring comment mentioned it, so the check flagged the app. Reworded
  the comment.
- Verify button edge cases: allocation of 0% total, empty multiselect - now
  handled with info messages instead of crashes.

## What I changed and why
Verified the app with the Streamlit testing harness (all tabs run with no
exceptions, base and fused funds both render) and with a headless
`streamlit run` returning HTTP 200. Deployment to Streamlit Cloud remains my
browser step, as required.
