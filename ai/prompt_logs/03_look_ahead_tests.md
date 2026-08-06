# Prompt log 03 - look-ahead-safe backtests and sentiment signal

## What I wanted
Out-of-sample walk-forward backtests and a sentiment signal with provably no
look-ahead, enforced by tests so a future edit cannot silently leak.

## Prompt(s)
"Write the walk-forward backtest so weights for a rebalance date use only the
trailing window up to that date, and lag the sentiment signal by one trading
day before it is tradable. Add tests that pin both."

## What the assistant produced
The backtest in src/portfolios.py: weights optimised on the window ending at
the rebalance date, applied from the next period. The sentiment signal in
src/sentiment.py lagged by one trading day.

## What was wrong or risky
Two subtle risks a code reviewer would miss:
- A "window + rebalance" off-by-one that could pull the first live date one
  period too early (the weight decision and the first application must not
  coincide).
- A signal built with a centred rolling mean (uses `t+1`) instead of a
  trailing mean, which would make day-t decisions use the future.

## What I changed and why
I added two tests that pin the invariants rather than trusting the code:
- `test_walk_forward_no_look_ahead_first_weight`: the first rebalance weight
  must equal `optimize_weights(wide.iloc[:window], ...)` exactly.
- `test_ticker_signal_wide_is_look_ahead_safe`: signal at `t` equals the mean
  of aligned sentiment over the window ending at `t-1`, checked against a
  hand-built trailing mean for a sample of dates.
Plus `test_first_live_date_is_window_index` asserting the first live date is
index `window`. All 31 tests pass.

## What I changed and why
See above - the tests are the evidence that the backtest and signal are
look-ahead-safe, which is what the report claims.
