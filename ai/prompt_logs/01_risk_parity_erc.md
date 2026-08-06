# Prompt log 01 - risk parity (ERC) done properly

## What I wanted
A risk-parity method for the walk-forward backtests that produces true equal
risk contribution across assets, respecting the 20% / 35% weight caps, with a
test proving the contributions are equal.

## Prompt(s)
"Add a risk parity method to src/portfolios.py that equalises risk
contributions (ERC) with a per-asset cap. Write a test that asserts risk
contributions are equal to tight tolerance and that no weight exceeds the
cap."

## What the assistant produced
A first cut using scipy's SLSQP minimising `sum((RC - target)^2)` over
weights, where `target = total_risk / n`. It found weights whose contributions
were spread around the target but not equal once the cap bound.

## What was wrong or risky
- With the cap binding, `sum((RC - target)^2)` is minimised by a compromise,
  not by equality - true ERC was not achieved (RC spread ~1e-5 even at 60
  assets).
- Slow (SLSQP with numerical Jacobians), and slow enough to be a pain in the
  60-asset backtest.

## What I changed and why
Rewrote it as the standard ERC optimisation: minimise `0.5 x'Σx - Σ log x`
with weights summing to one, solved with Newton's method on the KKT system
(quadratic convergence), then handle the cap recursively - pin capped assets,
re-run ERC on the free subset, scale down. The new test
`test_risk_parity_equal_contributions` asserts RC spread ~1e-11, and
`test_risk_parity_respects_caps` asserts the cap. Roughly 150x faster than the
SLSQP version.

---

### What I wanted
A clean long-only risk-parity that never lets a weight exceed the cap after
normalisation.

### Prompt(s)
"Make sure the risk-parity weights respect the cap exactly - the cap must hold
on the final weights, not just mid-solve."

### What the assistant produced
`normalize_capped`: a water-filling projection that renormalises below-cap
weights while keeping pinned weights at the cap, used in `walk_forward_backtest`
and the fusion tilt. This replaced `clip + renormalise`, which pushed weights
back over the cap.

### What was wrong or risky
clip-then-renormalise silently violated the cap for the fused/tilted weights.

### What I changed and why
Switched every capped weight to `normalize_capped` and added
`test_normalize_capped` covering simplex (no cap), capped, and over-cap-input
cases.

---

## What I changed and why
See above - the SLSQP compromise is the kind of "looks like risk parity" result
that a marker could reproduce and find wanting. The ERC rewrite plus the two
tests make the claim checkable.
