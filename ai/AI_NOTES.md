# AI notes - how I directed and checked the assistant (Quantis, Part B)

I used an opencode coding assistant for the engineering and a Word-first report
workflow (this file records the engineering side; the report work is logged in
`report/`). The rules I gave it up front are in `AGENTS.md`: no look-ahead in
backtests, raw data only through `src/data_access.py`, the deployed app reads
precomputed `results/` only, and every number must trace back to an artifact or
a test. `context/verify_ai_output.md` framed how I check AI output - treat it
as a draft, never as fact.

## Where the assistant got things wrong, and how I caught it

The three cases that cost real time were all "confident but wrong" outputs:

1. **Risk parity was not risk parity.** The assistant implemented equal risk
   contribution as a generic SLSQP objective `min sum((RC - target)^2)`.
   With the 20% cap binding, that objective no longer produces true ERC - the
   risk contributions came out spread around the target instead of equal
   (RC spread ~1e-5). I caught it by writing a test that asserts the risk
   contributions are equal to tight tolerance. I rewrote it as a proper ERC
   problem (minimise `0.5 x'Σx - Σ log x`) solved with Newton's method plus
   recursive cap pinning. The test now passes with RC spread ~1e-11 and it is
   ~150x faster than the SLSQP version.

2. **A VADER lexicon leak under NLTK 3.10.** The finance-lexicon extension
   mutated the analyser's constants in place. Under NLTK 3.10 those constant
   dictionaries are class-level and shared, so the extension silently leaked
   into the *plain* VADER analyser - finance terms were boosting non-finance
   sentences. I caught it with a test that scores a plain sentence before and
   after building the extended analyser and asserts the plain score is
   unchanged. The fix copies the shared tables into per-instance containers
   before mutating.

3. **A weight cap that did not hold.** Clipping then renormalising weights
   pushed some weights back above the cap. I caught it with a test asserting
   no weight exceeds the cap after normalisation. I replaced clip+renormalise
   with a water-filling projection (`normalize_capped`) used everywhere a cap
   must hold.

## How I checked the numbers

- Reproduced everything end-to-end with `python scripts/run_part_b.py` on a
  clean run; the headline numbers (502/957 out-of-sample days, 66.4% non-zero
  compound share, kappa sweep Sharpe values) matched the previous run.
- 31 unit tests pass (`python -m pytest -q`); the look-ahead tests pin the
  first rebalance to exactly the first `window` rows and compare the lagged
  sentiment signal to the trailing mean ending at `t-1`.
- `python scripts/check_handin.py` passes all mechanical checks.
- `streamlit run streamlit_app.py` boots and every tab loads (checked with the
  Streamlit testing harness too).

## Honest reporting of results

The sentiment fusion did not uniformly help: the Equal Weight fund's Sharpe
edged down (0.416 to 0.409 at kappa 3) while Minimum Variance improved (0.257
to 0.298 at kappa 2). I reported the sweep as measured rather than cherry-
picking the improvement - a negative result explained is worth more than a
cherry-picked positive.

## What I did myself

The design decisions (product identity Quantis, the Economist-style design
system, the kappa set, the turnover model, the fact-sheet content, which
methods to offer) were mine. The assistant implemented, and I reviewed every
diff, ran the checks above, and rewrote the three buggy areas myself.
