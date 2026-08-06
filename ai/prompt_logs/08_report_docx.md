# Prompt log 08 - Part B report.docx (Stations 3-4), innovations, checklist

## What I wanted
1. Author `report/report.docx` (and export `report/report.pdf`) for Part B,
   following the six-section structure the brief suggests (funds + backtest
   design; OOS results + fact sheets; the sentiment index; extensions and
   innovations; the app and the investor journey; critical reflection with three
   concrete recommendations). Every number must trace to a results/ artifact.
2. Detail the innovations for Station 3 and Station 4 in the report, and make
   the standalone Sentiment Index and the Fusion Extension explicit deliverables.
3. Update SUBMISSION_CHECKLIST.md and tick every item that is verifiably done.

## Prompt(s)
"Create report/report.docx under z5494216_projectB/report. Update the
innovations detailed in the report for Station 3 and 4 for Part B, alongside
taking note on the Sentiment Index (standalone) and the Fusion Extension.
Update and then tick every item in SUBMISSION_CHECKLIST.md."

## What the assistant produced
A generator script that builds the docx with python-docx, reading every number
live from the results/ tables (performance_metrics, fact_sheets,
fusion_comparison, fusion_kappa_sensitivity, turnover,
vader_extension_before_after, sentiment_coverage, news_volume_volatility,
news_volume_volatility_correlation, backtest_parameters, sector_sentiment_index)
so the report cannot drift from the artifacts. The report has:

- Title block + automatic TOC field, then six sections.
- Section 1: funds/universes/methods + backtest design (Table 1 from
  backtest_parameters.csv) and the no-look-ahead statement.
- Section 2: OOS results, Table 2 (all 14 funds) and Figures 1-4 (growth,
  drawdown, weights over time, Sharpe barplot), fact-sheet template + Appendix A.
- Section 3: the standalone sentiment index - headlines/dedup, extended VADER,
  neutral-fill policy, one-day lag, sector means; Figure 5.
- Section 4 (Extensions and innovations): 4.1 custom finance lexicon (Table 3
  before/after), 4.2 true ERC risk parity + cap pinning + weight-diversity
  diagnostic, 4.3 turnover/cost model, 4.4 the fusion extension (Table 4
  before/after + Figure 6 + honest sweep read), 4.5 the news-volume-vs-forward-
  volatility cross-check (Figure 7, mixed result incl. negative Energy), 4.6 the
  custom design system.
- Section 5: Station 4 app innovations (5.1) and the target user + customer
  journey (5.2), the four-tab journey, the four-file artifact contract, and a
  screenshot placeholder.
- Section 6: critical reflection with three concrete, evidence-based
  recommendations (sector-conditional volatility-scaled sentiment; turnover as a
  constraint; rebuild/drop Maximum Sharpe).
- References (marked [HUMAN EDIT REQUIRED]) and Appendix A-D (fact sheets,
  turnover, kappa sweep, news-volatility evidence).
- Every AI-drafted prose paragraph is prefixed with a red [DRAFT] tag for the
  student to rewrite, and a final caption lists the Word tasks (update TOC,
  screenshots, cross-references, verify references, accessibility check,
  re-export PDF).

## What was wrong or risky
1. The fusion exhibit initially had a figure and prose but no dedicated
   before/after TABLE, but the brief requires the fusion "as a table and a
   figure". Added Table 4 (base/fused Sharpe, change, growth of $1).
2. The outline earlier framed the fusion result around the best kappa cells
   (EW kappa 3, MinVar kappa 2). The shipped fused funds actually use kappa = 1
   (fusion_comparison.csv), so the report now leads with the shipped numbers
   (EW 0.4162 -> 0.4138, MinVar 0.2566 -> 0.2856) and reports the sweep (Appendix
   C) as the honest full picture, not a cherry-pick.
3. Page count: the narrative must stay within 10 pages excluding references and
   appendix. Verified via Word that sections 1-6 end on page 10 (References on
   page 11; appendix on 12-15). Total 15 pages / ~3,960 words.
4. The checklist contains items I cannot truthfully tick: the public GitHub repo
   and live app, the student's own rewrite of the drafted prose, and the final
   Moodle submission. I left those unticked with explicit STILL TO DO notes
   rather than mark them done.

## What I changed and why
1. Added the fusion comparison table to satisfy "table and a figure".
2. Corrected the fusion framing to the shipped kappa = 1 numbers + full sweep.
3. Confirmed the 10-page narrative constraint programmatically via Word.
4. Rewrote SUBMISSION_CHECKLIST.md item-by-item: ticked the mechanical items
   (folder name, report.pdf present, exhibits, combined fund + OOS backtest +
   fact sheet, app runs locally, data-access only, own AGENTS/CLAUDE files, ai/
   logs) and left the three student-action items unticked with concrete next
   steps.

## Verification
- `python scripts/run_part_b.py` unchanged (numbers in the report match the
  artifacts it produces).
- Word: report.docx/report.pdf rebuilt; narrative ends page 10.
- `python scripts/check_handin.py`: 22 checks passed, 1 warning (pycache).
- Every quantitative claim in the report reads from a results/ CSV at build
  time (verify_ai_output.md rule: no number from memory).
