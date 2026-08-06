# Submission checklist - Part B

Tick every item before you hand in. Run `python scripts/check_handin.py` to verify
the mechanical ones.

- [x] Folder is named <yourZID>_projectB (this folder: z5494216_projectB).
- [x] report/report.pdf is present (authored in Word as report/report.docx, exported
      to PDF; narrative sections 1-6 run pages 2-10, within the 10-page limit;
      exhibits in the appendix). Final re-export needed after you rewrite the
      red [DRAFT] paragraphs (item below).
- [x] The report includes every required exhibit from PROJECT_BRIEF.md, Section 5
      ("Required exhibits (Part B)"), each captioned and interpreted:
      performance table (Table 2), growth of $1 (Fig 1), drawdown (Fig 2),
      weights over time (Fig 3), Sharpe barplot (Fig 4), sentiment index
      (Fig 5), and the fusion before-vs-after as a table (Table 4) and a figure
      (Fig 6). Extra exhibits: lexicon before/after (Table 3) and the news
      attention vs forward volatility cross-check (Fig 7, Table D1/D2).
- [x] At least the required combined fund with two methods, backtested
      out-of-sample with no look-ahead, with a fact sheet (we ship 12 base
      funds across 3 universes x 4 methods plus 2 sentiment-tilted equity funds;
      look-ahead pinned by tests; fact sheets in Appendix A).
- [x] streamlit_app.py runs locally: streamlit run streamlit_app.py (verified
      with the Streamlit testing harness and headless HTTP 200; reads only the
      four required results/ CSVs).
- [x] The folder is its own GitHub repository, committed and pushed to a
      PUBLIC repo: github.com/hafizUNSW/z5494216_projectB (branch main),
      verified that no raw data or secrets are committed.
- [x] The live Streamlit app is deployed from Streamlit Community Cloud
      (repo: hafizUNSW/z5494216_projectB, branch main, entrypoint
      streamlit_app.py) and loads: https://z5494216projectb-3un3u83swdfcueilspfrhq.streamlit.app/
- [x] Raw data loads through src/data_access.py; no raw data or secrets committed
      (results/ artifacts - the CSVs the app reads - are committed;
      check_handin.py verifies the guard).
- [x] AGENTS.md or CLAUDE.md (your tool's file) is YOUR own, not the stub
      (both AGENTS.md and CLAUDE.md replaced with real instructions).
- [x] ai/ contains your prompt logs and AI notes (ai/AI_NOTES.md + prompt logs
       01-08).
- [ ] STILL TO DO (must be your own words): the writing and interpretation are
      your own. Every AI-drafted paragraph in report/report.docx is marked with
      a red [DRAFT] tag - rewrite each in your own words, delete the tags, then
      re-export report.pdf and update the report/report.docx last-rebalance
      facts if any fund data changed.
- [ ] STILL TO DO (final step): submit the zip to Moodle, the public repo link,
      and the live Streamlit URL.
