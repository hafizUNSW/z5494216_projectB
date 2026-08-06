# Prompt log 02 - extending VADER without leaking into the plain analyser

## What I wanted
Extend VADER with a finance lexicon (terms + phrases + boosters) built from
the Week 8 course method, score all headlines, and keep the extension isolated
so the plain VADER analyser is unchanged.

## Prompt(s)
"Add src/sentiment.py: build a finance-lexicon-extended VADER analyser using
the Week 8 build-and-test approach, score the headline panel, and build the
sector sentiment index. Include rejected-term controls ('debt', 'volatile') in
the before/after evidence."

## What the assistant produced
An `extend_lexicon()` that copied the finance terms into
`analyzer.lexicon` and mutated the VADER constants used for boosters, idioms,
and negation - then, inside the same analyser instance, that was fine.

## What was wrong or risky
Under NLTK 3.10 the VADER constant dictionaries (BOOSTER_DICT,
SPECIAL_CASE_IDIOMS, NEGATE) are **class-level and shared across instances**.
Because the extension mutated them in place, every plain VADER analyser in the
same process picked up the finance lexicon too. I caught it with
`test_extension_does_not_leak_into_plain_analyzer`: score a plain sentence,
build the extended analyser, score the same sentence again, and assert the
plain score is unchanged. It failed.

## What I changed and why
`build_analyzer()` now copies the shared tables into per-instance containers
(`analyzer.lexicon`, `constants.BOOSTER_DICT`, `constants.SPECIAL_CASE_IDIOMS`,
`constants.NEGATE`) before mutating them, so the leak is impossible. Tests now
cover: plain analyser stays clean, every finance term/phrase fires, rejected
terms leave sentences unchanged, and the look-ahead-safe ticker signal matches
a trailing mean ending at `t-1`.

## What I changed and why
The isolation fix, and I kept the rejected-term controls in the before/after
CSV so the "approval step" of the lexicon process is demonstrable rather than
asserted.
