# Changelog

## Unreleased

- Brackets `()`, `[]`, `<>`, `{}` are punctuation: drop the characters, keep enclosed words (`(cicho) start → cicho start`); previously `(…)`, `[…]`, `<…>` content was deleted. `BasicTextNormalizer` behaves the same.
- Docs: sync `README.md` / `docs/index.md` pipeline, bracket behavior, test count (870+), fix PL ordinal example (`pierwszego → 1`).
- Strip invisible formatting chars (U+200B-200F, U+202A-202E, U+2060-2064, U+2066-2069, U+FEFF) in all normalizers; formatting-only hypotheses normalize to `''` (10).
- Convert guarded Roman numerals (`XI wiek → 11 wiek`); all-uppercase + valid + followed by a word, so `i`/`Ci`/`mi`/lowercase stay (11).
- Polish dot thousands separator (`60.000 → 60000`, `1.000.000 → 1000000`); decimal comma untouched (`60,000 → 60`) (12).
- Reconcile `tys./mln/mld` (`50 tys. → 50000`), `kilometr* → km`, percent adjectives (`20-procentowy`, `dwudziestoprocentowy → 20%`) (13, 14).
- Keep `pół żartem/pół serio` idioms as words; partitive `jedną/jeden z …` stays a word while `jedną unię → 1 unię` (15, 16).
- Consistent hyphens (`omega-3 → omega 3`, `70-te → 70`, `70-latek → 70 latek`); leading `-10` kept (17).
- Perf: skip no-op regex scans via literal/digit guards (`sub_if_present`, `contains_digit`); ~1.6x faster (0.265 → 0.161 ms/call), output unchanged.

## 0.1.4 – 2026-09-07

- Fix publish workflow: `tomllib` is 3.11+ only, use regex parse so `3.10` runner publishes correctly
- Fix `scripts/bump_patch.py` to only bump when staged changes exist (avoid bump on `pre-commit run`)
- Docs: sync `index.md` / `api.md` with split modules and `21.05` date format

## 0.1.3 – 2026-09-07

- Chore: auto bump patch before commit (`scripts/bump_patch.py` + `.pre-commit-config.yaml` local hook + `.git/hooks/pre-commit` fallback, `pre-commit install` enabled)

## 0.1.2 – 2026-09-07

- Docs: clean `README` (remove sparkles), add Architecture section

## 0.1.1 – 2026-09-04

- Guard publish if version already on PyPI (`publish.yml`)

## 0.1.0 – 2026-09-03

- Initial Polish port of Whisper `english.py` / `basic.py` (diacritics-aware)
- Cardinal/ordinal with Morfeusz2 declensions, fractions, half, currency, percent, time, months (conditional)
- Strict typing (`py.typed`, `mypy`), ruff, pre-commit, CI, MkDocs

## Unreleased (refactor)

- Split monolithic `polish.py` (1172 lines) into `lemmatizer.py`, `utils.py`, `numbers.py`, `time.py`, `text.py` with `polish.py` shim for compat
- Fix diacritic-less ASR: `trzysta czterdziesci osiem` → `348` (was `300 czterdziesci 808 piec 7`) – expand lexicons with `utils.with_ascii_variants` and build declined ASCII map via `Morfeusz.generate`
- Handle `wpol`/`polnoc`/`poludnie`/`pol`/`poltora` and fractions/months via same map; `time`/`text` now delegate fully to `PolishLemmatizer`
- Ellipsis (`...`, `…`) no longer leaks stray `.` tokens; punctuation separates numerals (`10, 500, 90` → `10 500 90`) while valid composition (`dwa tysiące dwadzieścia trzy` → `2023`) is kept
- Expand year abbreviation `r.`/`r` → `roku` (`w r. 1860` → `w roku 1860`); accept colloquial `ośmnaście` → `18`
- Drop trailing ordinal dots in `PolishTextNormalizer` output so `piętnasty` ≡ `15.` ≡ `15` for WER
- Time: already-digit `godzina 20.15` → `20:15`, `o godzinie 16.05` → `o 16:05`, `o godzinie szesnastej piątej` → `o 16:05`
