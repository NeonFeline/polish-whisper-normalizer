# Polish Whisper Normalizer

Polish text normalizer for Whisper ASR output – preserves `ąćęłńóśźż`, normalizes numbers, time, currency, dates, fractions, half and declensions via Morfeusz2. Diacritic-less ASR (`czterdziesci`, `piec`) is handled automatically.

```python
from polish_whisper_normalizer import PolishTextNormalizer

n = PolishTextNormalizer()
n("Spotkanie dwudziestego pierwszego maja o piętnastej trzydzieści.")
# → "spotkanie 21.05 o 15:30"
n("trzysta czterdziesci osiem")
# → "348"
```

## Quickstart

```bash
uv sync                 # install + morfeusz2
uv run pytest -q        # 500+ tests
uv run mkdocs serve     # docs at http://127.0.0.1:8000
```

## Features

- **Diacritics** – `ąćęłńóśźż` preserved, diacritic-less fallback via Morfeusz
- **Numbers** – cardinal/ordinal with declensions via `PolishLemmatizer` (Morfeusz2)
- **Time** – `HH:MM`, `wpół do`, `za 15`, `po 15`, `o 5:00`, `5:00 rano`, `od 5:00 do 6:00`
- **Currency/percent/fractions/half** – `5 zł`, `5%`, `1/3`, `0.5 litra` (fraction via ordinal lemmas)
- **Dates** – `5 maja → 05.05`, `piątego maja 2026 → 05.05.2026`, conditional `maja` vs `Maja`
- **Geographic guard** – `na północ` stays, `jest północ → 0:00`

See [API](api.md) for details.

## Pipeline

`lower → brackets/ignore → sentence period → time → decimal ,→. → remove_symbols → numbers → months (conditional) → dates → cleanup` (`PolishTextNormalizer` in `text.py`)

## Architecture

```
src/polish_whisper_normalizer/
  basic.py        # Whisper basic, diacritics-aware
  lemmatizer.py   # Morfeusz2 wrapper (analyse/generate)
  utils.py        # ASCII variants
  numbers.py      # cardinals/ordinals/currency
  time.py         # HH:MM
  text.py         # full pipeline
  polish.py       # re-export shim
```

## Development

```bash
uv sync --group dev
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pytest --cov --cov-report=term-missing
uv build
mkdocs serve
```
