# Polish Whisper Normalizer

Polish text normalizer for Whisper ASR output – keeps diacritics, normalizes numbers/time/currency/dates.

```python
from polish_whisper_normalizer import PolishTextNormalizer

n = PolishTextNormalizer()
n("Spotkanie dwudziestego pierwszego maja o piętnastej trzydzieści.")
# → "spotkanie 21. 5 o 15:30"
```

## Quickstart

```bash
uv sync
uv run pytest -q
```

## Features

- **Diacritics** – `ąćęłńóśźż` preserved
- **Numbers** – cardinal/ordinal with declensions via Morfeusz2
- **Time** – `HH:MM`, `o 5:00`, `5:00 rano`, `od 5:00 do 6:00`
- **Currency/percent/fractions/half** – `5 zł`, `5%`, `1/3`, `0.5 litra`
- **Months conditional** – `3. maja → 3. 5` only after day ordinal

See [API](api.md) for details.

## Development

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest --cov
uv build
mkdocs serve
```
