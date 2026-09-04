# Polish Whisper Normalizer

<p align="center">
  <a href="https://github.com/NeonFeline/polish-whisper-normalizer/actions"><img alt="CI" src="https://github.com/NeonFeline/polish-whisper-normalizer/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/polish-whisper-normalizer/"><img alt="PyPI" src="https://img.shields.io/pypi/v/polish-whisper-normalizer.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-%3E%3D3.10-blue"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
  <a href="https://neonfeline.github.io/polish-whisper-normalizer/"><img alt="Docs" src="https://img.shields.io/badge/docs-MkDocs-blueviolet"></a>
</p>

> **Polish port of [OpenAI Whisper](https://github.com/openai/whisper) normalizers** — keeps `ąćęłńóśźż`, normalizes numbers/time/currency/dates with **Morfeusz2** declensions.

[🇬🇧 English](#english) | [🇵🇱 Polski](#polski)

---

## English

### Features

| Area | Example | Normalized |
|---|---|---|
| **Diacritics** | `Żółć!` | `żółć` (kept) / `zolc` with `remove_diacritics=True` |
| **Cardinal / ordinal** | `sto dwadzieścia trzy`, `dwudziestu pięciu`, `pierwszego → 1.` | `123`, `25`, `1.` |
| **Time** | `piąta trzydzieści`, `wpół do ósmej`, `o piątej`, `piąta rano`, `od piątej do szóstej` | `5:30`, `7:30`, `o 5:00`, `5:00 rano`, `od 5:00 do 6:00` |
| **Time with `minut`** | `dziesięć minut po piątej`, `za dwadzieścia minut ósma` | `5:10`, `7:40` |
| **Fractions** | `jedna trzecia`, `trzy czwarte` | `1/3`, `3/4` (keeps `/`, idempotent) |
| **Half** | `pół litra`, `półtora`, `dwa i pół` | `0.5 litra`, `1.5`, `2.5` |
| **Currency** | `pięć złotych`, `€10`, `pięć złotówek` | `5 zł`, `10 €`, `5 zł` |
| **Percent** | `pięć procentów` | `5%` |
| **Ordinal multipliers** | `tysiąc dziewięćsetny` | `1900.` |
| **Dates** ✨ | `5 maja` → `05.05` · `piątego maja` → `05.05` · `piątego maja 2026` → `05.05.2026` · `piątego maja roku dwa tysiące dwudziestego szóstego` → `05.05.2026` | `DD.MM` / `DD.MM.YYYY` zero-padded, **conditional** (`maja` alone stays `maja`, avoids `Maja`→`5`) |
| **Geographic guard** | `na północ` stays, `jest północ` → `jest 0:00` | no false `0:00` for north |

**Pipeline:** `lower → brackets/parens → time → decimal `,`→`.` → `remove_symbols(keep=".:/%$€£¢+-")` → `numbers` → `months (conditional)` → `dates` → cleanup. Configurable via `PolishTextNormalizer(date_format=...)`.

### Installation

```bash
uv sync                 # dev + morfeusz2
uv pip install -e .     # editable
# or from PyPI (once published)
uv pip install polish-whisper-normalizer
```

Requires **Python ≥3.10**, `regex`, `more-itertools`, `morfeusz2`.

### Usage

```python
from polish_whisper_normalizer import PolishTextNormalizer, BasicTextNormalizer

n = PolishTextNormalizer()
n("Było piętnaście po piątej, minus dziesięć stopni.")
# → "było 5:15 -10 stopni"

n("Spotkanie dwudziestego pierwszego maja o piętnastej trzydzieści.")
# → "spotkanie 21.05 o 15:30"   # was "21. 5" before 05.05 fix

n("piątego maja roku dwa tysiące dwudziestego szóstego")
# → "05.05.2026"   # uniform, no trailing "r"

n("5 maja")  # digit day works too
# → "05.05"

n("pięć złotówek, pięć procentów, pół litra, jedna trzecia")
# → "5 zł 5% 0.5 litra 1/3"

# custom date format
PolishTextNormalizer(date_format="%Y-%m-%d")("piątego maja 2026")
# → "2026-05-05"
PolishTextNormalizer(date_format="{day}/{month}/{year}")("piątego maja 2026")
# → "5/5/2026"

# diacritics
BasicTextNormalizer()("Żółć!")  # → "żółć"
BasicTextNormalizer(remove_diacritics=True)("Żółć!")  # → "zolc"

# jiwer WER (words→digits, dates, time normalized before WER)
# pip install polish-whisper-normalizer[jiwer]
import jiwer
from polish_whisper_normalizer.jiwer import wer, PolishTransform, polish_transform
wer("piątego maja 2026", "05.05.2026")  # → 0.0  (raw jiwer.wer → 1.0)
jiwer.wer("piątego maja 2026", "05.05.2026",
          reference_transform=polish_transform,
          hypothesis_transform=polish_transform)  # → 0.0
# custom date_format
wer("piątego maja 2026", "2026-05-05", date_format="%Y-%m-%d")  # → 0.0
```

<details><summary>Components</summary>

```python
from polish_whisper_normalizer import PolishNumberNormalizer, PolishTimeNormalizer, PolishLemmatizer

PolishNumberNormalizer()("dwudziestu pięciu złotych")  # → "25 zł"
PolishTimeNormalizer()("wpół do ósmej")  # → "7:30"
PolishLemmatizer().analyse("dwudziestu")  # → [("dwadzieścia","num")]
```

| Class | Description |
|---|---|
| `PolishTextNormalizer(date_format="{day:02d}.{month:02d}.{year}")` | Full pipeline, `date_format` supports `str.format` (`{day}`, `{month}`, `{year}`) and `strftime` (`%d.%m.%Y`) |
| `PolishNumberNormalizer` | Words → digits, currency/percent/decimals/signs/ordinals |
| `PolishTimeNormalizer` | Spoken time → `HH:MM`, guards `o 5. stronie` vs `o 5:00`, `na północ` vs `0:00` |
| `PolishLemmatizer` | Morfeusz2 wrapper |
| `BasicTextNormalizer` | Whisper `basic` but diacritics-aware |

</details>

### API Docs

Full API: **[neonfeline.github.io/polish-whisper-normalizer](https://neonfeline.github.io/polish-whisper-normalizer/)** (`mkdocs serve` locally).

### Development

```bash
uv sync --group dev
uv run pytest -q          # 486 tests
uv run mypy src           # strict, py.typed
uv run ruff check src tests && uv run ruff format --check src tests
uv build
mkdocs serve
```

- `py.typed` + `mypy --strict` (`warn_unused_ignores=false`)
- `ruff` + `pre-commit` + GitHub Actions (`ci.yml`: lint → mypy → pytest --cov → build)
- Validated on **BIGOS v2 + PELCRA** (2397 samples, ~5 % number words)

### License

MIT — see `LICENSE` (inherits Whisper MIT for `basic.py`).

---

## Polski

### Funkcje

| Obszar | Przykład | Po normalizacji |
|---|---|---|
| **Znaki diakrytyczne** | `Żółć!` | `żółć` (zachowane) |
| **Liczebniki** | `sto dwadzieścia trzy`, `pierwszego` | `123`, `1.` |
| **Czas** | `piąta trzydzieści`, `o piątej`, `od piątej do szóstej` | `5:30`, `o 5:00`, `od 5:00 do 6:00` |
| **Daty** ✨ | `5 maja` → `05.05` · `piątego maja 2026` → `05.05.2026` | `DD.MM` / `DD.MM.RRRR`, **warunkowo** (`maja` samo → `maja`) |
| **Waluta / procent / ułamki** | `pięć złotówek`, `procentów`, `1/3`, `pół litra` | `5 zł`, `5%`, `1/3`, `0.5 litra` |

**Potok:** `lower → czas → liczby → miesiące (warunkowo) → daty`.

### Instalacja i użycie (PL)

```bash
uv sync
```

```python
from polish_whisper_normalizer import PolishTextNormalizer
n = PolishTextNormalizer()  # date_format="{day:02d}.{month:02d}.{year}" domyślnie

n("piątego maja roku dwa tysiące dwudziestego szóstego")
# → "05.05.2026"  # jednolicie, bez "r"

n("5 maja")  # też działa
# → "05.05"

# własny format daty
PolishTextNormalizer(date_format="%Y-%m-%d")("piątego maja 2026")
# → "2026-05-05"
```

`maja` jako imię `Maja` zostaje `maja` (nie `5`), `na północ` nie staje się `0:00`.

### Rozwój / testy

Jak wyżej — `uv run pytest`, `mypy`, `ruff`, `mkdocs serve`.

---

<p align="center"><sub>Made for <a href="https://huggingface.co/datasets/amu-cai/pl-asr-bigos-v2">BIGOS</a> / <a href="https://openai.com/research/whisper">Whisper</a> WER — PRs welcome!</sub></p>
