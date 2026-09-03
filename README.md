# Polish Whisper Normalizer

Polish text normalizer for Whisper ASR output – **keeps diacritics**, normalizes numbers, time, currency, dates.

Port of OpenAI Whisper `english.py` / `basic.py` for Polish, with Morfeusz2 lemmatization for declensions.

## Features

- **Diacritics preserved** – `ąćęłńóśźż` kept via `basic.remove_symbols(keep="")` (NFKC), `remove_symbols_and_diacritics` optional.
- **Cardinal & ordinal** – `sto dwadzieścia trzy→123`, `dwudziestu pięciu→25`, `pierwszego→1.`, `dwudziesty pierwszy→21.`, `tysięczny→1000.` including declensions via Morfeusz2.
- **Time** – `piąta trzydzieści→5:30`, `wpół do ósmej→7:30`, `za piętnaście ósma→7:45`, `piętnaście po piątej→5:15`, `o piątej→o 5:00`, `piąta rano→5:00 rano`, `od piątej do szóstej→od 5:00 do 6:00`, `północ→0:00`.
- **Numbers with words** – `dziesięć minut po piątej→5:10`, `za dwadzieścia minut ósma→7:40` (handles `minuta/minutę/minuty` + `jedna/dwie`).
- **Fractions** – `jedna trzecia→1/3`, `trzy czwarte→3/4` (fractions are N/M, not decimal).
- **Half** – `pół litra→0.5 litra`, `półtora→1.5`, `dwa i pół→2.5` (standalone `pół` → `0.5`; `północ/półmetek` untouched).
- **Currency** – `pięć złotych→5 zł`, `sto euro→100 €`, `€10→10 €`, `pięć złotówek→5 zł` (colloquial `złotówka` via Morfeusz).
- **Percent** – `pięć procentów→5%`, `procenty/procenta/procentem` → `%`.
- **Ordinal multipliers** – `tysiąc dziewięćsetny→1900.`, `tysiąc dziewięćset dziewięćdziesiąty dziewiąty→1999.`.
- **Months (conditional)** – `trzeciego maja→3. 5` only when preceded by day ordinal (`\d+\.  month`), `maja` alone stays `maja` (avoids `Maja` name → `5`).
- **Idempotent** – `1/3→1/3` (keeps `/`), `5:30→5:30`.

## Installation

```bash
uv sync
# or
uv pip install -e .
# with Morfeusz for declensions
uv sync --extra morfeusz  # morfeusz2 is default dependency
```

Requires Python ≥3.10, `regex`, `more-itertools`, `morfeusz2`.

## Usage

```python
from polish_whisper_normalizer import PolishTextNormalizer, BasicTextNormalizer

n = PolishTextNormalizer()
n("Było piętnaście po piątej, minus dziesięć stopni.")
# → "było 5:15 -10 stopni"

n("Spotkanie dwudziestego pierwszego maja o piętnastej trzydzieści.")
# → "spotkanie 21. 5 o 15:30"

n("pięć złotówek, pięć procentów, pół litra, jedna trzecia")
# → "5 zł 5% 0.5 litra 1/3"

# Keep diacritics (default)
BasicTextNormalizer()("Żółć!")  # → "żółć"
BasicTextNormalizer(remove_diacritics=True)("Żółć!")  # → "zolc"
```

### Components

```python
from polish_whisper_normalizer import PolishNumberNormalizer, PolishTimeNormalizer, PolishLemmatizer

PolishNumberNormalizer()("dwudziestu pięciu złotych")  # → "25 zł"
PolishTimeNormalizer()("wpół do ósmej")  # → "7:30"
PolishLemmatizer().analyse("dwudziestu")  # → [("dwadzieścia","num"), ...]
```

## API

| Class | Description |
|---|---|
| `PolishTextNormalizer` | Full pipeline: `lower → time → numbers → months (conditional)` |
| `PolishNumberNormalizer` | Words → digits, currency/percent/decimals/signs/ordinals |
| `PolishTimeNormalizer` | Spoken time → `HH:MM` |
| `PolishLemmatizer` | Morfeusz2 wrapper, `analyse(word) -> list[(lemma, pos)]` |
| `BasicTextNormalizer` | Whisper basic (diacritics-aware) |

## Development

```bash
uv run pytest -q            # 446 tests
uv run mypy src --ignore-missing-imports  # strict
uv run ruff check src tests
uv build
```

- **Strict typing** – `py.typed` marker, `mypy --ignore-missing-imports` passes, `--strict` with `warn_unused_ignores=false`.
- **Tests** – `tests/test_polish.py` (original + BIGOS/PELCRA samples), `tests/test_additional.py`, `tests/test_dataset_samples.py`.
- **Datasets validated** – BIGOS v2 + PELCRA (2397 samples): 5 % number words normalized, diacritics preserved, no `północ` geographic false-positive.

## License

MIT – see `LICENSE` (inherits Whisper MIT for `basic.py`).
