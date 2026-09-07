"""Polish number → digits normalization (cardinals, ordinals, currency, …).

All declension is delegated to :class:`PolishLemmatizer` (Morfeusz2) – we only
store base nominative forms and let Morfeusz canonicalize declined variants.
ASCII-folded variants are generated automatically via ``utils`` + Morfeusz
``generate`` so diacritic-less ASR output works without duplicating lexicons.
"""

from __future__ import annotations

import copy
import logging
import re
import threading
from collections.abc import Iterator
from fractions import Fraction

from .lemmatizer import PolishLemmatizer
from .utils import strip_diacritics, with_ascii_variants, with_ascii_variants_set

logger = logging.getLogger(__name__)


def _windowed_3(seq: list[str | None]) -> Iterator[tuple[str | None, str | None, str | None]]:
    """Lightweight windowed(3) to avoid more-itertools dependency."""
    for i in range(len(seq) - 2):
        yield (seq[i], seq[i + 1], seq[i + 2])  # type: ignore[misc]


class PolishNumberNormalizer:
    """
    Convert spelled-out Polish numbers into arabic digits, while handling:

    - cardinal numbers ("sto dwadzieścia trzy" -> "123")
    - grammatical variants of big multipliers ("tysiąc/tysiące/tysięcy" -> 1000)
    - currency ("pięć złotych" -> "5 zł", "pięćdziesiąt groszy" -> "50 gr")
    - percents ("dwadzieścia procent" -> "20%")
    - decimals ("trzy przecinek czternaście" -> "3.14")
    - signs ("minus dziesięć" -> "-10")
    - ordinal numbers, including declension ("dwudziesty pierwszy" -> "21.",
      "pierwszego" -> "1.", "trzeciej" -> "3.")
    - declined cardinal forms ("pięciu" -> "5", "dwóm" -> "2", "tysiąca" -> "1000")
    """

    # class-level shared state – built once, reused (thread-safe)
    _CACHE: dict[str, object] | None = None
    _CACHE_LOCK = threading.Lock()
    _DECLINED_ASCII_CACHE: dict[str, str] | None = None
    _DECLINED_LOCK = threading.Lock()

    # pre-compiled patterns (avoid recompiling per token)
    _POLISH_WORD_RE = re.compile(r"[a-ząćęłńóśźż]+")
    _NUMERIC_RE = re.compile(r"^\d+(?:\.\d+)?$")
    _NUMERIC_PREFIX_RE = re.compile(r"^\d")
    _DIGIT_RE = re.compile(r"^\d+(?:\.\d+)?$")

    def __init__(self) -> None:
        super().__init__()

        self.zeros = {"zero"}
        self.ones = {
            "jeden": 1,
            "jedna": 1,
            "jedno": 1,
            "dwa": 2,
            "dwie": 2,
            "trzy": 3,
            "cztery": 4,
            "pięć": 5,
            "sześć": 6,
            "siedem": 7,
            "osiem": 8,
            "dziewięć": 9,
            "dziesięć": 10,
            "jedenaście": 11,
            "dwanaście": 12,
            "trzynaście": 13,
            "czternaście": 14,
            "piętnaście": 15,
            "szesnaście": 16,
            "siedemnaście": 17,
            "osiemnaście": 18,
            "dziewiętnaście": 19,
        }
        self.tens = {
            "dwadzieścia": 20,
            "trzydzieści": 30,
            "czterdzieści": 40,
            "pięćdziesiąt": 50,
            "sześćdziesiąt": 60,
            "siedemdziesiąt": 70,
            "osiemdziesiąt": 80,
            "dziewięćdziesiąt": 90,
        }
        self.hundreds = {
            "sto": 100,
            "dwieście": 200,
            "trzysta": 300,
            "czterysta": 400,
            "pięćset": 500,
            "sześćset": 600,
            "siedemset": 700,
            "osiemset": 800,
            "dziewięćset": 900,
        }
        self.multipliers = {
            "tysiąc": 1_000,
            "tysiące": 1_000,
            "tysięcy": 1_000,
            "milion": 1_000_000,
            "miliony": 1_000_000,
            "milionów": 1_000_000,
            "miliard": 1_000_000_000,
            "miliardy": 1_000_000_000,
            "miliardów": 1_000_000_000,
            "bilion": 1_000_000_000_000,
            "biliony": 1_000_000_000_000,
            "bilionów": 1_000_000_000_000,
            "biliard": 1_000_000_000_000_000,
            "biliardy": 1_000_000_000_000_000,
            "biliardów": 1_000_000_000_000_000,
            "trylion": 1_000_000_000_000_000_000,
            "tryliony": 1_000_000_000_000_000_000,
            "trylionów": 1_000_000_000_000_000_000,
        }

        # ordinal numbers (base masculine-nominative forms)
        self.ones_ordinal = {
            "pierwszy": 1,
            "drugi": 2,
            "trzeci": 3,
            "czwarty": 4,
            "piąty": 5,
            "szósty": 6,
            "siódmy": 7,
            "ósmy": 8,
            "dziewiąty": 9,
            "dziesiąty": 10,
            "jedenasty": 11,
            "dwunasty": 12,
            "trzynasty": 13,
            "czternasty": 14,
            "piętnasty": 15,
            "szesnasty": 16,
            "siedemnasty": 17,
            "osiemnasty": 18,
            "dziewiętnasty": 19,
        }
        self.tens_ordinal = {
            "dwudziesty": 20,
            "trzydziesty": 30,
            "czterdziesty": 40,
            "pięćdziesiąty": 50,
            "sześćdziesiąty": 60,
            "siedemdziesiąty": 70,
            "osiemdziesiąty": 80,
            "dziewięćdziesiąty": 90,
        }
        self.hundreds_ordinal = {
            "setny": 100,
            "dwusetny": 200,
            "trzysetny": 300,
            "czterysetny": 400,
            "pięćsetny": 500,
            "sześćsetny": 600,
            "siedemsetny": 700,
            "osiemsetny": 800,
            "dziewięćsetny": 900,
        }
        self.multipliers_ordinal = {
            "tysięczny": 1_000,
            "milionowy": 1_000_000,
            "miliardowy": 1_000_000_000,
            "bilionowy": 1_000_000_000_000,
        }

        self.decimals = {*self.ones, *self.tens, *self.zeros}

        self.preceding_prefixers = {
            "minus": "-",
            "plus": "+",
        }
        self.currencies = {
            # Polish złoty / grosz
            "złoty": "zł",
            "złote": "zł",
            "złotych": "zł",
            "złotego": "zł",
            "zł": "zł",
            "pln": "zł",
            "złotówka": "zł",
            "grosz": "gr",
            "grosze": "gr",
            "groszy": "gr",
            "grosza": "gr",
            "gr": "gr",
            # euro / cent
            "euro": "€",
            "eur": "€",
            "cent": "¢",
            "centy": "¢",
            "centów": "¢",
            "centa": "¢",
            "¢": "¢",
            # dolar
            "dolar": "$",
            "dolary": "$",
            "dolarów": "$",
            "dolara": "$",
            "usd": "$",
            # funt (pound sterling)
            "funt": "£",
            "funty": "£",
            "funtów": "£",
            "gbp": "£",
            # currency symbols (canonical)
            "€": "€",
            "$": "$",
            "£": "£",
        }
        self.suffixers = {
            "procent": "%",
            "procenta": "%",
        }
        self.specials = {"przecinek", "kropka"}
        self.conjunctions = {"i"}
        self.prefixes = set(self.preceding_prefixers.values())

        self.words = {
            key
            for mapping in [
                self.zeros,
                self.ones,
                self.tens,
                self.hundreds,
                self.multipliers,
                self.ones_ordinal,
                self.tens_ordinal,
                self.hundreds_ordinal,
                self.multipliers_ordinal,
                self.preceding_prefixers,
                self.currencies,
                self.suffixers,
                self.specials,
                self.conjunctions,
            ]
            for key in mapping
        }

        # lemma -> value lookups used to canonicalize declined forms
        self.cardinal_lemmas = set(self.ones) | set(self.tens) | set(self.hundreds)
        self.multiplier_lemmas = set(self.multipliers)
        self.ordinal_lemmas = (
            set(self.ones_ordinal)
            | set(self.tens_ordinal)
            | set(self.hundreds_ordinal)
            | set(self.multipliers_ordinal)
        )
        self.ordinal_values = {
            **self.ones_ordinal,
            **self.tens_ordinal,
            **self.hundreds_ordinal,
            **self.multipliers_ordinal,
        }

        # feminine/neuter cardinal forms used as fraction numerators
        self.fraction_numerators = {
            "jedna": 1,
            "dwie": 2,
            "trzy": 3,
            "cztery": 4,
            "pięć": 5,
            "sześć": 6,
            "siedem": 7,
            "osiem": 8,
            "dziewięć": 9,
            "dziesięć": 10,
        }
        # non-number words whose declined forms we canonicalize
        self.percent_lemmas = {"procent"}
        # colloquial currency nouns
        self.currency_lemmas = {"złotówka"}
        # month lemmas -> number string (for full-date normalization)
        self.month_lemmas = {
            "styczeń": "1",
            "luty": "2",
            "marzec": "3",
            "kwiecień": "4",
            "maj": "5",
            "czerwiec": "6",
            "lipiec": "7",
            "sierpień": "8",
            "wrzesień": "9",
            "październik": "10",
            "listopad": "11",
            "grudzień": "12",
        }

        # --- ASCII-folded variants (support diacritic-less ASR output) ---------------
        # e.g. "czterdzieści" -> "czterdziesci", "pięć" -> "piec", "pięćset" -> "piecset"
        # keep original lemma sets for generating declined ASCII map (avoid generating from
        # ASCII collisions like "piec" (stove) vs "pięć" (5))
        _orig_cardinal = set(self.ones) | set(self.tens) | set(self.hundreds)
        _orig_multiplier = set(self.multipliers)
        _orig_ordinal = (
            set(self.ones_ordinal)
            | set(self.tens_ordinal)
            | set(self.hundreds_ordinal)
            | set(self.multipliers_ordinal)
        )
        _orig_percent = set(self.percent_lemmas)
        _orig_currency = set(self.currency_lemmas)
        _orig_month = set(self.month_lemmas)
        self._expand_ascii_variants()
        # re-derive composite sets after expansion
        self._rebuild_composite_sets()

        self.lemmatizer = PolishLemmatizer()
        self._canon_cache: dict[str, str] = {}
        # map stripped declined forms -> original base lemma (for diacritic-less declensions)
        # use class-level cache to avoid rebuilding via Morfeusz.generate on every instance
        _all_orig_bases = (
            _orig_cardinal
            | _orig_multiplier
            | _orig_ordinal
            | _orig_percent
            | _orig_currency
            | _orig_month
        )
        self._declined_ascii_map = self._get_or_build_declined_map(_all_orig_bases, self.lemmatizer)

    def _expand_ascii_variants(self) -> None:
        """Expand lexicons with ASCII-folded variants for diacritic-less ASR."""
        self.zeros = with_ascii_variants_set(self.zeros)
        self.ones = with_ascii_variants(self.ones)
        self.tens = with_ascii_variants(self.tens)
        self.hundreds = with_ascii_variants(self.hundreds)
        self.multipliers = with_ascii_variants(self.multipliers)
        self.ones_ordinal = with_ascii_variants(self.ones_ordinal)
        self.tens_ordinal = with_ascii_variants(self.tens_ordinal)
        self.hundreds_ordinal = with_ascii_variants(self.hundreds_ordinal)
        self.multipliers_ordinal = with_ascii_variants(self.multipliers_ordinal)
        self.currencies = with_ascii_variants(self.currencies)
        self.suffixers = with_ascii_variants(self.suffixers)
        self.specials = with_ascii_variants_set(self.specials)
        self.conjunctions = with_ascii_variants_set(self.conjunctions)
        self.fraction_numerators = with_ascii_variants(self.fraction_numerators)
        self.month_lemmas = with_ascii_variants(self.month_lemmas)
        self.percent_lemmas = with_ascii_variants_set(self.percent_lemmas)
        self.currency_lemmas = with_ascii_variants_set(self.currency_lemmas)

    def _rebuild_composite_sets(self) -> None:
        """Re-derive sets that depend on expanded lexicons."""
        self.words = {
            key
            for mapping in [
                self.zeros,
                self.ones,
                self.tens,
                self.hundreds,
                self.multipliers,
                self.ones_ordinal,
                self.tens_ordinal,
                self.hundreds_ordinal,
                self.multipliers_ordinal,
                self.preceding_prefixers,
                self.currencies,
                self.suffixers,
                self.specials,
                self.conjunctions,
            ]
            for key in mapping
        }
        self.cardinal_lemmas = set(self.ones) | set(self.tens) | set(self.hundreds)
        self.multiplier_lemmas = set(self.multipliers)
        self.ordinal_lemmas = (
            set(self.ones_ordinal)
            | set(self.tens_ordinal)
            | set(self.hundreds_ordinal)
            | set(self.multipliers_ordinal)
        )
        self.ordinal_values = {
            **self.ones_ordinal,
            **self.tens_ordinal,
            **self.hundreds_ordinal,
            **self.multipliers_ordinal,
        }
        self.decimals = {*self.ones, *self.tens, *self.zeros}
        self.prefixes = set(self.preceding_prefixers.values())

    @classmethod
    def _get_or_build_declined_map(
        cls, bases: set[str], lemmatizer: PolishLemmatizer
    ) -> dict[str, str]:
        """Return cached declined ASCII map or build via Morfeusz.generate."""
        if cls._DECLINED_ASCII_CACHE is not None:
            return copy.deepcopy(cls._DECLINED_ASCII_CACHE)
        declined: dict[str, str] = {}
        if lemmatizer.morf is not None:
            for base in bases:
                try:
                    forms = lemmatizer.generate(base)
                except Exception:
                    logger.debug("generate failed for %r", base)
                    continue
                for surface, lemma, _tag in forms:
                    clean_lemma = lemma.split(":")[0]
                    if clean_lemma != base:
                        continue
                    stripped_form = strip_diacritics(surface)
                    if stripped_form != surface and stripped_form not in declined:
                        declined[stripped_form] = base
        with cls._DECLINED_LOCK:
            if cls._DECLINED_ASCII_CACHE is None:
                cls._DECLINED_ASCII_CACHE = copy.deepcopy(declined)
        return declined

    def _canonicalize(self, word: str) -> str:
        """Map a declined number word to its base lemma, if it is one."""
        if word in self.words:
            return word
        if self._POLISH_WORD_RE.fullmatch(word) is None:
            return word
        cached = self._canon_cache.get(word)
        if cached is not None:
            return cached

        candidates: dict[str, str | None] = {
            "num": None,
            "adj": None,
            "subst": None,
            "percent": None,
            "currency": None,
        }
        for base, pos in self.lemmatizer.analyse(word):
            if pos == "num" and base in self.cardinal_lemmas:
                candidates["num"] = base
            elif pos == "adj" and base in self.ordinal_lemmas:
                candidates["adj"] = base
            elif pos == "subst" and base in self.multiplier_lemmas:
                candidates["subst"] = base
            elif pos == "subst" and base in self.percent_lemmas:
                candidates["percent"] = base
            elif pos == "subst" and base in self.currency_lemmas:
                candidates["currency"] = base

        result = word
        for key in ("num", "adj", "subst", "percent", "currency"):
            if candidates[key] is not None:
                result = candidates[key]  # type: ignore[assignment]
                break
        # fallback: diacritic-less declined form (e.g. "pieciu" -> "pięć")
        if result == word and word in self._declined_ascii_map:
            result = self._declined_ascii_map[word]
        self._canon_cache[word] = result
        return result

    def process_words(self, words: list[str]) -> Iterator[str]:
        prefix: str | None = None
        value: str | int | None = None
        ordinal = False
        skip = False

        def to_fraction(s: str) -> Fraction | None:
            try:
                return Fraction(s)
            except ValueError:
                return None

        def output(result: str | int) -> str:
            nonlocal prefix, value, ordinal
            result = str(result)
            if prefix is not None:
                result = prefix + result
            if ordinal:
                result += "."
            value = None  # type: ignore[assignment]
            prefix = None
            ordinal = False
            return result

        if len(words) == 0:
            return

        for prev, current, next in _windowed_3([None, *words, None]):  # type: ignore[list-item]
            if skip:
                skip = False
                continue
            if current is None:  # type narrowing for mypy; never happens for non-empty words
                continue

            next_is_numeric = next is not None and self._NUMERIC_RE.match(next) is not None
            has_prefix = current[0] in self.prefixes
            current_without_prefix = current[1:] if has_prefix else current  # type: ignore[index]

            if self._NUMERIC_RE.match(current_without_prefix):  # type: ignore[arg-type]
                # arabic numbers (potentially with signs/currency prefixes)
                f = to_fraction(current_without_prefix)  # type: ignore[arg-type]
                assert f is not None
                if value is not None:
                    if isinstance(value, str) and value.endswith("."):
                        value = str(value) + str(current)
                        continue
                    else:
                        yield output(value)  # type: ignore[arg-type]

                prefix = current[0] if has_prefix else prefix  # type: ignore[index]
                value = f.numerator if f.denominator == 1 else current_without_prefix
            elif current not in self.words:
                # non-numeric words
                if value is not None:
                    yield output(value)  # type: ignore[arg-type]
                yield output(current)  # type: ignore[arg-type]
            elif current in self.zeros:
                value = str(value or "") + "0"
            elif current in self.ones:
                ones = self.ones[current]
                if value is None:
                    value = ones
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:
                        assert value[-1] == "0"  # type: ignore[index]
                        value = value[:-1] + str(ones)  # type: ignore[index, union-attr]
                    else:
                        value = str(value) + str(ones)
                elif ones < 10:
                    if value % 10 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
                else:  # eleven to nineteen
                    if value % 100 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
            elif current in self.tens:
                tens = self.tens[current]
                if value is None:
                    value = tens
                elif isinstance(value, str):
                    value = str(value) + str(tens)
                else:
                    if value % 100 == 0:
                        value += tens
                    else:
                        value = str(value) + str(tens)
            elif current in self.hundreds:
                hundred = self.hundreds[current]
                if value is None:
                    value = hundred
                elif isinstance(value, str):
                    value = str(value) + str(hundred)
                else:
                    if value % 1000 == 0:
                        value += hundred
                    else:
                        value = str(value) + str(hundred)
            elif current in self.multipliers:
                multiplier = self.multipliers[current]
                if value is None:
                    value = multiplier
                elif isinstance(value, str) or value == 0:
                    f = to_fraction(str(value))  # type: ignore[arg-type]
                    p = f * multiplier if f is not None else None  # type: ignore[operator]
                    if f is not None and p.denominator == 1:  # type: ignore[union-attr]
                        value = p.numerator  # type: ignore[union-attr]
                    else:
                        yield output(value)  # type: ignore[arg-type]
                        value = multiplier
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
            elif current in self.ones_ordinal:
                ones = self.ones_ordinal[current]
                ordinal = True
                if value is None:
                    yield output(ones)
                elif isinstance(value, str):
                    yield output(str(value) + str(ones))
                elif ones < 10:
                    if value % 10 == 0:
                        yield output(str(value + ones))
                    else:
                        yield output(str(value) + str(ones))
                else:  # 11-19
                    if value % 100 == 0:
                        yield output(str(value + ones))
                    else:
                        yield output(str(value) + str(ones))
            elif current in self.tens_ordinal:
                tens = self.tens_ordinal[current]
                ordinal = True
                if value is None:
                    value = tens
                elif isinstance(value, str):
                    value = str(value) + str(tens)
                else:
                    if value % 100 == 0:
                        value += tens
                    else:
                        value = str(value) + str(tens)
            elif current in self.hundreds_ordinal:
                # ordinal hundred; composes with a preceding multiplier
                # ("tysiąc dziewięćsetny" -> "1900.")
                hundred = self.hundreds_ordinal[current]
                ordinal = True
                if value is None:
                    yield output(hundred)
                elif isinstance(value, str):
                    yield output(str(value) + str(hundred))
                else:
                    if value % 1000 == 0:
                        value += hundred
                    else:
                        value = str(value) + str(hundred)
            elif current in self.multipliers_ordinal:
                if value is not None:
                    yield output(value)  # type: ignore[arg-type]
                ordinal = True
                yield output(self.multipliers_ordinal[current])
            elif current in self.preceding_prefixers:
                # apply prefix (minus, plus, etc.) if it precedes a number
                if value is not None:
                    yield output(value)  # type: ignore[arg-type]
                if next in self.words or next_is_numeric:
                    prefix = self.preceding_prefixers[current]
                else:
                    yield output(current)  # type: ignore[arg-type]
            elif current in self.currencies:
                # currency word/abbreviation follows the amount -> suffix
                if value is not None:
                    yield output(str(value) + " " + self.currencies[current])
                elif current.isalpha():
                    yield output(current)  # type: ignore[arg-type]
                # else: stray currency symbol with no amount -> drop
            elif current in self.suffixers:
                # apply suffix symbols (procent -> '%')
                if value is not None:
                    yield output(str(value) + self.suffixers[current])
                else:
                    yield output(current)  # type: ignore[arg-type]
            elif current in self.specials:
                # decimal separators ("przecinek", "kropka")
                if next in self.decimals or next_is_numeric:
                    value = str(value or "") + "."
                else:
                    if value is not None:
                        yield output(value)  # type: ignore[arg-type]
                    yield output(current)  # type: ignore[arg-type]
            elif current in self.conjunctions:
                # drop "i" only when it joins two numeric tokens
                # (e.g. "sto złotych i pięćdziesiąt groszy" -> "100 zł 50 gr")
                prev_numeric = prev is not None and (
                    prev in self.words or self._NUMERIC_PREFIX_RE.match(prev or "") is not None
                )
                next_numeric = next in self.words or next_is_numeric
                if prev_numeric and next_numeric:
                    if value is not None:
                        yield output(value)  # type: ignore[arg-type]
                else:
                    if value is not None:
                        yield output(value)  # type: ignore[arg-type]
                    yield output(current)  # type: ignore[arg-type]
            else:
                raise ValueError(f"Unexpected token: {current}")

        if value is not None:
            yield output(value)

    def preprocess(self, s: str) -> str:
        # "i pół" -> "przecinek pięć" (two and a half -> 2.5) – also ASCII "pol"
        s = re.sub(r"\bi\s+(?:pół|pol)\b", "przecinek pięć", s)
        s = re.sub(r"\b(?:półtora|poltora)\b", "jeden przecinek pięć", s)
        s = re.sub(r"\b(?:półtorej|poltorej)\b", "jeden przecinek pięć", s)
        # standalone "pół" (half) -> "0.5"
        s = re.sub(r"\b(?:pół|pol)\b", "zero przecinek pięć", s)

        # normalize currency symbols to follow the amount ("€10" -> "10 €",
        # "10€" -> "10 €", "$1.50" -> "1.50 $")
        s = re.sub(r"([€$£¢])\s*(\d+(?:\.\d+)?)", r"\2 \1", s)
        s = re.sub(r"(\d+(?:\.\d+)?)\s*([€$£¢])", r"\1 \2", s)

        # put a space at number/letter boundary
        s = re.sub(r"([^\W\d_])([0-9])", r"\1 \2", s)
        s = re.sub(r"([0-9])([^\W\d_])", r"\1 \2", s)

        return s

    def postprocess(self, s: str) -> str:
        return s

    def _fraction_denominator(self, word: str) -> int | None:
        """Return the ordinal value of `word` if it is a fraction denominator."""
        for base, pos in self.lemmatizer.analyse(word):
            if pos == "adj" and base in self.ordinal_values:
                value = self.ordinal_values[base]
                if value >= 2:
                    return value
        # fallback for ASCII-folded declensions (e.g. "piate" -> "piąty")
        mapped = self._declined_ascii_map.get(word)
        if mapped is not None and mapped in self.ordinal_values:
            value = self.ordinal_values[mapped]
            if value >= 2:
                return value
        # also handle bare ASCII ordinal base itself (e.g., "piaty")
        if word in self.ordinal_values:
            value = self.ordinal_values[word]
            if value >= 2:
                return value
        return None

    def _convert_fractions(self, words: list[str]) -> list[str]:
        """Turn "jedna trzecia" -> "1/3", "trzy czwarte" -> "3/4", etc."""
        result = []
        i = 0
        n = len(words)
        while i < n:
            numerator = self.fraction_numerators.get(words[i])
            if numerator is not None and i + 1 < n:
                denominator = self._fraction_denominator(words[i + 1])
                if denominator is not None:
                    result.append(f"{numerator}/{denominator}")
                    i += 2
                    continue
            result.append(words[i])
            i += 1
        return result

    def __call__(self, s: str) -> str:
        s = self.preprocess(s)
        words = self._convert_fractions(s.split())
        words = [self._canonicalize(w) for w in words]
        s = " ".join(word for word in self.process_words(words) if word is not None)
        s = self.postprocess(s)
        return s
