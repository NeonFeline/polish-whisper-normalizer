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
from .utils import (
    contains_digit,
    normalize_vulgar_fractions,
    strip_diacritics,
    with_ascii_variants,
    with_ascii_variants_set,
)

logger = logging.getLogger(__name__)


# masculine nominative singular ordinals (2-10) are never fraction
# denominators in Polish ("jeden drugi" must not become 1/2). Shared by
# _fraction_denominator and the denominator prefilter (perf) so the two
# cannot drift apart.
_MASC_NOM_SING_ORDINALS: frozenset[str] = frozenset(
    {
        "drugi",
        "trzeci",
        "czwarty",
        "piąty",
        "szósty",
        "siódmy",
        "ósmy",
        "dziewiąty",
        "dziesiąty",
        "piaty",
        "szosty",
        "siodmy",
        "osmy",
        "dziewiaty",
        "dziesiaty",
    }
)


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
    # all possible fraction-denominator surfaces (perf prefilter, built once)
    _DENOM_SURFACE_CACHE: frozenset[str] | None = None
    _DENOM_LOCK = threading.Lock()

    # pre-compiled patterns (avoid recompiling per token)
    _POLISH_WORD_RE = re.compile(r"[a-ząćęłńóśźż]+")
    # invisible formatting chars (zero-width, bidi) are dropped (10)
    _FORMAT_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]+")
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
            "ośmnaście": 18,
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
            # abbreviated months (Morfeusz-independent, common in refs)
            # NOTE: "sie" (sierpień) removed – collides with the reflexive
            # pronoun "się"/"sie" (diacritic-less ASR), causing false dates
            # ("5 sie" pronoun -> "05.08"). Use full "sierpień"/"sierpnia".
            "sty": "1",
            "lut": "2",
            "mar": "3",
            "kwi": "4",
            "cze": "6",
            "lip": "7",
            "sier": "8",
            "wrz": "9",
            "paź": "10",
            "lis": "11",
            "gru": "12",
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
        self._denom_surfaces = self._get_or_build_denom_surfaces(
            self.ordinal_values, self.lemmatizer
        )

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

    @classmethod
    def _get_or_build_denom_surfaces(
        cls, ordinal_values: dict[str, int], lemmatizer: PolishLemmatizer
    ) -> frozenset[str]:
        """Return cached set of possible fraction-denominator surfaces.

        Built once via Morfeusz.generate over ordinal bases (value >= 2),
        plus ASCII folds and the bare bases themselves (e.g. "setny",
        "dwudziesty", ASCII "piaty"-style forms). Mirrors
        _fraction_denominator exactly: masculine nominative singulars
        (2-10) can never be denominators and are excluded.
        """
        if cls._DENOM_SURFACE_CACHE is not None:
            return cls._DENOM_SURFACE_CACHE
        surfaces: set[str] = set()
        for base, value in ordinal_values.items():
            if value < 2:
                continue
            # Bare masculine nominative singulars (2-10) are never
            # denominators, but their declined forms may be (e.g. "piatego"
            # via the ASCII fallback) – so generate everything, only the
            # bare excluded forms stay out of the set.
            if base not in _MASC_NOM_SING_ORDINALS:
                surfaces.add(base)
            if lemmatizer.morf is not None:
                try:
                    forms = lemmatizer.generate(base)
                except Exception:
                    logger.debug("generate failed for %r", base)
                    continue
                for surface, lemma, _tag in forms:
                    if lemma.split(":")[0] != base:
                        continue
                    surfaces.add(surface)
                    folded = strip_diacritics(surface)
                    if folded != surface:
                        surfaces.add(folded)
        with cls._DENOM_LOCK:
            if cls._DENOM_SURFACE_CACHE is None:
                cls._DENOM_SURFACE_CACHE = frozenset(surfaces)
        return cls._DENOM_SURFACE_CACHE

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
            elif pos == "adj" and base in self.cardinal_lemmas:
                # Polish numerals like "jeden" are tagged as adj in some forms (e.g. "jednej")
                candidates["num"] = base
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
        if len(self._canon_cache) > 20000:
            for k in list(self._canon_cache.keys())[:10000]:
                del self._canon_cache[k]
        self._canon_cache[word] = result
        return result

    def process_words(self, words: list[str], orig_words: list[str] | None = None) -> Iterator[str]:
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

        padded: list[str | None] = [None, *words, None]
        orig_padded: list[str | None] = (
            [None, *orig_words, None] if orig_words is not None else padded
        )
        for idx in range(len(padded) - 2):
            prev, current, next = padded[idx], padded[idx + 1], padded[idx + 2]
            orig_current = orig_padded[idx + 1]
            if orig_current is None:
                orig_current = current
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
                if current == "piec":
                    # "piec" collides with verb (to bake) and noun (oven).
                    # Only treat as 5 in a numeric context or when single-word utterance.
                    is_numeric_ctx = (
                        (
                            prev is not None
                            and (
                                prev in self.words
                                or self._NUMERIC_PREFIX_RE.match(prev) is not None
                            )
                        )
                        or (next is not None and (next in self.words or next_is_numeric))
                        or len(words) == 1
                    )
                    if not is_numeric_ctx:
                        if value is not None:
                            yield output(value)  # type: ignore[arg-type]
                        yield output(current)  # type: ignore[arg-type]
                        continue
                # Pronominal "jeden" (not numeral): "wszystko mi było jedno",
                # "chciałem tylko jednego" stays a word (bug 7, like
                # "Maja"/"na północ" guards). Partitive "jeden/jedną/jedno z …"
                # ("one of them") also stays a word (16). Preserve original
                # inflection ("jedną" stays "jedną", not lemmatized to "jeden").
                if current in {"jeden", "jedna", "jedno"} and (
                    prev
                    in {
                        "było",
                        "bylo",
                        "była",
                        "byla",
                        "jest",
                        "tylko",
                        "mi",
                    }
                    or next == "z"
                ):
                    if value is not None:
                        yield output(value)  # type: ignore[arg-type]
                    yield output(orig_current)  # type: ignore[arg-type]
                    continue
                ones = self.ones[current]
                if value is None:
                    value = ones
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:
                        assert value[-1] == "0"  # type: ignore[index]
                        value = value[:-1] + str(ones)  # type: ignore[index, union-attr]
                    elif isinstance(value, str) and "." in value:
                        # decimal fraction after "przecinek/kropka"
                        # ("3."+"14" -> "3.14", "0."+"5" -> "0.5",
                        # "3.1"+"4" -> "3.14"): always glue
                        value = str(value) + str(ones)
                    elif ones < 10:
                        # digit dictation ("jeden dwa trzy" -> "123") only for
                        # single digits with small values; larger chunks start
                        # new numerals (bug 5: "701"+"12" -> "701 12").
                        try:
                            iv = int(value)  # type: ignore[arg-type]
                        except (ValueError, TypeError):
                            iv = None
                        if iv is not None and iv < 100:
                            value = str(value) + str(ones)
                        else:
                            yield output(value)  # type: ignore[arg-type]
                            value = ones
                    else:
                        # teens (10-19) never glue as dictation; separate
                        yield output(value)  # type: ignore[arg-type]
                        value = ones
                elif ones < 10:
                    if value % 10 == 0:
                        value += ones
                    elif value < 100:  # type: ignore[operator]
                        value = str(value) + str(ones)
                    else:
                        yield output(value)  # type: ignore[arg-type]
                        value = ones
                else:  # eleven to nineteen
                    if value % 100 == 0:
                        value += ones
                    else:
                        yield output(value)  # type: ignore[arg-type]
                        value = ones
            elif current in self.tens:
                tens = self.tens[current]
                if value is None:
                    value = tens
                elif isinstance(value, str):
                    if "." in value:
                        # decimal ("0."+"20" -> "0.20"): glue
                        value = str(value) + str(tens)
                    else:
                        # dictation string + tens starts a new numeral
                        # (bug 5: avoid "12"+"40" -> "1240")
                        yield output(value)  # type: ignore[arg-type]
                        value = tens
                elif value % 100 == 0:
                    value += tens
                else:
                    # invalid magnitude order ("dziesięć dwadzieścia") -> separate numeral
                    yield output(value)  # type: ignore[arg-type]
                    value = tens
            elif current in self.hundreds:
                hundred = self.hundreds[current]
                if value is None:
                    value = hundred
                elif isinstance(value, str):
                    if "." in value:
                        value = str(value) + str(hundred)
                    else:
                        # dictation string + hundreds starts a new numeral
                        yield output(value)  # type: ignore[arg-type]
                        value = hundred
                elif value % 1000 == 0:
                    value += hundred
                elif value >= 100:
                    # invalid order with large value ("340"+"800") -> separate
                    # (bug 5: avoid 21-digit tokens; "340 800" not "340800")
                    yield output(value)  # type: ignore[arg-type]
                    value = hundred
                else:
                    # invalid magnitude order ("dziesięć pięćset") -> separate numeral
                    yield output(value)  # type: ignore[arg-type]
                    value = hundred
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
                    if residual == 0:
                        # bare multiplier after a complete thousand-group
                        # ("tysiąc tysięcy", "dwa miliony tysiąc"): do not
                        # drop the second multiplier (old code computed
                        # before+0*M == before). Split into separate numerals
                        # to preserve information and avoid inventing huge
                        # tokens (bug 5 philosophy).
                        yield output(value)  # type: ignore[arg-type]
                        value = multiplier
                    else:
                        value = before + residual * multiplier
            elif current in self.ones_ordinal:
                ones = self.ones_ordinal[current]
                if value is None:
                    ordinal = True
                    yield output(ones)
                elif isinstance(value, str):
                    ordinal = True
                    yield output(str(value) + str(ones))
                elif ones < 10:
                    if value % 10 == 0 and value >= 10:
                        ordinal = True
                        yield output(str(value + ones))
                    else:
                        yield output(value)
                        ordinal = True
                        yield output(ones)
                else:  # 11-19
                    if value % 100 == 0 and value >= 100:
                        ordinal = True
                        yield output(str(value + ones))
                    else:
                        yield output(value)
                        ordinal = True
                        yield output(ones)
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
                    # standalone currency without amount: keep original
                    # inflection ("złotówkę" stays, "procentach" stays) instead
                    # of lemmatized artifact.
                    yield output(orig_current)  # type: ignore[arg-type]
                # else: stray currency symbol with no amount -> drop
            elif current in self.suffixers:
                # apply suffix symbols (procent -> '%')
                if value is not None:
                    yield output(str(value) + self.suffixers[current])
                else:
                    yield output(orig_current)  # type: ignore[arg-type]
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
        s = self._FORMAT_RE.sub("", s)  # invisible formatting chars (10)
        s = normalize_vulgar_fractions(s)
        # perf: guards are necessary conditions — skipping never changes output
        # join hyphenated compound words with prefixes "pół-", "ćwierć-":
        # "pół-finał" -> "półfinał", "ćwierć-finał" -> "ćwierćfinał", "pół-żartem" -> "półżartem"
        if "-" in s:
            s = re.sub(r"\b(pół|pol|ćwierć|cwierc)-\s*([a-ząćęłńóśźż]+)\b", r"\1\2", s)
        # ellipsis marks a sentence boundary: keep the boundary (so adjacent
        # spelled-out numbers are not merged) but drop the "." later
        if "." in s or "…" in s:
            s = re.sub(r"(?:\s*\.){2,}\s*|\s*…\s*", " . ", s)
        # "i pół" -> "przecinek pięć" (two and a half -> 2.5) – also ASCII "pol"
        if "ół" in s or "pol" in s:
            s = re.sub(r"\bi\s+(?:pół|pol)\b(?!-)", "przecinek pięć", s)
        if "półtora" in s or "poltora" in s:
            s = re.sub(r"\b(?:półtora|poltora)\b(?!-[a-ząćęłńóśźż])", "jeden przecinek pięć", s)
        if "półtorej" in s or "poltorej" in s:
            s = re.sub(r"\b(?:półtorej|poltorej)\b(?!-[a-ząćęłńóśźż])", "jeden przecinek pięć", s)
        # standalone "pół" (half) -> "0.5"
        # Protect hyphenated compound words (e.g. "pół-finał", "pół-żartem", "pół-serio")
        if "ół" in s or "pol" in s:
            s = re.sub(r"(?<!-)\b(?:pół|pol)\b(?!-[a-ząćęłńóśźż])", "zero przecinek pięć", s)
        # quarter: "ćwierć" (one quarter) -> "0.25", "i ćwierć" -> "+0.25"
        # (ASCII "cwierc" for diacritic-less ASR). Only singular nominative;
        # plural "ćwierci" stays a word (like "zerowych").
        # Protect hyphenated compounds (e.g. "ćwierć-finał")
        if "ćwierć" in s or "cwierc" in s:
            s = re.sub(r"\bi\s+(?:ćwierć|cwierc)\b(?!-)", "przecinek dwa pięć", s)
            s = re.sub(
                r"(?<!-)\b(?:ćwierć|cwierc)\b(?!-[a-ząćęłńóśźż])", "zero przecinek dwa pięć", s
            )

        # normalize currency symbols to follow the amount ("€10" -> "10 €",
        # "10€" -> "10 €", "$1.50" -> "1.50 $").
        # Only when the symbol is a true prefix (not already a suffix of a
        # previous amount): "5 $ 18 ¢" must NOT become "5 18 $ ¢" (bug 3).
        def _currency_prefix_repl(m: re.Match[str]) -> str:
            prev = m.string[: m.start()].rstrip()[-1:] if m.string[: m.start()].rstrip() else ""
            if prev.isdigit():
                return m.group(0)
            return f"{m.group(2)} {m.group(1)}"

        if "€" in s or "$" in s or "£" in s or "¢" in s:
            s = re.sub(r"([€$£¢])\s*(\d+(?:\.\d+)?)", _currency_prefix_repl, s)
            s = re.sub(r"(\d+(?:\.\d+)?)\s*([€$£¢])", r"\1 \2", s)

        # put a space at number/letter boundary
        if contains_digit(s):
            s = re.sub(r"([^\W\d_])([0-9])", r"\1 \2", s)
            s = re.sub(r"([0-9])([^\W\d_])", r"\1 \2", s)

        return s

    def postprocess(self, s: str) -> str:
        # Normalize decimal fractions expressed as "21 i 5/10" (from
        # "dwadzieścia jeden i pięć dziesiątych") to decimal "21.5"
        # for equivalence with "21.5" reference (WER on jednostki).
        # Only decimal denominators (10,100,1000) and half (1/2) for 0.5==1/2.
        def _is_decimal_den(den: int) -> bool:
            s_den = str(den)
            return s_den[0] == "1" and all(c == "0" for c in s_den[1:])

        def _fraction_to_decimal_str(num: int, den: int, int_part: str | None = None) -> str | None:
            if num >= den or num == 0:
                return None
            # decimal denominators (10,100,1000) or half 1/2
            is_decimal = _is_decimal_den(den)
            is_half = num == 1 and den == 2
            if not (is_decimal or is_half):
                return None
            if is_half:
                # 1/2 -> 0.5, 21 i 1/2 -> 21.5
                return f"{int_part}.5" if int_part is not None else "0.5"
            # power-of-10 decimal
            k = len(str(den)) - 1
            frac_str = f"{num:0{k}d}"
            dec = f"{int_part}.{frac_str}" if int_part is not None else f"0.{frac_str}"
            dec = dec.rstrip("0").rstrip(".")
            if dec.endswith("."):
                dec += "0"
            if "." not in dec:
                dec = f"{int_part if int_part is not None else '0'}.0"
            return dec

        def _decimal_repl(m: re.Match[str]) -> str:
            int_part = m.group(1)
            num = int(m.group(2))
            den = int(m.group(3))
            dec = _fraction_to_decimal_str(num, den, int_part)
            if dec is None:
                return m.group(0)
            return dec

        s = re.sub(r"\b(\d+)\s+i\s+(\d+)/(\d+)\b", _decimal_repl, s) if "/" in s else s

        def _standalone_frac_repl(m: re.Match[str]) -> str:
            num = int(m.group(1))
            den = int(m.group(2))
            dec = _fraction_to_decimal_str(num, den, None)
            if dec is None:
                return m.group(0)
            return dec

        if "/" in s:
            s = re.sub(r"\b(\d+)/(\d+)\b", _standalone_frac_repl, s)
        # Normalize signed zero ("-0", "+0", "-0.0" -> "0"): negative zero
        # equals zero and otherwise breaks WER convergence ("minus zero").
        if "-0" in s or "+0" in s:
            s = re.sub(r"(?<![\d.:/-])([+-])0+(?:\.0+)?(?![\d.:/-])", "0", s)
        # Strip leading zeros for idempotency ("007" -> "7", "005" -> "5"):
        # word-form "zero zero siedem" glues to "007" on pass 1 but digit
        # "007" goes via Fraction to 7, so N(N(x)) != N(x) (bug 6a).
        # Preserve dates (05.05, 05.05.2026, 2026-05-05, 5/5/2026) and times
        # (0:00, 12:00): no stripping adjacent to ".", ":", "-", "/".
        if "0" in s and contains_digit(s):
            s = re.sub(r"(?<![\d.:/-])([+-]?)0+(\d)(?![\d.:/-])", r"\1\2", s)
        # Collapse "00" alone -> "0" (e.g. "00" from "zero zero").
        if "00" in s:
            s = re.sub(r"(?<![\d.:/-])([+-]?)0{2,}(?![\d.:/-])", r"\g<1>0", s)
        # Normalize time hours ("00:00" -> "0:00", "07:05" -> "7:05")
        # to converge with word-form times ("0:00", "7:05"); minutes stay padded.
        if "0" in s and ":" in s:
            s = re.sub(r"\b0+(\d+:\d{2})\b", r"\1", s)
        return s

    def _fraction_denominator(self, word: str) -> int | None:
        """Return the ordinal value of `word` if it is a fraction denominator."""
        # Cardinals are never denominators: "czterdzieści" (40, cardinal) must
        # not become denominator 40 via its adj analysis "czterdziesty"
        # (bug 4: "dwadzieścia trzy czterdzieści pięć" -> "23/40 5").
        if (
            word in self.ones
            or word in self.tens
            or word in self.hundreds
            or word in self.zeros
            or word in self.fraction_numerators
        ):
            return None
        # In Polish, fraction denominators are feminine or plural forms
        # (e.g. "druga", "trzecia", "czwarta", "trzecie", "czwarte", "piątych", "setnych").
        # Masculine nominative singular forms ("drugi", "trzeci", "czwarty", "piąty")
        # are never fraction denominators in Polish and must not collide (e.g. "jeden drugi").
        if word in _MASC_NOM_SING_ORDINALS:
            return None
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

    def _fraction_numerator_value(self, word: str) -> int | None:
        """Return cardinal value for fraction numerator via Morfeusz.

        Supports declined forms (e.g. "jednej" -> 1, "dwóch" -> 2) and
        broader range (1-19, 20-90, 100-900) via Morfeusz lemmatization.
        Hundreds are included for decimal fractions like
        "czterysta pięćdziesiąt sześć tysięcznych" -> 456/1000.
        Collision with ordinal compounds (e.g. "sto dwudziesty" -> 100/20)
        is avoided by numerator < denominator check in caller.
        """
        # Fractions with value 1 in Polish are always feminine ("jedna", "jednej"),
        # never masculine ("jeden") or neuter ("jedno"), preventing false
        # fractions from counting sequences ("jeden drugi").
        if word in {"jeden", "jedno"}:
            return None
        # direct feminine dict (fast path, includes ASCII variants)
        if word in self.fraction_numerators:
            return self.fraction_numerators[word]
        # direct cardinal maps (covers 1-19, tens, hundreds, zero + ASCII)
        if word in self.ones:
            return self.ones[word]
        if word in self.tens:
            return self.tens[word]
        if word in self.hundreds:
            return self.hundreds[word]
        if word in self.zeros:
            return 0
        # via Morfeusz – handle declensions like "jednej" -> "jeden" (adj)
        for base, _pos in self.lemmatizer.analyse(word):
            if base in self.fraction_numerators:
                return self.fraction_numerators[base]
            if base in self.ones:
                return self.ones[base]
            if base in self.tens:
                return self.tens[base]
            if base in self.hundreds:
                return self.hundreds[base]
            if base in self.zeros:
                return 0
        # fallback diacritic-less declined map (e.g. "pieciu" -> "pięć")
        mapped = self._declined_ascii_map.get(word)
        if mapped is not None:
            if mapped in self.fraction_numerators:
                return self.fraction_numerators[mapped]
            if mapped in self.ones:
                return self.ones[mapped]
            if mapped in self.tens:
                return self.tens[mapped]
            if mapped in self.hundreds:
                return self.hundreds[mapped]
            if mapped in self.zeros:
                return 0
        return None

    def _parse_fraction_numerator(self, words: list[str], start: int) -> tuple[int, int] | None:
        """Parse 1-3 word cardinal numerator at words[start:].

        Handles "dwadzieścia trzy" -> 23, "czterysta pięćdziesiąt sześć" -> 456
        for fractions. Returns (value, length) or None.
        """
        n = len(words)
        # try 3-word hundreds + tens + ones (e.g. "czterysta pięćdziesiąt sześć" -> 456)
        if start + 2 < n:
            v1 = self._fraction_numerator_value(words[start])
            v2 = self._fraction_numerator_value(words[start + 1])
            v3 = self._fraction_numerator_value(words[start + 2])
            if v1 is not None and v2 is not None and v3 is not None:
                # hundreds 100-900 + tens 10-90 + ones 1-9
                if v1 in {100, 200, 300, 400, 500, 600, 700, 800, 900}:
                    if (
                        v2
                        in {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90}
                        and 1 <= v3 <= 9
                    ):
                        # need tens+ones distinct: e.g. 400+50+6, but also 100+20+3
                        # also handle 100+11+? but 11 already includes ones, so avoid double
                        if v2 < 20 and v3 < 10:
                            # v2 is 10-19, v3 would be extra ones -> invalid (e.g. 10 + 1)
                            pass
                        else:
                            return v1 + v2 + v3, 3
                    # hundreds + tens (e.g. "czterysta pięćdziesiąt" -> 450)
                    if v2 in {
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                        20,
                        30,
                        40,
                        50,
                        60,
                        70,
                        80,
                        90,
                    } and v3 in {1, 2, 3, 4, 5, 6, 7, 8, 9}:
                        # already handled above as 3-word, but we try 3-word only if both
                        pass
                # also try hundreds + ones (e.g. "sto pięć" -> 105)
                if (
                    v1 in {100, 200, 300, 400, 500, 600, 700, 800, 900}
                    and 1 <= v3 <= 9
                    and v2 in {1, 2, 3, 4, 5, 6, 7, 8, 9}
                ):
                    # this is actually 2-word hundreds+ones, handled below, but 3-word with middle tens missing
                    pass
        # try 2-word
        if start + 1 < n:
            v1 = self._fraction_numerator_value(words[start])
            v2 = self._fraction_numerator_value(words[start + 1])
            if v1 is not None and v2 is not None:
                # tens 20-90 + ones 1-9
                if v1 in {20, 30, 40, 50, 60, 70, 80, 90} and 1 <= v2 <= 9:
                    return v1 + v2, 2
                # hundreds 100-900 + tens 10-90 or ones 1-9 or teens 10-19
                if v1 in {100, 200, 300, 400, 500, 600, 700, 800, 900} and v2 in {
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    30,
                    40,
                    50,
                    60,
                    70,
                    80,
                    90,
                }:
                    return v1 + v2, 2
                # hundreds + ones with tens missing already covered above, but also try 100+5
                # tens + ones already handled, also try teens + ones? not needed
        # try 3-word again for hundreds+tens+ones where we missed: do generic sum check
        if start + 2 < n:
            vals = [self._fraction_numerator_value(words[start + i]) for i in range(3)]
            if all(v is not None for v in vals):  # type: ignore[arg-type]
                v1, v2, v3 = vals  # type: ignore[assignment]
                # allow sum if values are descending magnitude and <1000
                # e.g. 400+50+6, 100+20+3, 200+11+? but 11 includes ones
                # simple check: v1 is hundreds, v2 is tens/10-19, v3 is ones, and v1>v2>v3
                if (
                    v1 in {100, 200, 300, 400, 500, 600, 700, 800, 900}
                    and v2
                    in {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90}
                    and v3 in {1, 2, 3, 4, 5, 6, 7, 8, 9}
                    and (v1 > v2 > v3 or (v1 > v2 and v2 >= 10 and v3 < 10))
                    and not (10 <= v2 <= 19 and v3 < 10)
                ):
                    return v1 + v2 + v3, 3
                # also 100+20+3 case already, but ensure
        # single word
        v = self._fraction_numerator_value(words[start])
        if v is not None:
            return v, 1
        return None

    def _convert_fractions(self, words: list[str]) -> list[str]:
        """Turn "jedna trzecia" -> "1/3", "trzy czwarte" -> "3/4", etc.

        Uses Morfeusz for declined numerators (e.g. "jednej trzeciej" -> "1/3",
        "dwóch trzecich" -> "2/3") and supports broader range (11-19) and
        multi-word numerators like "dwadzieścia trzy setne" -> "23/100".
        Requires numerator < denominator to avoid colliding with ordinal
        compounds like "sto dwudziesty" -> 120 (not 100/20).

        Perf: a fraction needs a denominator surface, so inputs without one
        skip the per-position parsing (the common case).
        """
        if len(words) < 2 or not any(w in self._denom_surfaces for w in words):
            return list(words)
        result: list[str] = []
        i = 0
        n = len(words)
        while i < n:
            parsed = self._parse_fraction_numerator(words, i)
            if parsed is not None and i + parsed[1] < n:
                numerator, length = parsed
                denominator = self._fraction_denominator(words[i + length])
                if denominator is not None and numerator < denominator:
                    result.append(f"{numerator}/{denominator}")
                    i += length + 1
                    continue
            result.append(words[i])
            i += 1
        return result

    def __call__(self, s: str) -> str:
        if not isinstance(s, str):
            raise TypeError(f"Expected str, got {type(s).__name__}")
        s = self.preprocess(s)
        frac_words = self._convert_fractions(s.split())
        canon_words = [self._canonicalize(w) for w in frac_words]
        s = " ".join(
            word
            for word in self.process_words(canon_words, frac_words)
            if word is not None and word != "."
        )
        s = self.postprocess(s)
        return s
