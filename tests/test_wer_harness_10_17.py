"""Regression tests for WER-harness failure catalog items 10-17."""

from __future__ import annotations

import pytest

from polish_whisper_normalizer import (
    BasicTextNormalizer,
    PolishNumberNormalizer,
    PolishTextNormalizer,
    PolishTimeNormalizer,
)


@pytest.fixture(scope="module")
def normalize() -> PolishTextNormalizer:
    return PolishTextNormalizer()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 10. Invisible Unicode formatting characters are stripped (MEDIUM)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Ala \u202cma kota", "ala ma kota"),
        ("\u202b\u202c", ""),
        ("a\u200bb", "ab"),
        ("s\u0142owo\u200btest", "s\u0142owotest"),
        ("a\u200bb\u200cc\u200dd\u200ee\u200ff", "abcdef"),
        ("a\u202aa\u202bb\u202cc\u202dd\u202ee", "aabcde"),
        ("a\ufeffb", "ab"),
        ("a\u2060b", "ab"),  # word joiner
        ("a\u2066b\u2069c", "abc"),  # isolates
        ("\u200e\u200f", ""),
    ],
)
def test_invisible_formatting_stripped(
    normalize: PolishTextNormalizer, text: str, expected: str
) -> None:
    assert normalize(text) == expected


def test_invisible_only_hypothesis_is_empty(normalize: PolishTextNormalizer) -> None:
    # a hypothesis made only of formatting chars must not score as a word
    assert normalize("\u2064\u2062\u2063\u2064") == ""


@pytest.mark.parametrize(
    "text, expected",
    [
        ("\u2064\u2062", ""),
        ("a\u200bb", "ab"),
    ],
)
def test_basic_invisible_formatting(text: str, expected: str) -> None:
    assert BasicTextNormalizer()(text) == expected


def test_number_time_invisible_formatting() -> None:
    assert PolishNumberNormalizer()("a\u200bb") == "ab"
    assert PolishTimeNormalizer()("pi\u200b\u0105ta trzydzie\u015bci") == "5:30"


# 11. Roman numerals are converted (MEDIUM)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("XI wiek", "11 wiek"),
        ("II kadencji", "2 kadencji"),
        ("XX wiek", "20 wiek"),
        ("III Rzesza", "3 rzesza"),
        ("jedenasty wiek", "11 wiek"),
        ("drugiej kadencji", "2 kadencji"),
    ],
)
def test_roman_numerals(normalize: PolishTextNormalizer, text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # guards: conjunction "i", title-case, lowercase, single letters stay
        ("i wiek", "i wiek"),
        ("Ci ludzie", "ci ludzie"),
        ("xi wiek", "xi wiek"),
        ("V wiek", "v wiek"),
        ("II", "ii"),
        ("w domu", "w domu"),
    ],
)
def test_roman_numeral_guards(normalize: PolishTextNormalizer, text: str, expected: str) -> None:
    assert normalize(text) == expected


# ---------------------------------------------------------------------------
# 12. Dot thousands separator keeps the thousands (HIGH)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("rs. 60.000", "rs 60000"),
        ("60.000", "60000"),
        ("1.000.000", "1000000"),
        ("sześćdziesiąt tysięcy", "60000"),
        # Polish decimal comma still means decimal, dates/decimals untouched
        ("60,000", "60"),
        ("3.14", "3.14"),
        ("05.05.2026", "05.05.2026"),
        ("temperatura 36,6", "temperatura 36.6"),
    ],
)
def test_thousands_dot(normalize: PolishTextNormalizer, text: str, expected: str) -> None:
    assert normalize(text) == expected


# ---------------------------------------------------------------------------
# 13. tys./mln/mld abbreviations are reconciled (LOW)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("50 tys.", "50000"),
        ("50 tysięcy", "50000"),
        ("50 tys", "50000"),
        ("5 mln", "5000000"),
        ("5 milionów", "5000000"),
        ("3 mld", "3000000000"),
        ("3 miliardy", "3000000000"),
        # full forms containing "tys" must not be mangled
        ("tysiąc", "1000"),
        ("tysiące", "1000"),
    ],
)
def test_magnitude_abbreviations(normalize: PolishTextNormalizer, text: str, expected: str) -> None:
    assert normalize(text) == expected


# ---------------------------------------------------------------------------
# 14. Units and percentage adjectives are reconciled (LOW)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("80 km", "80 km"),
        ("80 kilometrów", "80 km"),
        ("20-procentowy", "20%"),
        ("dwudziestoprocentowy", "20%"),
        ("20 procent", "20%"),
        ("dwadzieścia procentowy", "20%"),
        ("pięćdziesięcioprocentowy", "50%"),
        ("trzyprocentowy", "3%"),
        ("dwudziestotrzyprocentowy", "23%"),
        ("osmioprocentowy", "8%"),  # diacritic-less ASR
        # unknown stems stay words
        ("mikroprocentowy", "mikroprocentowy"),
        ("punkt procentowy", "punkt procent"),
    ],
)
def test_units_percent_adjectives(
    normalize: PolishTextNormalizer, text: str, expected: str
) -> None:
    assert normalize(text) == expected


# ---------------------------------------------------------------------------
# 15. pół stays a word inside idioms (LOW)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("pół żartem", "półżartem"),
        ("półżartem", "półżartem"),
        ("pół serio", "półserio"),
        # real halves still become numbers
        ("pół litra", "0.5 litra"),
        ("dwa i pół", "2.5"),
    ],
)
def test_pol_idioms(normalize: PolishTextNormalizer, text: str, expected: str) -> None:
    assert normalize(text) == expected


# ---------------------------------------------------------------------------
# 16. Bug 7 extension: partitive "jedna/jedną z ..." stays a word (LOW)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("jedną z nich", "jedną z nich"),
        ("jeden z nich", "jeden z nich"),
        ("jednej z nich", "jednej z nich"),
        ("jedno z nich", "jedno z nich"),
        ("chciałem tylko jednego", "chciałem tylko jednego"),
        # numeral + accusative noun still converts (consistent with drugą -> 2)
        ("jedną unię", "1 unię"),
        ("jedną książkę", "1 książkę"),
        ("drugą", "2"),
    ],
)
def test_jedna_partitive(normalize: PolishTextNormalizer, text: str, expected: str) -> None:
    assert normalize(text) == expected


# ---------------------------------------------------------------------------
# Perf prefilter: fractions still convert, non-fractions skip parsing (LOW)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("jedna druga", "0.5"),
        ("jednej trzeciej", "1/3"),
        ("dwóch trzecich", "2/3"),
        ("trzy czwarte", "3/4"),
        ("dwie trzecie", "2/3"),
        ("zero trzecich", "0/3"),
        # guards still hold (no false fractions / ordinal compounds intact)
        ("jeden drugi", "1 2"),
        ("ani jeden ani drugi", "ani 1 ani 2"),
        ("sto dwudziesty", "120"),
        # plain sentences pass through untouched
        ("kot ma kota", "kot ma kota"),
        ("pięć zerowych", "5 zerowych"),
    ],
)
def test_fractions_prefilter(normalize: PolishTextNormalizer, text: str, expected: str) -> None:
    assert normalize(text) == expected


# ---------------------------------------------------------------------------
# 17. Hyphens are handled consistently (LOW)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("omega-3", "omega 3"),
        ("omega 3", "omega 3"),
        ("covid-19", "covid 19"),
        ("lata 70-te", "lata 70"),
        ("70-latek", "70 latek"),
        ("70 latek", "70 latek"),
        ("70-tych", "70"),
        # leading minus for negatives is kept
        ("minus dziesięć", "-10"),
        ("5-10", "5-10"),
    ],
)
def test_hyphens(normalize: PolishTextNormalizer, text: str, expected: str) -> None:
    assert normalize(text) == expected
