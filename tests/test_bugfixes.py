"""Regression tests for polish-whisper-normalizer 0.1.12 bug report.

Covers bugs 1-8 from the WER harness findings (BIGOS v2 + Whisper large-v3):
digit times, HH.00 data loss, currency reordering, spurious fractions,
inconsistent gluing, idempotency (leading zeros, colon ordinals, currency),
pronominal "jeden", and Polish abbreviations.
"""

import pytest

from polish_whisper_normalizer import PolishTextNormalizer


@pytest.fixture(scope="module")
def normalize():
    return PolishTextNormalizer()


# --- Bug 1: digit-form clock times converge with word-form times (HIGH) ---


@pytest.mark.parametrize(
    "word_form, digit_form, expected",
    [
        ("osiemnasta trzydzieści", "18.30", "18:30"),
        ("ósma piętnaście", "8.15", "8:15"),
        ("dwudziesta czterdzieści pięć", "20.45", "20:45"),
    ],
)
def test_digit_dot_times_converge(normalize, word_form, digit_form, expected):
    assert normalize(word_form) == expected
    assert normalize(digit_form) == expected
    assert normalize(word_form) == normalize(digit_form)


def test_bare_dot_times_do_not_break_decimals_or_dates(normalize):
    # decimals (comma input, dot output) stay dotted
    assert normalize("3,14") == "3.14"
    assert normalize("3.14") == "3.14"
    # currency amounts stay decimal
    assert normalize("$1.50") == "1.50 $"
    # dates stay dotted and zero-padded
    assert normalize("05.05.2026") == "05.05.2026"
    assert normalize("21.05") == "21.05"
    assert normalize("01.01") == "01.01"


# --- Bug 2: HH.00 loses minutes (HIGH) ---


@pytest.mark.parametrize(
    "text, expected",
    [
        ("18.00", "18:00"),
        ("godzina 18.00", "18:00"),
        ("do godziny 18.00 czy będzie to możliwe", "do 18:00 czy będzie to możliwe"),
        ("12.00", "12:00"),
    ],
)
def test_hh00_preserves_minutes(normalize, text, expected):
    assert normalize(text) == expected


def test_godziny_declensions_trigger_time(normalize):
    assert normalize("do godziny 18.00") == "do 18:00"
    assert normalize("o godzinie 16.05") == "o 16:05"
    assert normalize("godz. 18.00") == "18:00"


# --- Bug 3 (+6c): currency pairs idempotent (HIGH) ---


@pytest.mark.parametrize(
    "text, expected",
    [
        ("pięć dolarów i osiemnaście centów", "5 $ 18 ¢"),
        ("5 $ 18 ¢", "5 $ 18 ¢"),
        (
            "trzynaście tysięcy czterysta pięćdziesiąt siedem dolarów i osiemnaście centów",
            "13457 $ 18 ¢",
        ),
        ("osiemnaście centów", "18 ¢"),
        ("$5", "5 $"),
        ("€10", "10 €"),
    ],
)
def test_currency_pairs_idempotent(normalize, text, expected):
    once = normalize(text)
    assert once == expected
    assert normalize(once) == once


# --- Bug 4: spurious fraction (HIGH) ---


def test_no_spurious_fraction(normalize):
    assert normalize("dwadzieścia trzy czterdzieści pięć") == "23 45"
    # true fractions still work
    assert normalize("trzy czwarte") == "3/4"
    assert normalize("jedna trzecia") == "1/3"
    assert normalize("dwie trzecie") == "2/3"


# --- Bug 5: gluing of adjacent numbers (MEDIUM) ---


@pytest.mark.parametrize(
    "text, expected",
    [
        ("siedemset jeden dwanaście", "701 12"),
        ("trzy sto", "3 100"),
        ("czterdzieści siedemset", "40 700"),
        ("dwa tysiące dwadzieścia trzy", "2023"),
        ("sto dwadzieścia trzy", "123"),
        ("jeden dwa trzy", "123"),
        ("pięć pięć pięć", "555"),
    ],
)
def test_number_gluing_consistent(normalize, text, expected):
    assert normalize(text) == expected


def test_long_digit_string_split(normalize):
    once = normalize(
        "czterdzieści siedemset jeden dwanaście dziesięć osiemset "
        "jedenaście dwieście trzydzieści sto"
    )
    # no 21-digit token; split into short numerals
    assert "70112108001120030100" not in once
    assert once == "40 701 12 10 811 230 100"
    assert normalize(once) == once


# --- Bug 6a: leading zeros idempotent (MEDIUM) ---


@pytest.mark.parametrize(
    "text, expected",
    [
        ("zero zero siedem", "7"),
        ("007", "7"),
        ("pokój zero zero pięć", "pokój 5"),
        ("minus zero pięć", "-5"),
        ("zero", "0"),
        ("00", "0"),
    ],
)
def test_leading_zeros_idempotent(normalize, text, expected):
    once = normalize(text)
    assert once == expected
    assert normalize(once) == once


def test_dates_and_times_keep_padding(normalize):
    assert normalize("05.05") == "05.05"
    assert normalize("05.05.2026") == "05.05.2026"
    assert normalize("00:00") == "0:00"
    assert normalize("07:05") == "7:05"


# --- Bug 6b: colon blocks ordinal (MEDIUM) ---


def test_colon_ordinal_idempotent(normalize):
    assert normalize("I drugi: Różne plany") == "i 2 różne plany"
    assert normalize("i drugi różne plany") == "i 2 różne plany"
    assert normalize("18:30") == "18:30"


# --- Bug 6 general idempotency ---


@pytest.mark.parametrize(
    "text",
    [
        "osiemnasta trzydzieści",
        "18.30",
        "18.00",
        "5 $ 18 ¢",
        "dwadzieścia trzy czterdzieści pięć",
        "siedemset jeden dwanaście",
        "zero zero siedem",
        "I drugi: Różne plany",
        "tak zwany",
        "tzw",
    ],
)
def test_idempotency_bugfixes(normalize, text):
    once = normalize(text)
    assert normalize(once) == once


# --- Bug 7: pronominal jeden (LOW) ---


def test_pronominal_jeden_stays_word(normalize):
    assert normalize("wszystko mi było jedno") == "wszystko mi było jedno"
    assert normalize("chciałem tylko jednego") == "chciałem tylko jednego"
    assert normalize("było jedną") == "było jedną"
    # numeral uses still convert
    assert normalize("jeden") == "1"
    assert normalize("jeden kot") == "1 kot"
    assert normalize("jedno dziecko") == "1 dziecko"


# --- Bug 8: abbreviations (MEDIUM) ---


@pytest.mark.parametrize(
    "full, short",
    [
        ("tak zwany", "tzw"),
        ("tak zwana", "tzw"),
        ("na przykład", "np"),
        ("i tak dalej", "itd"),
        ("m.in.", "między innymi"),
        ("ul. Mickiewicza", "ulica Mickiewicza"),
        ("numer 5", "nr 5"),
        ("doktor Kowalski", "dr Kowalski"),
        ("profesor Nowak", "prof. Nowak"),
    ],
)
def test_abbreviations_converge(normalize, full, short):
    assert normalize(full) == normalize(short)


def test_abbrev_with_periods(normalize):
    assert normalize("tzw.") == "tzw"
    assert normalize("np.") == "np"
    assert normalize("itd.") == "itd"
    assert normalize("tj.") == normalize("to jest")
    assert normalize("to jest złoty") == "to jest złoty"
