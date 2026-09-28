"""Regression tests for failure catalog (issue: adversarial probe).

Each case failed before the fix; all must stay idempotent.
"""

from __future__ import annotations

import pytest

from polish_whisper_normalizer import PolishTextNormalizer


@pytest.fixture(scope="module")
def normalize() -> PolishTextNormalizer:
    return PolishTextNormalizer()


@pytest.mark.parametrize(
    "text, expected",
    [
        # percent with space glues (was dropped to "5")
        ("5 %", "5%"),
        ("100 %", "100%"),
        ("5 % rabatu", "5% rabatu"),
        # vulgar fractions (were split to "1 2" / "11 2")
        ("½", "0.5"),
        ("1½", "1.5"),
        ("¼", "1/4"),
        ("¾", "3/4"),
        # spaced wpół (was "w 0.5 do 8")
        ("w pół do ósmej", "7:30"),
        ("w pol do osmej", "7:30"),
        # declined kwadrans (was unconverted)
        ("za kwadransa piąta", "4:45"),
        # zero hour (new word forms)
        ("zerowa trzydzieści", "0:30"),
        ("o zerowej", "o 0:00"),
        # cardinal "zero trzydzieści" stays two numbers (hour form required)
        ("zero trzydzieści", "0 30"),
        # o godzinie + cardinal minutes (was "o 5:00 30")
        ("o godzinie piątej trzydzieści", "o 5:30"),
        ("o godzinie szesnastej trzydzieści", "o 16:30"),
        # o + genitive hour + ordinal minutes (was "o 5 30")
        ("o piątej trzydziestej", "o 5:30"),
        # digit time with context (was kept decimal / lost minutes)
        ("o 0.00", "o 0:00"),
        ("o 5.05", "o 5:05"),
        ("o 5.30", "o 5:30"),
        ("5.30 rano", "5:30 rano"),
        # digit hour after o / with markers
        ("o 11", "o 11:00"),
        ("o 11 rano", "o 11:00 rano"),
        ("11 rano", "11:00 rano"),
        # midnight double conversion (was "12:00 w 12:00")
        ("12:00 w południe", "12:00 w południe"),
        # sentences with multiple times or digit times + noon/midnight
        ("Spotkanie jest o 8:00, a obiad w południe.", "spotkanie jest o 8:00 a obiad w 12:00"),
        ("Wyjeżdżamy o 6:00 rano i wracamy o północy.", "wyjeżdżamy o 6:00 rano i wracamy o 0:00"),
        # chained multipliers split (were collapsed, second dropped)
        ("tysiąc tysięcy", "1000 1000"),
        ("tysiąc milionów", "1000 1000000"),
        ("dwa tysiące milionów", "2000 1000000"),
        # digit dates converge (were unpadded / kept roku)
        ("5.5.2026", "05.05.2026"),
        ("1.5.2026", "01.05.2026"),
        ("5.5.2026 roku", "05.05.2026"),
        ("05.05.2026 roku", "05.05.2026"),
        ("5.5.2026 r.", "05.05.2026"),
        # decimals must NOT become dates
        ("3.14", "3.14"),
        ("2.5", "2.5"),
        ("12.05", "12.05"),
        # date ranges with MM 01-12 must stay dates, not clock times
        ("od 5.05 do 10.05", "od 5.05 do 10.05"),
        ("od 1.05 do 3.05", "od 1.05 do 3.05"),
        # time ranges with minutes 00 or >12 convert to clock times
        ("od 8.00 do 16.00", "od 8:00 do 16:00"),
        ("od 8.30 do 16.30", "od 8:30 do 16:30"),
        # math delta / difference / non-time "o <digit>" contexts must not become clock times
        ("zwiększył wynik o 5, a potem", "zwiększył wynik o 5 a potem"),
        ("pomylił się o 2.", "pomylił się o 2"),
        ("cena wzrosła o 3, nie o 4", "cena wzrosła o 3 nie o 4"),
        ("zmniejsz o 1.", "zmniejsz o 1"),
        ("chodzi o 11 osób", "chodzi o 11 osób"),
        ("różnica o 11", "różnica o 11"),
        # standalone year trailing r (was "1860 r")
        ("1860 r.", "1860"),
        ("1860 roku", "1860"),
        # sie pronoun must not become month (was "05.08")
        ("5 sie", "5 sie"),
        ("5 sierpnia", "05.08"),
        ("5 marca", "05.03"),
        # ulica instrumental (was unchanged)
        ("ulicą Mickiewicza", "ul mickiewicza"),
        # isolated declined words stay (were lemmatized)
        ("procentach", "procentach"),
        ("złotówkach", "złotówkach"),
        ("złotówkę", "złotówkę"),
        ("kupię złotówkę", "kupię złotówkę"),
        ("chciałem tylko jednego", "chciałem tylko jednego"),
        ("było jedną", "było jedną"),
        # signed zero (was "-0")
        ("minus zero", "0"),
        # quarter (was unchanged)
        ("ćwierć", "0.25"),
        ("cwierc", "0.25"),
        ("dwa i ćwierć", "2.25"),
        # hyphenated compound words with quarter/half must stay words
        ("ćwierć-finał", "ćwierćfinał"),
        ("pół-finał", "półfinał"),
        ("pół-żartem", "półżartem"),
        ("pół-serio", "półserio"),
    ],
)
def test_failure_catalog(normalize: PolishTextNormalizer, text: str, expected: str) -> None:
    once = normalize(text)
    assert once == expected
    assert normalize(once) == once


def test_strftime_invalid_fallback() -> None:
    assert PolishTextNormalizer(date_format="%Q")("piątego maja 2026") == "05.05.2026"


@pytest.mark.parametrize("bad", [None, 123, 12.5, ["test"]])
def test_non_str_raises_typeerror(normalize: PolishTextNormalizer, bad: object) -> None:
    with pytest.raises(TypeError):
        normalize(bad)  # type: ignore[arg-type]
