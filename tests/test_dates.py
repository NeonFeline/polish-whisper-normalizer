import pytest
from polish_whisper_normalizer import PolishTextNormalizer


@pytest.fixture(scope="module")
def normalize():
    return PolishTextNormalizer()


@pytest.mark.parametrize(
    "text, expected",
    [
        # basic day+month -> conditional month (no year, stays spaced, not DD.MM.)
        ("pierwszego stycznia", "1. 1"),
        ("trzeciego maja", "3. 5"),
        ("piątego maja", "5. 5"),
        ("dwudziestego pierwszego maja", "21. 5"),
        ("dziewiętnastego grudnia", "19. 12"),
        # isolated month stays word
        ("maja", "maja"),
        ("w maju", "w maju"),
        ("styczeń", "styczeń"),
        # full date: day month year (spoken) -> DD.MM.YYYY / DD.MM.YYYYr.
        ("piątego maja roku dwa tysiące dwudziestego szóstego", "05.05.2026"),
        ("piątego maja dwa tysiące dwudziestego szóstego roku", "05.05.2026"),
        ("piątego maja dwa tysiące dwudziestego szóstego", "05.05.2026"),
        ("piątego maja 2026", "05.05.2026"),
        ("piątego maja 2026 roku", "05.05.2026"),
        ("pierwszego stycznia dwa tysiące dwudziestego trzeciego roku", "01.01.2023"),
        ("pierwszego stycznia 2023", "01.01.2023"),
        # zero-padding checks
        ("pierwszego stycznia 2023 roku", "01.01.2023"),
        ("drugiego lutego 2020", "02.02.2020"),
        ("trzeciego marca 1999 roku", "03.03.1999"),
        ("trzydziestego pierwszego grudnia 1999 roku", "31.12.1999"),
        ("dwudziestego dziewiątego lutego dwa tysiące dwudziestego roku", "29.02.2020"),
        # month declensions (genitive) + year
        ("siódmego września tysiąc dziewięćset dziewięćdziesiątego pierwszego roku", "07.09.1991"),
        ("jedenastego listopada 1918 roku", "11.11.1918"),
        # arabic day without dot should NOT trigger month conversion (by design)
        ("5 maja 2026", "5 maja 2026"),
        ("1 stycznia 2023", "1 stycznia 2023"),
        # already normalized full date stays idempotent
        ("05.05.2026", "05.05.2026"),
        ("05.05.2026", "05.05.2026"),
        ("01.01.2023", "01.01.2023"),
        # day month without year stays spaced (not DD.MM.) – existing contract
        ("1. stycznia", "1. 1"),
        ("12. grudnia", "12. 12"),
        # year as cardinal vs ordinal
        ("pierwszego stycznia dwa tysiące dwadzieścia trzy", "01.01.2023"),
        # with "roku" word before year vs after
        ("piątego maja roku 2026", "05.05.2026"),
        ("piątego maja 2026 roku", "05.05.2026"),
    ],
)
def test_dates(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "piątego maja roku dwa tysiące dwudziestego szóstego",
        "05.05.2026",
        "05.05.2026",
        "pierwszego stycznia",
        "1. 1",
        "3. 5",
    ],
)
def test_dates_idempotent(normalize, text):
    once = normalize(text)
    assert normalize(once) == once
