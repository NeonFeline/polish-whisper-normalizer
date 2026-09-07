import pytest

from polish_whisper_normalizer import (
    PolishNumberNormalizer,
    PolishTextNormalizer,
    PolishTimeNormalizer,
)
from polish_whisper_normalizer.basic import remove_symbols_and_diacritics
from polish_whisper_normalizer.utils import (
    strip_diacritics,
    with_ascii_variants,
    with_ascii_variants_set,
)


@pytest.fixture(scope="module")
def number_norm():
    return PolishNumberNormalizer()


@pytest.fixture(scope="module")
def text_norm():
    return PolishTextNormalizer()


@pytest.fixture(scope="module")
def time_norm():
    return PolishTimeNormalizer()


# ---------------------------------------------------------------------------
# Core bug: trzysta czterdzieści osiem - diacritic vs stripped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("trzysta czterdzieści osiem", "348"),
        ("trzysta czterdziesci osiem", "348"),
        ("trzysta czterdzieści pięć", "345"),
        ("trzysta czterdziesci piec", "345"),
        ("dwieście trzydzieści siedem", "237"),
        ("dwiescie trzydziesci siedem", "237"),
        ("pięćset dwadzieścia trzy", "523"),
        ("piecset dwadziescia trzy", "523"),
        ("dziewięćset dziewięćdziesiąt dziewięć", "999"),
        ("dziewiecset dziewiecdziesiat dziewiec", "999"),
        ("sześćdziesiąt", "60"),
        ("szescdziesiat", "60"),
        ("pięćdziesiąt", "50"),
        ("piecdziesiat", "50"),
        ("osiemdziesiąt", "80"),
        ("osiemdziesiat", "80"),
        ("dziewięćdziesiąt", "90"),
        ("dziewiecdziesiat", "90"),
        ("czterdzieści dwa", "42"),
        ("czterdziesci dwa", "42"),
        ("pięć", "5"),
        ("piec", "5"),
        ("dziesięć", "10"),
        ("dziesiec", "10"),
        ("piętnaście", "15"),
        ("pietnascie", "15"),
    ],
)
def test_numbers_diacritic_vs_stripped(number_norm, text, expected):
    assert number_norm(text) == expected
    # also via text pipeline (lower + remove_symbols)
    text_norm = PolishTextNormalizer()
    assert text_norm(text) == expected
    stripped = remove_symbols_and_diacritics(text)
    assert number_norm(stripped) == expected
    assert text_norm(stripped) == expected


def test_long_mixed_number_stripped(number_norm):
    # original bug report: long number that was 300 ... piec 7
    assert number_norm("trzysta czterdzieści osiemset osiem pięć siedem") == "340800857"
    stripped = remove_symbols_and_diacritics("trzysta czterdzieści osiemset osiem pięć siedem")
    assert stripped == "trzysta czterdziesci osiemset osiem piec siedem"
    assert number_norm(stripped) == "340800857"


# ---------------------------------------------------------------------------
# Declined forms stripped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("pięciu", "5"),
        ("pieciu", "5"),
        ("dwudziestu", "20"),
        ("pięćdziesięciu", "50"),
        ("piecdziesieciu", "50"),
        ("stu", "100"),
        ("tysiąca", "1000"),
        ("tysiecy", "1000"),
        ("tysięcy", "1000"),
        ("tysiace", "1000"),
        ("tysiąca", "1000"),
        ("pięciuset", "500"),
        ("pieciuset", "500"),
        ("pierwszego", "1."),
        ("piątego", "5."),
        ("piatego", "5."),
        ("ósmej", "8."),
        ("osmej", "8."),
        ("dwudziestego", "20."),
        ("czterdziestego", "40."),
        ("czterdziestego", "40."),
        ("dwudziestej pierwszej", "21."),
        ("dwudziestej pierwszej", "21."),
    ],
)
def test_declined_stripped(number_norm, text, expected):
    assert number_norm(text) == expected
    stripped = remove_symbols_and_diacritics(text)
    assert number_norm(stripped) == expected


# ---------------------------------------------------------------------------
# Time stripped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("piąta trzydzieści", "5:30"),
        ("piata trzydziesci", "5:30"),
        ("wpół do ósmej", "7:30"),
        ("wpol do osmej", "7:30"),
        ("północ", "0:00"),
        ("polnoc", "0:00"),
        ("południe", "12:00"),
        ("poludnie", "12:00"),
        ("za piętnaście ósma", "7:45"),
        ("za pietnascie osma", "7:45"),
        ("dziesięć minut po piątej", "5:10"),
        ("dziesiec minut po piatej", "5:10"),
        ("za pięć minut ósma", "7:55"),
        ("za piec minut osma", "7:55"),
        ("na północ", "na północ"),  # geographic guard
        ("na polnoc", "na polnoc"),
        ("jest północ", "jest 0:00"),
        ("jest polnoc", "jest 0:00"),
    ],
)
def test_time_stripped(time_norm, text, expected):
    assert time_norm(text) == expected
    # also via text normalizer (lower)
    text_norm = PolishTextNormalizer()
    assert text_norm(text) == expected


# ---------------------------------------------------------------------------
# Currency / percent / half stripped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("pięć złotych", "5 zł"),
        ("piec zlotych", "5 zł"),
        ("piec złotych", "5 zł"),
        ("dwie złotówki", "2 zł"),
        ("dwie zlotowki", "2 zł"),
        ("pięć złotówek", "5 zł"),
        ("piec zlotowek", "5 zł"),
        ("pięć procentów", "5%"),
        ("piec procentow", "5%"),
        ("pół litra", "0.5 litra"),
        ("pol litra", "0.5 litra"),
        ("półtora", "1.5"),
        ("poltora", "1.5"),
        ("półtorej", "1.5"),
        ("poltorej", "1.5"),
        ("dwa i pół", "2.5"),
        ("dwa i pol", "2.5"),
        ("trzy przecinek pięć", "3.5"),
        ("trzy przecinek piec", "3.5"),
    ],
)
def test_currency_percent_half_stripped(text_norm, text, expected):
    assert text_norm(text) == expected
    stripped = remove_symbols_and_diacritics(text)
    assert text_norm(stripped) == expected


# ---------------------------------------------------------------------------
# Fractions stripped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("jedna trzecia", "1/3"),
        ("trzy piąte", "3/5"),
        ("trzy piate", "3/5"),
        ("jedna piąta", "1/5"),
        ("jedna piata", "1/5"),
        ("dwie trzecie", "2/3"),
        ("pięć szóstych", "5/6"),
        ("piec szostych", "5/6"),
    ],
)
def test_fractions_stripped(text_norm, text, expected):
    assert text_norm(text) == expected
    stripped = remove_symbols_and_diacritics(text)
    # stripped fraction should also work (piec vs pięć)
    assert text_norm(stripped) == expected


# ---------------------------------------------------------------------------
# Months stripped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("pierwszego stycznia 2023", "01.01.2023"),
        ("pierwszego styczen 2023", "01.01.2023"),
        ("pierwszego stycznia", "01.01"),
        ("piątego maja 2026", "05.05.2026"),
        ("piatego maja 2026", "05.05.2026"),
        ("trzeciego maja", "03.05"),
        ("trzeciego maja", "03.05"),
        ("siódmego września 1991", "07.09.1991"),
        ("siodmego wrzesnia 1991", "07.09.1991"),
    ],
)
def test_months_stripped(text_norm, text, expected):
    assert text_norm(text) == expected
    stripped = remove_symbols_and_diacritics(text)
    assert text_norm(stripped) == expected


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


def test_strip_diacritics():
    assert strip_diacritics("czterdzieści") == "czterdziesci"
    assert strip_diacritics("pięć") == "piec"
    assert strip_diacritics("pięćset") == "piecset"
    assert strip_diacritics("dwieście") == "dwiescie"
    assert strip_diacritics("żółć") == "zolc"
    assert strip_diacritics("Łąka") == "Laka"


def test_with_ascii_variants():
    m = {"pięć": 5, "czterdzieści": 40}
    expanded = with_ascii_variants(m)
    assert expanded["pięć"] == 5
    assert expanded["piec"] == 5
    assert expanded["czterdzieści"] == 40
    assert expanded["czterdziesci"] == 40
    # original not overwritten
    assert "piec" in expanded


def test_with_ascii_variants_set():
    s = {"pół", "przecinek"}
    expanded = with_ascii_variants_set(s)
    assert "pół" in expanded
    assert "pol" in expanded
    assert "przecinek" in expanded


# ---------------------------------------------------------------------------
# Lemmatizer via new modules
# ---------------------------------------------------------------------------


def test_lemmatizer_generate_declined_ascii_map(number_norm):
    # declined ascii map should contain stripped forms
    assert "pieciu" in number_norm._declined_ascii_map
    assert number_norm._declined_ascii_map["pieciu"] == "pięć"
    # base stripped is also in map via generate (czterdzieści -> czterdziesci)
    assert "czterdziesci" in number_norm._declined_ascii_map or "czterdziesci" in number_norm.words
    assert "piate" in number_norm._declined_ascii_map
    assert number_norm._declined_ascii_map["piate"] == "piąty"
    assert "styczen" in number_norm._declined_ascii_map


def test_imports_via_new_modules():
    # ensure split modules are importable and shim still works
    from polish_whisper_normalizer.lemmatizer import PolishLemmatizer as L1
    from polish_whisper_normalizer.numbers import PolishNumberNormalizer as N1
    from polish_whisper_normalizer.polish import PolishNumberNormalizer as N2
    from polish_whisper_normalizer.text import PolishTextNormalizer as T1
    from polish_whisper_normalizer.time import PolishTimeNormalizer as Ti1

    assert L1().analyse("dwudziestu")
    assert N1()("pięć") == "5"
    assert Ti1()("piąta trzydzieści") == "5:30"
    assert T1()("pięć") == "5"
    assert N2()("pięć") == "5"


def test_idempotency_stripped(text_norm):
    for s in [
        "trzysta czterdzieści osiem",
        "trzysta czterdziesci osiem",
        "piąta trzydzieści",
        "piata trzydziesci",
        "trzy piąte",
        "trzy piate",
        "piątego maja 2026",
        "piatego maja 2026",
    ]:
        once = text_norm(s)
        assert text_norm(once) == once
        stripped = remove_symbols_and_diacritics(s)
        once_s = text_norm(stripped)
        assert text_norm(once_s) == once_s
