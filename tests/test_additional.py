import pytest

from polish_whisper_normalizer import PolishTextNormalizer


@pytest.fixture(scope="module")
def normalize():
    return PolishTextNormalizer()


# --- original 9 fixes coverage, extended ---


@pytest.mark.parametrize(
    "text,expected",
    [
        ("o piątej", "o 5:00"),
        ("o piątej rano", "o 5:00 rano"),
        ("piąta rano", "5:00 rano"),
        ("szósta rano", "6:00 rano"),
        ("o drugiej stronie", "o 2 stronie"),  # no false positive
        ("o piątej stronie", "o 5 stronie"),
        ("piąta rocznica", "5 rocznica"),  # not time
    ],
)
def test_standalone_o_time_extended(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("dziesięć minut po piątej", "5:10"),
        ("pięć minut po piątej", "5:05"),
        ("za dwadzieścia minut ósma", "7:40"),
        ("za pięć minut ósma", "7:55"),
        ("za dwie minuty ósma", "7:58"),
        ("jedna minuta po piątej", "5:01"),
        ("za pięć minut", "za 5 minut"),  # duration, not time
        ("pięć minut", "5 minut"),
    ],
)
def test_time_minut_extended(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("od piątej do szóstej", "od 5:00 do 6:00"),
        ("od ósmej do dziesiątej", "od 8:00 do 10:00"),
        ("od piątego do szóstego", "od 5 do 6"),  # ordinal, not time
    ],
)
def test_time_ranges_extended(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("pięć procentów", "5%"),
        ("dwudziestu procentów", "20%"),
        ("pięćdziesiąt procenty", "50%"),
        ("pięć procenta", "5%"),
        ("sto procent", "100%"),
    ],
)
def test_percent_declined_extended(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("pięć złotówek", "5 zł"),
        ("dwie złotówki", "2 zł"),
        ("sto złotówek", "100 zł"),
        ("jedna złotówka", "1 zł"),
        ("złotówkę", "złotówka"),  # standalone without number -> lemmatized artifact
        ("kupię złotówkę", "kupię złotówka"),
    ],
)
def test_currency_zlotowka_extended(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("pół litra", "0.5 litra"),
        ("pół godziny", "0.5 godziny"),
        ("pół kilo", "0.5 kilo"),
        ("półtora", "1.5"),
        ("północy", "północy"),  # not half
        ("półmetek", "półmetek"),
        ("wpół do ósmej", "7:30"),
        ("północ", "0:00"),
    ],
)
def test_half_extended(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("tysiąc dziewięćsetny", "1900"),
        ("tysiąc dziewięćsetny rok", "1900 rok"),
        ("tysiąc dwusetny", "1200"),
        ("dziewięćsetny", "900"),
        ("tysiąc dziewięćset dziewięćdziesiąty dziewiąty", "1999"),
    ],
)
def test_ordinal_multiplier_extended(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("jedna trzecia", "1/3"),
        ("trzy czwarte", "3/4"),
        ("dwie trzecie", "2/3"),
        ("jedna druga", "0.5"),  # 1/2 == 0.5 for WER (decimal)
        ("trzecia osoba", "3 osoba"),  # no false positive
        ("druga strona", "2 strona"),
    ],
)
def test_fractions_extended(normalize, text, expected):
    assert normalize(text) == expected


# --- months: conditional (only after day ordinal) ---


@pytest.mark.parametrize(
    "text,expected",
    [
        # isolated months stay words (conditional mapping prevents name collision)
        ("stycznia", "stycznia"),
        ("maja", "maja"),
        ("grudnia", "grudnia"),
        ("styczeń", "styczeń"),
        ("maj", "maj"),
        ("grudzień", "grudzień"),
        ("w grudniu", "w grudniu"),
        ("w maju", "w maju"),
        # with day ordinal they become numbers
        ("01.01", "01.01"),
        ("1. maja", "01.05"),
        ("12.12", "12.12"),
    ],
)
def test_months_genitive_and_lemma(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("pierwszego stycznia", "01.01"),
        ("trzeciego maja", "03.05"),
        ("dwudziestego pierwszego maja", "21.05"),
        ("dziewiętnastego grudnia", "19.12"),
        ("piątego maja 2023", "05.05.2023"),
        ("piątego maja roku dwa tysiące dwudziestego szóstego", "05.05.2026"),
        ("piątego maja 2026 roku", "05.05.2026"),
    ],
)
def test_months_with_day(normalize, text, expected):
    assert normalize(text) == expected


def test_months_name_ambiguity_documents_issue(normalize):
    # Conditional mapping keeps personal name "Maja" intact, only "3. maja" -> "03.05"
    assert normalize("Maja") == "maja"
    assert normalize("moja siostra Maja") == "moja siostra maja"
    assert normalize("trzeciego maja") == "03.05"  # day ordinal triggers month conversion
    assert normalize("w maju") == "w maju"  # no day -> no conversion
    # Verb form not mapped
    assert normalize("Maję") == "maję"


@pytest.mark.parametrize(
    "text",
    [
        "o piątej",
        "piąta rano",
        "dziesięć minut po piątej",
        "pięć procentów",
        "pięć złotówek",
        "pół litra",
        "tysiąc dziewięćsetny",
        "od piątej do szóstej",
        "jedna trzecia",
        "pierwszego stycznia",
    ],
)
def test_idempotency_extended(normalize, text):
    once = normalize(text)
    assert normalize(once) == once


# --- decimal fractions: 21.5 == 21 i 5/10 (jednostki dataset) ---


@pytest.mark.parametrize(
    "text,expected",
    [
        ("dwadzieścia jeden i pięć dziesiątych", "21.5"),
        ("21 i 5/10", "21.5"),
        ("dwadzieścia jeden i pięćdziesiąt setnych", "21.5"),  # 50/100 -> 0.5
        ("trzy i czternaście setnych", "3.14"),
        ("sto dwadzieścia trzy i czterysta pięćdziesiąt sześć tysięcznych", "123.456"),
        ("pięć dziesiątych", "0.5"),
        ("jedna dziesiąta", "0.1"),
        ("jedna druga", "0.5"),  # 1/2 == 0.5
        ("dwie i pół", "2.5"),  # via i pół -> already 2.5
        ("21.5", "21.5"),
        ("21,5", "21.5"),
    ],
)
def test_decimal_fractions_via_morfeusz(normalize, text, expected):
    assert normalize(text) == expected


def test_decimal_fraction_wer_equivalence(normalize):
    # 21.5 reference vs spoken decimal fractions should have WER 0
    assert normalize("dwadzieścia jeden i pięć dziesiątych") == normalize("21.5")
    assert normalize("21 i 5/10") == normalize("21.5")
    assert normalize("21,5") == normalize("21.5")
    # 0.5 vs 1/2
    assert normalize("0.5") == normalize("1/2")
    assert normalize("0.5") == normalize("jedna druga")
    assert normalize("0.5") == normalize("pięć dziesiątych")
    # via jiwer
    try:
        from polish_whisper_normalizer.jiwer import wer

        assert wer("dwadzieścia jeden i pięć dziesiątych", "21.5") == 0.0
        assert wer("21 i 5/10", "21.5") == 0.0
        assert wer("0.5", "1/2") == 0.0
        assert wer("0.5", "jedna druga") == 0.0
    except ImportError:
        pytest.skip("jiwer not installed")


@pytest.mark.parametrize(
    "text",
    [
        "dwadzieścia jeden i pięć dziesiątych",
        "21 i 5/10",
        "21.5",
        "pięć dziesiątych",
        "0.5",
        "1/2",
        "jedna druga",
        "trzy i czternaście setnych",
        "123.456",
    ],
)
def test_decimal_fraction_idempotency(normalize, text):
    once = normalize(text)
    assert normalize(once) == once


# --- Mailabs: ellipsis boundaries, colloquial "ośmnaście", "r." year abbreviation ---


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Piętnaście...", "15"),
        ("Piętnaście…", "15"),
        ("piętnaście . . .", "15"),
        ("Piętnaście... siedemnaście...", "15 17"),
        ("Piętnaście... siedemnaście... ośmnaście... 1860...", "15 17 18 1860"),
        ("ośmnaście", "18"),
        ("ośmnaście...", "18"),
        ("ośmnastu lat", "18 lat"),
        ("w r. 1860", "w roku 1860"),
        ("w r 1860", "w roku 1860"),
    ],
)
def test_ellipsis_colloquial_osmascie_year_abbrev(normalize, text, expected):
    assert normalize(text) == expected


def test_ellipsis_adds_no_spurious_dots(normalize):
    assert normalize("Piętnaście... siedemnaście... ośmnaście... 1860...") == "15 17 18 1860"


@pytest.mark.parametrize(
    "text",
    [
        "Piętnaście...",
        "ośmnaście...",
        "w r. 1860...",
        "Piętnaście... siedemnaście... ośmnaście... Było to w r. 1860...",
    ],
)
def test_ellipsis_colloquial_year_abbrev_idempotency(normalize, text):
    once = normalize(text)
    assert normalize(once) == once


# --- numeral separation: punctuation boundaries + invalid magnitude order ---


@pytest.mark.parametrize(
    "text,expected",
    [
        ("dziesięć, pięćset, dziewięćdziesiąt", "10 500 90"),
        ("dziesięć. pięćset. dziewięćdziesiąt", "10 500 90"),
        ("dwa, siedemset, siedem", "2 700 7"),
        ("piętnaście, siedemnaście, osiemnaście", "15 17 18"),
        # no separators: invalid magnitude order starts a new numeral
        ("dziesięć pięćset dziewięćdziesiąt", "10 590"),
        ("dwa siedemset siedem", "2 707"),
        # valid composition must still work
        ("dwa tysiące dwadzieścia trzy", "2023"),
        ("sto dwadzieścia trzy", "123"),
        ("siedemset siedem", "707"),
        # long digit-strings stay split (bug 5: avoid 9/21-digit tokens;
        # "340 808 57" not "340800857" so one ASR error != whole-token error)
        ("trzysta czterdzieści osiemset osiem pięć siedem", "340 808 57"),
    ],
)
def test_numeral_separation(normalize, text, expected):
    assert normalize(text) == expected


# --- spoken/digit time consistency ("godzina", "o godzinie") ---


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Jest godzina dwudziesta piętnaście", "jest 20:15"),
        ("Jest godzina 20.15", "jest 20:15"),
        ("Jest godzina 20:15", "jest 20:15"),
        ("Jest godzina 20,15", "jest 20:15"),
        ("o godzinie 16.05", "o 16:05"),
        ("o godzinie 16:05", "o 16:05"),
        ("o godzinie szesnastej piątej", "o 16:05"),
        ("o godzinie szesnastej", "o 16:00"),
        ("o godzinie dwudziestej pierwszej", "o 21:00"),
        ("o godzinie dwudziestej pierwszej piątej", "o 21:05"),
        ("o godzinie czternastej trzydziestej piątej", "o 14:35"),
    ],
)
def test_godzina_time_consistency(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Jest godzina 20.15",
        "o godzinie szesnastej piątej",
        "o godzinie dwudziestej pierwszej",
        "o godzinie 16",
        "dziesięć, pięćset, dziewięćdziesiąt",
        "dwa siedemset siedem",
        "trzysta czterdzieści osiemset osiem pięć siedem",
    ],
)
def test_numeral_time_idempotency(normalize, text):
    once = normalize(text)
    assert normalize(once) == once


# --- ordinal dot consistency: "piętnasty" and "15." both -> "15" ---


@pytest.mark.parametrize(
    "text,expected",
    [
        ("piętnasty", "15"),
        ("15.", "15"),
        ("dwudziesty pierwszy wiek", "21 wiek"),
        ("21. wiek", "21 wiek"),
        ("tysiąc dziewięćsetny rok", "1900 rok"),
        ("minął 20. i z tego", "minął 20 i z tego"),
    ],
)
def test_ordinal_dot_consistency(normalize, text, expected):
    assert normalize(text) == expected


def test_ordinal_vs_cardinal_equivalent(normalize):
    assert normalize("piętnasty") == normalize("15.")
    assert normalize("piętnasty") == normalize("15")
    assert normalize("dwudziesty pierwszy wiek") == normalize("21. wiek")
