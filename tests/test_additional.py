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
        ("o drugiej stronie", "o 2. stronie"),  # no false positive
        ("o piątej stronie", "o 5. stronie"),
        ("piąta rocznica", "5. rocznica"),  # not time
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
        ("od piątego do szóstego", "od 5. do 6."),  # ordinal, not time
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
        ("tysiąc dziewięćsetny", "1900."),
        ("tysiąc dziewięćsetny rok", "1900. rok"),
        ("tysiąc dwusetny", "1200."),
        ("dziewięćsetny", "900."),
        ("tysiąc dziewięćset dziewięćdziesiąty dziewiąty", "1999."),
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
        ("jedna druga", "1/2"),
        ("trzecia osoba", "3. osoba"),  # no false positive
        ("druga strona", "2. strona"),
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
