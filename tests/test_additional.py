import pytest
from polish_whisper_normalizer import PolishTextNormalizer

@pytest.fixture(scope="module")
def normalize():
    return PolishTextNormalizer()

# --- original 9 fixes coverage, extended ---

@pytest.mark.parametrize("text,expected", [
    ("o piątej", "o 5:00"),
    ("o piątej rano", "o 5:00 rano"),
    ("piąta rano", "5:00 rano"),
    ("szósta rano", "6:00 rano"),
    ("o drugiej stronie", "o 2. stronie"),  # no false positive
    ("o piątej stronie", "o 5. stronie"),
    ("piąta rocznica", "5. rocznica"),  # not time
])
def test_standalone_o_time_extended(normalize, text, expected):
    assert normalize(text) == expected

@pytest.mark.parametrize("text,expected", [
    ("dziesięć minut po piątej", "5:10"),
    ("pięć minut po piątej", "5:05"),
    ("za dwadzieścia minut ósma", "7:40"),
    ("za pięć minut ósma", "7:55"),
    ("za dwie minuty ósma", "7:58"),
    ("jedna minuta po piątej", "5:01"),
    ("za pięć minut", "za 5 minut"),  # duration, not time
    ("pięć minut", "5 minut"),
])
def test_time_minut_extended(normalize, text, expected):
    assert normalize(text) == expected

@pytest.mark.parametrize("text,expected", [
    ("od piątej do szóstej", "od 5:00 do 6:00"),
    ("od ósmej do dziesiątej", "od 8:00 do 10:00"),
    ("od piątego do szóstego", "od 5. do 6."),  # ordinal, not time
])
def test_time_ranges_extended(normalize, text, expected):
    assert normalize(text) == expected

@pytest.mark.parametrize("text,expected", [
    ("pięć procentów", "5%"),
    ("dwudziestu procentów", "20%"),
    ("pięćdziesiąt procenty", "50%"),
    ("pięć procenta", "5%"),
    ("sto procent", "100%"),
])
def test_percent_declined_extended(normalize, text, expected):
    assert normalize(text) == expected

@pytest.mark.parametrize("text,expected", [
    ("pięć złotówek", "5 zł"),
    ("dwie złotówki", "2 zł"),
    ("sto złotówek", "100 zł"),
    ("jedna złotówka", "1 zł"),
    ("złotówkę", "złotówka"),  # standalone without number -> lemmatized artifact
    ("kupię złotówkę", "kupię złotówka"),
])
def test_currency_zlotowka_extended(normalize, text, expected):
    assert normalize(text) == expected

@pytest.mark.parametrize("text,expected", [
    ("pół litra", "0.5 litra"),
    ("pół godziny", "0.5 godziny"),
    ("pół kilo", "0.5 kilo"),
    ("półtora", "1.5"),
    ("północy", "północy"),  # not half
    ("półmetek", "półmetek"),
    ("wpół do ósmej", "7:30"),
    ("północ", "0:00"),
])
def test_half_extended(normalize, text, expected):
    assert normalize(text) == expected

@pytest.mark.parametrize("text,expected", [
    ("tysiąc dziewięćsetny", "1900."),
    ("tysiąc dziewięćsetny rok", "1900. rok"),
    ("tysiąc dwusetny", "1200."),
    ("dziewięćsetny", "900."),
    ("tysiąc dziewięćset dziewięćdziesiąty dziewiąty", "1999."),
])
def test_ordinal_multiplier_extended(normalize, text, expected):
    assert normalize(text) == expected

@pytest.mark.parametrize("text,expected", [
    ("jedna trzecia", "1/3"),
    ("trzy czwarte", "3/4"),
    ("dwie trzecie", "2/3"),
    ("jedna druga", "1/2"),
    ("trzecia osoba", "3. osoba"),  # no false positive
    ("druga strona", "2. strona"),
])
def test_fractions_extended(normalize, text, expected):
    assert normalize(text) == expected

# --- months: comprehensive, highlights global-mapping trade-off ---

@pytest.mark.parametrize("text,expected", [
    ("stycznia", "1"),
    ("lutego", "2"),
    ("marca", "3"),
    ("kwietnia", "4"),
    ("maja", "5"),
    ("czerwca", "6"),
    ("lipca", "7"),
    ("sierpnia", "8"),
    ("września", "9"),
    ("października", "10"),
    ("listopada", "11"),
    ("grudnia", "12"),
    # nominative too
    ("styczeń", "1"),
    ("maj", "5"),
    ("grudzień", "12"),
    # locative / dative
    ("w grudniu", "w 12"),
    ("w maju", "w 5"),
])
def test_months_genitive_and_lemma(normalize, text, expected):
    assert normalize(text) == expected

@pytest.mark.parametrize("text,expected", [
    ("pierwszego stycznia", "1. 1"),
    ("trzeciego maja", "3. 5"),
    ("dwudziestego pierwszego maja", "21. 5"),
    ("dziewiętnastego grudnia", "19. 12"),
    ("piątego maja 2023", "5. 5 2023"),  # current spaced output, not 05.05.2023
])
def test_months_with_day(normalize, text, expected):
    assert normalize(text) == expected

def test_months_name_ambiguity_documents_issue(normalize):
    # Global mapping converts personal name "Maja" -> "5", which is undesirable.
    # This test documents the current behaviour and questions its sensibleness.
    assert normalize("Maja") == "5"  # name incorrectly mapped
    assert normalize("moja siostra Maja") == "moja siostra 5"
    # Verb form not mapped
    assert normalize("Maję") == "maję"

@pytest.mark.parametrize("text", [
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
])
def test_idempotency_extended(normalize, text):
    once = normalize(text)
    assert normalize(once) == once
