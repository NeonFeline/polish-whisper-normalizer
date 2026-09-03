import pytest

from polish_whisper_normalizer import (
    BasicTextNormalizer,
    PolishLemmatizer,
    PolishNumberNormalizer,
    PolishTextNormalizer,
    PolishTimeNormalizer,
    remove_symbols,
    remove_symbols_and_diacritics,
)


@pytest.fixture(scope="module")
def normalize():
    return PolishTextNormalizer()


@pytest.mark.parametrize(
    "text, expected",
    [
        # cardinal numbers
        ("zero", "0"),
        ("pięć", "5"),
        ("sześć", "6"),
        ("dziesięć", "10"),
        ("dwadzieścia jeden", "21"),
        ("trzydzieści siedem", "37"),
        ("czterdzieści dwa", "42"),
        ("dziewięćdziesiąt dziewięć", "99"),
        ("sto", "100"),
        ("sto dwadzieścia trzy", "123"),
        ("sto jedenaście", "111"),
        ("dwieście trzydzieści siedem", "237"),
        ("dwieście pięćdziesiąt", "250"),
        ("pięćset", "500"),
        ("dziewięćset dziewięćdziesiąt dziewięć", "999"),
        ("tysiąc", "1000"),
        ("tysiąc dwieście", "1200"),
        ("dwa tysiące", "2000"),
        ("dwa tysiące dwadzieścia", "2020"),
        ("pięć tysięcy", "5000"),
        ("sto dwadzieścia tysięcy trzysta", "120300"),
        ("milion", "1000000"),
        ("milion dwieście tysięcy", "1200000"),
        ("dwa miliony sto dwadzieścia trzy", "2000123"),
        ("trzy miliony", "3000000"),
        # decimals
        ("trzy przecinek czternaście", "3.14"),
        ("trzy przecinek jeden cztery", "3.14"),
        ("dwa kropka pięć", "2.5"),
        # signs
        ("minus dziesięć", "-10"),
        ("plus pięć", "+5"),
        # currency
        ("pięć złotych", "5 zł"),
        ("pięćdziesiąt groszy", "50 gr"),
        ("sto złotych i pięćdziesiąt groszy", "100 zł 50 gr"),
        ("sto euro", "100 €"),
        # percent
        ("dwadzieścia procent", "20%"),
        ("pięć procent", "5%"),
        # halves
        ("półtora kilograma", "1.5 kilograma"),
        ("dwa i pół", "2.5"),
    ],
)
def test_numbers(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("piąta trzydzieści", "5:30"),
        ("dwudziesta trzydzieści", "20:30"),
        ("dwudziesta piętnaście", "20:15"),
        ("dwudziesta dwadzieścia pięć", "20:25"),
        ("wpół do ósmej", "7:30"),
        ("wpół do pierwszej", "0:30"),
        ("za piętnaście ósma", "7:45"),
        ("za pięć dwunasta", "11:55"),
        ("za kwadrans piąta", "4:45"),
        ("piętnaście po piątej", "5:15"),
        ("kwadrans po ósmej", "8:15"),
        ("godzina piętnasta trzydzieści", "15:30"),
        ("godzina ósma", "8:00"),
        ("północ", "0:00"),
        ("południe", "12:00"),
        ("piętnasta zero pięć", "15:05"),
        ("o piętnastej trzydzieści", "o 15:30"),
        ("o dwudziestej pierwszej trzydzieści", "o 21:30"),
    ],
)
def test_time(normalize, text, expected):
    assert normalize(text) == expected


def test_time_no_false_positive(normalize):
    # a bare ordinal with no minutes is not turned into a time
    assert normalize("druga trzydzieści") == "2:30"
    assert normalize("piąta piętnaście") == "5:15"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("źle", "źle"),
        ("mąż", "mąż"),
        ("pięć", "5"),
        ("sześć", "6"),
        ("żółć", "żółć"),
        ("ćma", "ćma"),
        ("łąka", "łąka"),
        ("nię", "nię"),
    ],
)
def test_diacritics_preserved(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Dzień Dobry!", "dzień dobry"),
        ("(cicho) start", "start"),
        ("[muzyka] słowo", "słowo"),
        ("eee coś", "coś"),
        ("hmm tak", "tak"),
    ],
)
def test_punctuation_and_fillers(normalize, text, expected):
    assert normalize(text) == expected


def test_full_sentence(normalize):
    assert normalize("Było piętnaście po piątej, minus dziesięć stopni.") == "było 5:15 -10 stopni"


def test_arabic_numbers_passthrough(normalize):
    assert normalize("rok 2020") == "rok 2020"
    assert normalize("3,14") == "3.14"
    assert normalize("temperatura 36,6") == "temperatura 36.6"


def test_basic_normalizer_keeps_diacritics():
    n = BasicTextNormalizer()
    assert n("Żółć!") == "żółć"


@pytest.mark.parametrize(
    "text, expected",
    [
        # declined cardinal forms
        ("pięciu mężczyzn", "5 mężczyzn"),
        ("dwóm osobom", "2 osobom"),
        ("dwóch braci", "2 braci"),
        ("dwudziestu lat", "20 lat"),
        ("stu żołnierzy", "100 żołnierzy"),
        ("pięciuset osób", "500 osób"),
        ("tysiąca ludzi", "1000 ludzi"),
        ("milionów ludzi", "1000000 ludzi"),
    ],
)
def test_declined_cardinals(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("pierwszy", "1."),
        ("drugi", "2."),
        ("trzeci", "3."),
        ("czwarty", "4."),
        ("piąty", "5."),
        ("dziewiętnasty", "19."),
        ("dwudziesty", "20."),
        ("trzydziesty", "30."),
        ("setny", "100."),
        ("dwusetny", "200."),
        ("tysięczny", "1000."),
        ("milionowy", "1000000."),
        ("dwudziesty pierwszy", "21."),
        ("trzydziesty drugi", "32."),
        ("sto pierwszy", "101."),
        ("sto dwudziesty pierwszy", "121."),
        ("pierwszego", "1."),
        ("trzeciej", "3."),
        ("czwartym", "4."),
        ("piątej", "5."),
        ("ósmego", "8."),
        ("dwunastego", "12."),
        ("dziewiętnastego", "19."),
        ("dwudziestego", "20."),
        ("dwudziestego pierwszego", "21."),
        ("trzydziestego", "30."),
        ("druga strona", "2. strona"),
        ("trzecia osoba", "3. osoba"),
        ("dwudziesty pierwszy wiek", "21. wiek"),
        ("piąta rocznica", "5. rocznica"),
        ("trzeciego maja", "3. 5"),
        ("dwudziestego pierwszego maja", "21. 5"),
        ("dziewiętnastego grudnia", "19. 12"),
    ],
)
def test_declined_ordinals(normalize, text, expected):
    assert normalize(text) == expected


def test_non_number_words_untouched(normalize):
    assert normalize("maja") == "maja"
    assert normalize("jedynka") == "jedynka"
    assert normalize("dwójka") == "dwójka"
    assert normalize("setka") == "setka"
    assert normalize("półka") == "półka"


# ---------------------------------------------------------------------------
# Cardinal numbers, exhaustive units/tens/hundreds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("jeden", "1"),
        ("dwa", "2"),
        ("trzy", "3"),
        ("cztery", "4"),
        ("sześć", "6"),
        ("siedem", "7"),
        ("osiem", "8"),
        ("dziewięć", "9"),
        ("dziesięć", "10"),
        ("jedenaście", "11"),
        ("dwanaście", "12"),
        ("trzynaście", "13"),
        ("czternaście", "14"),
        ("piętnaście", "15"),
        ("szesnaście", "16"),
        ("siedemnaście", "17"),
        ("osiemnaście", "18"),
        ("dziewiętnaście", "19"),
        ("dwadzieścia", "20"),
        ("trzydzieści", "30"),
        ("czterdzieści", "40"),
        ("pięćdziesiąt", "50"),
        ("sześćdziesiąt", "60"),
        ("siedemdziesiąt", "70"),
        ("osiemdziesiąt", "80"),
        ("dziewięćdziesiąt", "90"),
        ("dwadzieścia dziewięć", "29"),
        ("trzydzieści jeden", "31"),
        ("dziewięćdziesiąt dziewięć", "99"),
    ],
)
def test_cardinal_tens_units(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("sto", "100"),
        ("dwieście", "200"),
        ("trzysta", "300"),
        ("czterysta", "400"),
        ("pięćset", "500"),
        ("sześćset", "600"),
        ("siedemset", "700"),
        ("osiemset", "800"),
        ("dziewięćset", "900"),
        ("sto jeden", "101"),
        ("sto dziesięć", "110"),
        ("sto dwadzieścia", "120"),
        ("dwieście dwadzieścia dwa", "222"),
        ("trzysta czterdzieści pięć", "345"),
        ("dziewięćset dziewięćdziesiąt dziewięć", "999"),
    ],
)
def test_cardinal_hundreds(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("tysiąc", "1000"),
        ("tysiąc jeden", "1001"),
        ("tysiąc sto", "1100"),
        ("tysiąc dwieście", "1200"),
        ("dwa tysiące", "2000"),
        ("dwa tysiące trzysta", "2300"),
        ("pięć tysięcy", "5000"),
        ("dziesięć tysięcy", "10000"),
        ("sto tysięcy", "100000"),
        ("sto dwadzieścia trzy tysiące czterysta pięćdziesiąt sześć", "123456"),
        ("milion", "1000000"),
        ("milion dwieście tysięcy", "1200000"),
        ("dwa miliony", "2000000"),
        ("dwa miliony sto dwadzieścia trzy", "2000123"),
    ],
)
def test_cardinal_thousands_millions(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("miliard", "1000000000"),
        ("dwa miliardy", "2000000000"),
        ("trzy miliardy", "3000000000"),
        ("bilion", "1000000000000"),
        ("biliard", "1000000000000000"),
        ("trylion", "1000000000000000000"),
    ],
)
def test_cardinal_long_scale(normalize, text, expected):
    assert normalize(text) == expected


# ---------------------------------------------------------------------------
# Decimals and fractions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("zero przecinek pięć", "0.5"),
        ("jeden przecinek zero", "1.0"),
        ("dwa przecinek pięć", "2.5"),
        ("trzy przecinek czternaście", "3.14"),
        ("przecinek pięć", ".5"),
        ("zero kropka zero", "0.0"),
    ],
)
def test_decimals(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("półtora", "1.5"),
        ("półtorej", "1.5"),
        ("dwa i pół", "2.5"),
        ("trzy i pół", "3.5"),
        ("pięć i pół", "5.5"),
        ("sześć i pół", "6.5"),
        ("dwadzieścia dwa i pół", "22.5"),
        ("półtora miliona", "1500000"),
    ],
)
def test_fractions(normalize, text, expected):
    assert normalize(text) == expected


# ---------------------------------------------------------------------------
# Currency, percent, signs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("jeden złoty", "1 zł"),
        ("dwa złote", "2 zł"),
        ("pięć złotych", "5 zł"),
        ("dziesięć złotych", "10 zł"),
        ("jeden grosz", "1 gr"),
        ("pięć groszy", "5 gr"),
        ("dwadzieścia euro", "20 €"),
        ("pięćdziesiąt centów", "50 ¢"),
        ("sto dolarów", "100 $"),
        ("pięć dolarów", "5 $"),
    ],
)
def test_currency(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("jeden procent", "1%"),
        ("pięć procent", "5%"),
        ("dwadzieścia procent", "20%"),
        ("sto procent", "100%"),
        ("trzydzieści pięć procent", "35%"),
    ],
)
def test_percent(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("minus dziesięć", "-10"),
        ("minus piętnaście", "-15"),
        ("plus pięć", "+5"),
        ("plus dwadzieścia", "+20"),
        ("minus pięć stopni", "-5 stopni"),
        ("minus dwadzieścia stopni", "-20 stopni"),
    ],
)
def test_signs(normalize, text, expected):
    assert normalize(text) == expected


# ---------------------------------------------------------------------------
# Declension (cardinals)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("trzech", "3"),
        ("czterech", "4"),
        ("pięciu", "5"),
        ("sześciu", "6"),
        ("siedmiu", "7"),
        ("ośmiu", "8"),
        ("dziewięciu", "9"),
        ("dziesięciu", "10"),
        ("jedenastu", "11"),
        ("dwunastu", "12"),
        ("piętnastu", "15"),
        ("dwudziestu", "20"),
        ("trzydziestu", "30"),
        ("czterdziestu", "40"),
        ("pięćdziesięciu", "50"),
        ("dziewięćdziesięciu", "90"),
        ("dwudziestu pięciu", "25"),
        ("trzydziestu pięciu", "35"),
        ("dwudziestu dwóch", "22"),
        ("czterdziestu czterech", "44"),
        ("stu", "100"),
        ("pięciuset", "500"),
        ("tysiąca", "1000"),
        ("milionów", "1000000"),
    ],
)
def test_declined_cardinals_more(normalize, text, expected):
    assert normalize(text) == expected


# ---------------------------------------------------------------------------
# Declension (ordinals) — cases and genders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        # genitive
        ("pierwszego", "1."),
        ("drugiego", "2."),
        ("trzeciego", "3."),
        ("czwartego", "4."),
        ("piątego", "5."),
        ("szóstego", "6."),
        ("siódmego", "7."),
        ("ósmego", "8."),
        ("dziewiątego", "9."),
        ("dziesiątego", "10."),
        ("jedenastego", "11."),
        ("dwunastego", "12."),
        ("dwudziestego", "20."),
        ("trzydziestego", "30."),
        ("czterdziestego", "40."),
        # dative / locative feminine
        ("pierwszej", "1."),
        ("drugiej", "2."),
        ("trzeciej", "3."),
        ("czwartej", "4."),
        ("piątej", "5."),
        ("szóstej", "6."),
        ("siódmej", "7."),
        ("ósmej", "8."),
        ("dziewiątej", "9."),
        ("dziesiątej", "10."),
        ("jedenastej", "11."),
        ("dwunastej", "12."),
        ("dwudziestej", "20."),
        ("trzydziestej", "30."),
        ("czterdziestej", "40."),
        # instrumental / locative masculine
        ("pierwszym", "1."),
        ("drugim", "2."),
        ("trzecim", "3."),
        ("czwartym", "4."),
        ("piątym", "5."),
        # accusative feminine
        ("pierwszą", "1."),
        ("drugą", "2."),
        ("trzecią", "3."),
        ("czwartą", "4."),
        ("piątą", "5."),
        # plural genitive
        ("pierwszych", "1."),
        ("drugich", "2."),
        ("trzecich", "3."),
        ("czwartych", "4."),
    ],
)
def test_declined_ordinals_cases(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("dwudziestym pierwszym", "21."),
        ("dwudziestą pierwszą", "21."),
        ("dwudziestego pierwszego", "21."),
        ("dwudziestej pierwszej", "21."),
        ("trzydziestego drugiego", "32."),
        ("trzydziesty drugi", "32."),
        ("sto pierwszy", "101."),
        ("sto dwudziesty pierwszy", "121."),
        ("setnego", "100."),
        ("dwusetnego", "200."),
        ("pięćsetnego", "500."),
        ("tysięcznego", "1000."),
        ("milionowego", "1000000."),
    ],
)
def test_compound_ordinals(normalize, text, expected):
    assert normalize(text) == expected


# ---------------------------------------------------------------------------
# Robustness: whitespace, idempotency, symbols
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", ""),
        ("   ", ""),
        ("\t\n  test \n", "test"),
        ("  sto   dwadzieścia   ", "120"),
    ],
)
def test_whitespace(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "sto dwadzieścia trzy",
        "dwudziestego pierwszego maja o piętnastej trzydzieści",
        "pięć złotych pięćdziesiąt groszy",
        "trzy przecinek czternaście",
        "minus dwadzieścia stopni",
        "trzydziesty drugi",
    ],
)
def test_idempotency(normalize, text):
    once = normalize(text)
    assert normalize(once) == once


@pytest.mark.parametrize(
    "text, expected",
    [
        ("rok 2020", "rok 2020"),
        ("$5", "5 $"),
        ("€10", "10 €"),
        ("20:30", "20:30"),
        ("3,14", "3.14"),
        ("temperatura 36,6", "temperatura 36.6"),
    ],
)
def test_arabic_and_symbols_passthrough(normalize, text, expected):
    assert normalize(text) == expected


# ---------------------------------------------------------------------------
# Conjunction handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ja i ty", "ja i ty"),
        ("dwa i trzy", "2 3"),
        ("pierwszy i drugi", "1. 2."),
        ("sto złotych i pięćdziesiąt groszy", "100 zł 50 gr"),
    ],
)
def test_conjunction_i(normalize, text, expected):
    assert normalize(text) == expected


# ---------------------------------------------------------------------------
# Full sentences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Spotkanie odbędzie się dwudziestego pierwszego maja o piętnastej trzydzieści.",
            "spotkanie odbędzie się 21. 5 o 15:30",
        ),
        (
            "Kosztowało to sto złotych i pięćdziesiąt groszy, a zniżka wyniosła dwadzieścia procent.",
            "kosztowało to 100 zł 50 gr a zniżka wyniosła 20%",
        ),
        (
            "Na konferencję przybyło pięciuset uczestników z dwudziestu krajów.",
            "na konferencję przybyło 500 uczestników z 20 krajów",
        ),
        ("Było piętnaście po piątej, minus dziesięć stopni.", "było 5:15 -10 stopni"),
    ],
)
def test_full_sentences(normalize, text, expected):
    assert normalize(text) == expected


# ---------------------------------------------------------------------------
# Component isolation / unit tests
# ---------------------------------------------------------------------------


def test_number_normalizer_direct():
    n = PolishNumberNormalizer()
    assert n("dwudziestu pięciu złotych") == "25 zł"
    assert n("sto dwadzieścia trzy") == "123"


def test_time_normalizer_direct():
    n = PolishTimeNormalizer()
    assert n("wpół do dziesiątej") == "9:30"
    assert n("piąta trzydzieści") == "5:30"


def test_lemmatizer_direct():
    n = PolishLemmatizer()
    analyses = n.analyse("dwudziestu")
    assert any(base == "dwadzieścia" for base, _ in analyses)


def test_remove_symbols_functions():
    assert remove_symbols("Coś, tu jest! 123") == "Coś  tu jest  123"
    assert remove_symbols("Coś, tu jest! 123", keep="!,") == "Coś, tu jest! 123"
    assert remove_symbols_and_diacritics("Żółć") == "Zolc"


def test_basic_normalizer_variants():
    assert BasicTextNormalizer()("Żółć!!") == "żółć"
    assert BasicTextNormalizer(remove_diacritics=True)("Żółć!!") == "zolc"
    assert BasicTextNormalizer(split_letters=True)("kot") == "k o t"


# ---------------------------------------------------------------------------
# Currencies: words, ISO codes, and symbols
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        # Polish złoty / grosz
        ("jeden złoty", "1 zł"),
        ("dwa złote", "2 zł"),
        ("pięć złotych", "5 zł"),
        ("sto złotych i pięćdziesiąt groszy", "100 zł 50 gr"),
        # euro
        ("dziesięć euro", "10 €"),
        ("dwadzieścia euro", "20 €"),
        # dolar
        ("sto dolarów", "100 $"),
        ("pięć dolarów", "5 $"),
        # funt
        ("trzydzieści funtów", "30 £"),
        # cent
        ("pięćdziesiąt centów", "50 ¢"),
    ],
)
def test_currency_words(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("10 eur", "10 €"),
        ("10 usd", "10 $"),
        ("5 pln", "5 zł"),
        ("20 gbp", "20 £"),
        ("100 pln", "100 zł"),
    ],
)
def test_currency_iso_codes(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # symbol before amount -> moved after
        ("€10", "10 €"),
        ("$5", "5 $"),
        ("£20", "20 £"),
        ("€ 10", "10 €"),
        # symbol already after amount
        ("10 €", "10 €"),
        ("5 $", "5 $"),
        ("10€", "10 €"),
        ("5$", "5 $"),
        # symbols with decimals
        ("€1,50", "1.50 €"),
        ("$1.50", "1.50 $"),
    ],
)
def test_currency_symbols(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # declined currency amount
        ("dwudziestu euro", "20 €"),
        ("dwustu dolarów", "200 $"),
        ("pięciu złotych", "5 zł"),
    ],
)
def test_currency_declined_amounts(normalize, text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # stray symbols without an amount are dropped
        ("cena € to", "cena to"),
        ("€ cena", "cena"),
        # standalone currency words are kept as-is
        ("kupię euro", "kupię euro"),
        ("to jest złoty", "to jest złoty"),
    ],
)
def test_currency_stray(normalize, text, expected):
    assert normalize(text) == expected
