"""Stress tests, fuzzing, and edge-case validation for Polish Whisper Normalizer.

Designed to be critical, adversarial, and push the normalizer to its limits:
- Extreme lengths and repetitive patterns (ReDoS resistance)
- Complex punctuation and unusual Unicode (emojis, zero-width, non-Latin)
- Idempotency verification across diverse linguistic constructs
- Semantic collisions (e.g. 'piec' stove vs 5, 'uliczny' vs 'ul', 'ok.' vs 'około')
- Numeric, fractional, and ordinal edge cases (e.g. 'dwa pierwsze', 'jeden drugi')
- Time and date boundaries (out-of-bounds hours, invalid leap dates, strftime fallback)
- Unbounded cache growth observation
- Multi-threaded concurrent execution
"""

from __future__ import annotations

import concurrent.futures

import pytest

from polish_whisper_normalizer import PolishTextNormalizer


@pytest.fixture(scope="module")
def normalizer() -> PolishTextNormalizer:
    return PolishTextNormalizer()


# ---------------------------------------------------------------------------
# 1. Adversarial & ReDoS Stress Testing
# ---------------------------------------------------------------------------


def test_adversarial_long_whitespace(normalizer: PolishTextNormalizer) -> None:
    """Stress test with long whitespace sequences."""
    text = "początek" + " " * 2000 + "środek" + "\t" * 500 + "koniec"
    res = normalizer(text)
    assert res == "początek środek koniec"


def test_adversarial_repeated_dots_and_ellipses(normalizer: PolishTextNormalizer) -> None:
    """Stress test with repetitive dots and ellipsis tokens."""
    text = "tekst" + "..." * 100 + " " + ". " * 100 + "koniec"
    res = normalizer(text)
    assert "tekst" in res
    assert "koniec" in res


def test_adversarial_nested_brackets(normalizer: PolishTextNormalizer) -> None:
    """Stress test deeply nested and mismatched brackets."""
    text = "a" + "[" * 50 + "wewnątrz" + "]" * 50 + "b" + "(" * 50 + "nawias" + ")" * 50 + "c"
    res = normalizer(text)
    assert "a" in res
    assert "b" in res
    assert "c" in res


def test_adversarial_punctuation_storm(normalizer: PolishTextNormalizer) -> None:
    """Stress test with dense, chaotic punctuation sequences."""
    storm = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~" * 10
    text = f"początek {storm} koniec"
    res = normalizer(text)
    assert res.startswith("początek")
    assert res.endswith("koniec")


def test_unusual_unicode_and_zero_width(normalizer: PolishTextNormalizer) -> None:
    """Stress test with zero-width characters, non-breaking spaces, and emojis."""
    text = "słowo\u200b\u200c\u200d\ufeffz\u00a0niewidzialnymi\u202fznakami oraz emoji 🚀🎉🔥"
    res = normalizer(text)
    assert "słowo" in res
    assert "niewidzialnymi" in res


def test_mixed_scripts(normalizer: PolishTextNormalizer) -> None:
    """Ensure non-Latin characters (Cyrillic, Greek, Chinese) don't crash."""
    text = "Spotkanie w Warszawie: привет мир, γεια σας, 你好, שלום"
    res = normalizer(text)
    assert "spotkanie w warszawie" in res


# ---------------------------------------------------------------------------
# 2. Idempotency Stress Testing: norm(norm(x)) == norm(x)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "sto dwadzieścia trzy",
        "dwudziesty pierwszy wiek",
        "piętnaście po piątej rano",
        "wpół do ósmej wieczorem",
        "pięćdziesiąt złotych i osiemdziesiąt groszy",
        "dwadzieścia procent rabatu",
        "minus dziesięć stopni Celsjusza",
        "trzy czwarte szklanki mąki",
        "półtora litra wody",
        "dwudziestego pierwszego maja roku dwa tysiące dwudziestego szóstego",
        "05.05.2026",
        "18:30",
        "12:00",
        "0:00",
        "123 456 789",
        "10 500",
        "1/3",
        "0.5",
        "21.5",
        "dr Kowalski",
        "prof Nowak",
        "tzw problem",
        "np jabłko",
        "itd itp",
        "między innymi",
    ],
)
def test_idempotency_standard_constructs(normalizer: PolishTextNormalizer, text: str) -> None:
    """Verify that applying normalization twice yields the same output."""
    pass1 = normalizer(text)
    pass2 = normalizer(pass1)
    assert pass1 == pass2, f"Idempotency failed for {text!r}: pass1={pass1!r}, pass2={pass2!r}"


def test_idempotency_decimals(normalizer: PolishTextNormalizer) -> None:
    """Verify that decimals like 0,05 / 2.25 remain decimals and are idempotent."""
    for val in ["0,05", "0.05", "0,25", "0.25", "2.25", "2,30 kg", "14,37 kg"]:
        pass1 = normalizer(val)
        pass2 = normalizer(pass1)
        assert pass1 == pass2, f"Failed idempotency for {val!r}: {pass1!r} != {pass2!r}"
        assert ":" not in pass1, f"Decimal {val!r} converted to clock time: {pass1!r}"


# ---------------------------------------------------------------------------
# 3. Numeric, Fractional & Ordinal Edge Cases
# ---------------------------------------------------------------------------


def test_plural_ordinals_no_gluing(normalizer: PolishTextNormalizer) -> None:
    """Ensure 'dwa pierwsze' does not glue into '21'."""
    assert normalizer("dwa pierwsze miejsca") == "2 1 miejsca"
    assert normalizer("trzy pierwsze dni") == "3 1 dni"
    assert normalizer("cztery pierwsze osoby") == "4 1 osoby"


def test_jeden_drugi_no_fraction(normalizer: PolishTextNormalizer) -> None:
    """Ensure masculine 'jeden drugi' does not convert to fraction 0.5."""
    assert normalizer("jeden drugi trzeci") == "1 2 3"
    assert normalizer("ani jeden ani drugi") == "ani 1 ani 2"
    # True fractions still work
    assert normalizer("jedna druga") == "0.5"
    assert normalizer("dwie trzecie") == "2/3"
    assert normalizer("trzy czwarte") == "3/4"


def test_large_numbers_and_overflow(normalizer: PolishTextNormalizer) -> None:
    """Stress test with quadrillions, quintillions, and chained multipliers."""
    text = "sto trylionów dwieście biliardów trzysta bilionów czterysta miliardów"
    res = normalizer(text)
    assert any(c.isdigit() for c in res)


def test_fraction_zero_numerator_and_denominator(normalizer: PolishTextNormalizer) -> None:
    """Test fractions with zero components."""
    assert normalizer("zero trzecich") == "0/3"
    assert "zerowych" in normalizer("pięć zerowych")


def test_fractions_greater_than_one(normalizer: PolishTextNormalizer) -> None:
    """Verify improper fractions or mixed phrases."""
    res = normalizer("cztery całe i trzy czwarte")
    assert "4" in res
    assert "3/4" in res


# ---------------------------------------------------------------------------
# 4. Spoken Time & Date Edge Cases
# ---------------------------------------------------------------------------


def test_poludnie_temporal_vs_geographic(normalizer: PolishTextNormalizer) -> None:
    """Ensure 'w południe' normalizes to 12:00, while geographic directions stay words."""
    assert normalizer("jest południe") == "jest 12:00"
    assert normalizer("spotkamy się w południe") == "spotkamy się w 12:00"
    assert normalizer("przed południem") == "przed 12:00"
    assert normalizer("po południu") == "po 12:00"
    assert normalizer("na południe") == "na południe"
    assert normalizer("na południu") == "na południu"
    assert normalizer("z południa") == "z południa"


def test_out_of_bounds_time_digits(normalizer: PolishTextNormalizer) -> None:
    """Verify out-of-bounds clock times are not mangled."""
    assert normalizer("godzina 25:70") == "godzina 25:70"
    assert normalizer("o godzinie 25") == "o godzinie 25"
    # 25.00 is out of bounds for time (>24h), so it is parsed as a number and .00 is stripped to 25
    assert normalizer("25.00") == "25"
    assert normalizer("12.65") == "12.65"


def test_invalid_date_strftime_no_leak() -> None:
    """Ensure invalid date with strftime format falls back cleanly and does not leak %Y."""
    norm = PolishTextNormalizer(date_format="%Y-%m-%d")
    res = norm("32 maja 2024 roku")
    assert "%Y" not in res
    assert res == "32.05.2024"


# ---------------------------------------------------------------------------
# 5. Semantic & Abbreviation Overmatching
# ---------------------------------------------------------------------------


def test_abbreviation_no_overmatching(normalizer: PolishTextNormalizer) -> None:
    """Ensure abbreviations do not corrupt valid Polish words."""
    assert normalizer("uliczny grajek") == "uliczny grajek"
    assert normalizer("uliczka w miasteczku") == "uliczka w miasteczku"
    assert normalizer("doktorat z chemii") == "doktorat z chemii"
    assert normalizer("doktorant na uczelni") == "doktorant na uczelni"
    assert normalizer("numerolog") == "numerolog"
    assert normalizer("numeracja stron") == "numeracja stron"
    assert normalizer("wszystko jest ok.") == "wszystko jest ok"
    # Real abbreviations still convert properly:
    assert normalizer("ul. Mickiewicza") == "ul mickiewicza"
    assert normalizer("ulica Mickiewicza") == "ul mickiewicza"
    assert normalizer("dr Kowalski") == "dr kowalski"
    assert normalizer("doktor Kowalski") == "dr kowalski"
    assert normalizer("nr 5") == "nr 5"
    assert normalizer("numer 5") == "nr 5"
    assert normalizer("ok. 5 osób") == "około 5 osób"


def test_piec_contextual_collision(normalizer: PolishTextNormalizer) -> None:
    """Ensure 'piec' (oven / to bake) is not converted to 5 in non-numeric context."""
    assert normalizer("będziemy piec chleb") == "będziemy piec chleb"
    assert normalizer("nowy piec kaflowy") == "nowy piec kaflowy"
    assert normalizer("wstaw do pieca") == "wstaw do pieca"
    # Diacritic-less number uses still convert:
    assert normalizer("piec") == "5"
    assert normalizer("dwadziescia piec") == "25"
    assert normalizer("piec zlotych") == "5 zł"
    assert normalizer("piec tysiecy") == "5000"


# ---------------------------------------------------------------------------
# 6. Cache Growth & Concurrency Stress Testing
# ---------------------------------------------------------------------------


def test_cache_growth_bounded_check(normalizer: PolishTextNormalizer) -> None:
    """Observe cache growth behavior when processing arbitrary out-of-vocabulary words.

    Documents that PolishLemmatizer._cache and PolishNumberNormalizer._canon_cache
    grow monotonically with unique words.
    """
    initial_lem_cache = len(normalizer.standardize_numbers.lemmatizer._cache)

    # Generate 50 unique alphabetic Polish pseudo-words
    unique_words = [f"słówkotestowexy{chr(97 + (i % 26))}{chr(97 + (i // 26))}" for i in range(50)]
    for w in unique_words:
        normalizer(w)

    final_lem_cache = len(normalizer.standardize_numbers.lemmatizer._cache)
    # Documents that lemmatizer cache grew by the number of unique words:
    assert final_lem_cache >= initial_lem_cache + 50


def test_concurrent_multithreaded_stress(normalizer: PolishTextNormalizer) -> None:
    """Stress test shared normalizer instance across multiple concurrent threads."""
    samples = [
        "Było piętnaście po piątej rano w Warszawie.",
        "Koszt wynosi sto dwadzieścia trzy złote i pięćdziesiąt groszy.",
        "Spotkanie dwudziestego pierwszego maja dwa tysiące dwudziestego szóstego roku.",
        "Zgodnie z prognozą temperatura spadnie o pięć procent.",
        "Pociąg odjeżdża o godzinie osiemnastej czterdzieści pięć z peronu drugiego.",
    ]

    def worker(worker_id: int) -> int:
        for _ in range(25):
            for s in samples:
                res = normalizer(s)
                assert len(res) > 0
        return worker_id

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(8)]
        results = [f.result() for f in futures]

    assert len(results) == 8
