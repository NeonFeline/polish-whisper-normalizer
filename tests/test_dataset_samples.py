import pytest

from polish_whisper_normalizer import PolishTextNormalizer


@pytest.fixture(scope="module")
def normalize():
    return PolishTextNormalizer()


# Real BIGOS/PELCRA ref_orig samples – validates that our 9 fixes are triggered
# and that diacritics are kept, no over-normalization for names.
@pytest.mark.parametrize(
    "text, expected",
    [
        # BIGOS – Clarin Studio date (conditional month)
        (
            "na tę wiadomość napoleon postanawia ruszyć niezwłocznie do warszawy gdzie staje osiemnastego grudnia za nim pójdzie gwardja",
            "na tę wiadomość napoleon postanawia ruszyć niezwłocznie do warszawy gdzie staje 18. 12 za nim pójdzie gwardja",
        ),
        # PELCRA Diabiz – date with month + time context
        (
            "Wie pani co Chwileczkę Nie faktura jest piętnastego maja termin płatności minął dwudziestego I z tego względu plan spłaty został zerwany",
            "wie pani co chwileczkę nie faktura jest 15. 5 termin płatności minął 20. i z tego względu plan spłaty został zerwany",
        ),
        # PELCRA – currency + half
        (
            "połączenia głosowe z poprzedniego miesiąca za cztery i pół złotego połączenia na infolinię za osiem złotych",
            "połączenia głosowe z poprzedniego miesiąca za 4.5 zł połączenia na infolinię za 8 zł",
        ),
        # PELCRA – percent via Morfeusz
        (
            "No nie jestem pani zagwarantować w stanie stu procentach jeżeli faktycznie tutaj coś się takiego zadziało",
            "no nie jestem pani zagwarantować w stanie 100% jeżeli faktycznie tutaj coś się takiego zadziało",
        ),
        # FLEURS – geographic północ should NOT become 0:00 (declined form)
        (
            "w północnej części pasma sentinel range znajdują się najwyższe góry antarktydy",
            "w północnej części pasma sentinel range znajdują się najwyższe góry antarktydy",
        ),
        # Mailabs – cardinal
        (
            "on gdy hanowi na srebrny półmisek rzucił łeb księcia iflaku wdzięczny mu hagan dziesięć odalisek",
            "on gdy hanowi na srebrny półmisek rzucił łeb księcia iflaku wdzięczny mu hagan 10 odalisek",
        ),
        # CommonVoice – punctuation stripping but diacritics kept
        ("Ale tego chyba nie myślisz na serio.", "ale tego chyba nie myślisz na serio"),
        (
            "później zabroniła ruszać się z miejsc światła same pogasły",
            "później zabroniła ruszać się z miejsc światła same pogasły",
        ),
        # Azon – scientific, number
        (
            "energia wzbudzenia pułapkowana umożliwia przeprowadzenie fotochemicznej reakcji",
            "energia wzbudzenia pułapkowana umożliwia przeprowadzenie fotochemicznej reakcji",
        ),
        # Name preservation (conditional months)
        ("pana Marka Maja doradcę ekonomicznego", "pana marka maja doradcę ekonomicznego"),
        ("moja siostra Maja", "moja siostra maja"),
        # Time via PELCRA – o jedenastej trzydzieści
        ("widzimy się we wtorek o jedenastej trzydzieści", "widzimy się we wtorek o 11:30"),
    ],
)
def test_bigos_real_samples(normalize, text, expected):
    assert normalize(text) == expected


def test_diacritics_preserved_on_dataset(normalize):
    assert normalize("później zabroniła") == "później zabroniła"
    assert normalize("żółć") == "żółć"
    out = normalize("Zażółć gęślą jaźń")
    assert "ą" in out and "ę" in out and "ł" in out and "ś" in out and "ź" in out and "ż" in out


def test_month_conditional_on_dataset(normalize):
    # isolated month stays, with day converts
    assert normalize("maja") == "maja"
    assert normalize("w maju") == "w maju"
    assert normalize("trzeciego maja") == "3. 5"
    assert normalize("pierwszego stycznia") == "1. 1"


def test_fraction_idempotent_dataset(normalize):
    assert normalize("jedna trzecia") == "1/3"
    assert normalize("1/3") == "1/3"
