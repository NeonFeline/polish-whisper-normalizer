"""Full pipeline – time → numbers → dates → cleanup."""

from __future__ import annotations

import datetime
import logging
import re

from .basic import remove_symbols
from .numbers import PolishNumberNormalizer
from .time import PolishTimeNormalizer
from .utils import contains_digit, normalize_vulgar_fractions, strip_diacritics, sub_if_present

logger = logging.getLogger(__name__)


# strict Roman numeral validation (1-3999) + conversion (issue 11)
_ROMAN_VALID_RE = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _roman_to_int(tok: str) -> int | None:
    """Return the value of a Roman numeral token, or None if invalid."""
    if not tok or _ROMAN_VALID_RE.match(tok) is None:
        return None
    total = 0
    prev = 0
    for ch in reversed(tok):
        val = _ROMAN_VALUES[ch]
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total if 1 <= total <= 3999 else None


def _roman_repl(m: re.Match[str]) -> str:
    val = _roman_to_int(m.group(1))
    return str(val) if val is not None else m.group(0)


# combining stems for glued percent adjectives ("dwudziestoprocentowy" -> 20%)
# (issue 14); ASCII variants derived for diacritic-less ASR
_PERCENT_STEMS: dict[str, int] = {
    "jedno": 1,
    "dwu": 2,
    "trzy": 3,
    "cztero": 4,
    "pięcio": 5,
    "sześcio": 6,
    "siedmio": 7,
    "ośmio": 8,
    "dziewięcio": 9,
    "dziesięcio": 10,
    "jedenasto": 11,
    "dwunasto": 12,
    "trzynasto": 13,
    "czternasto": 14,
    "piętnasto": 15,
    "szesnasto": 16,
    "siedemnasto": 17,
    "osiemnasto": 18,
    "dziewiętnasto": 19,
    "dwudziesto": 20,
    "trzydziesto": 30,
    "czterdziesto": 40,
    "pięćdziesięcio": 50,
    "sześćdziesięcio": 60,
    "siedemdziesięcio": 70,
    "osiemdziesięcio": 80,
    "dziewięćdziesięcio": 90,
    "stu": 100,
    "dwustu": 200,
}
_PERCENT_TENS_STEMS: tuple[tuple[str, int], ...] = (
    ("dziewięćdziesięcio", 90),
    ("siedemdziesięcio", 70),
    ("sześćdziesięcio", 60),
    ("pięćdziesięcio", 50),
    ("osiemdziesięcio", 80),
    ("czterdziesto", 40),
    ("trzydziesto", 30),
    ("dwudziesto", 20),
)
_PERCENT_UNIT_STEMS: dict[str, int] = {
    "jedno": 1,
    "dwu": 2,
    "trzy": 3,
    "cztero": 4,
    "pięcio": 5,
    "sześcio": 6,
    "siedmio": 7,
    "ośmio": 8,
    "dziewięcio": 9,
}


def _with_ascii_int(mapping: dict[str, int]) -> dict[str, int]:
    expanded: dict[str, int] = dict(mapping)
    for key, value in mapping.items():
        stripped = strip_diacritics(key)
        if stripped != key and stripped not in expanded:
            expanded[stripped] = value
    return expanded


_PERCENT_STEMS_ALL = _with_ascii_int(_PERCENT_STEMS)
_PERCENT_TENS_ALL = tuple(
    (stripped, val)
    for stem, val in _PERCENT_TENS_STEMS
    for stripped in {stem, strip_diacritics(stem)}
)
_PERCENT_UNITS_ALL = _with_ascii_int(_PERCENT_UNIT_STEMS)


def _percent_glued_repl(m: re.Match[str]) -> str:
    """Convert a glued percent adjective stem to digits + % (issue 14)."""
    stem = m.group(1)
    val = _PERCENT_STEMS_ALL.get(stem)
    if val is None:
        # tens + unit compounds ("dwudziestotrzy" -> 23)
        for tstem, tval in _PERCENT_TENS_ALL:
            if stem.startswith(tstem):
                uval = _PERCENT_UNITS_ALL.get(stem[len(tstem) :])
                if uval is not None:
                    val = tval + uval
                    break
    if val is None:
        return m.group(0)
    return f"{val}%"


class PolishTextNormalizer:
    # pre-compiled patterns shared across instances
    # NOTE: brackets (), [], <>, {} are punctuation – only the bracket
    # characters are dropped (via remove_symbols below), enclosed words kept.
    _WS_RE = re.compile(r"\s+")
    _IGNORE_RE = re.compile(r"\b(?:eee+|yyy+|hmm+|mhm+|mmm+|uh+|um+)\b")
    # invisible formatting chars (zero-width, bidi) are dropped (10)
    _FORMAT_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]+")
    # Roman numerals: all-uppercase + valid + len>=2, followed by a word
    # ("XI wiek" -> "11"); guards "i" (and), "Ci", "mi", lowercase (11)
    _ROMAN_RE = re.compile(r"\b([IVXLCDM]{2,})(?=\s+[a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ]{2,})")
    # Polish dot thousands separator ("60.000" -> "60000"); runs BEFORE
    # decimal comma conversion so "60,000" (decimal 60) is untouched (12)
    _THOUSANDS_DOT_RE = re.compile(r"(\d)\.(\d{3})(?!\d)")
    # hyphenated ordinal suffix after digits ("70-te" -> "70"); hyphenated
    # only – spaced "te" is a real word (17)
    _DIGIT_ORD_SUFFIX_RE = re.compile(
        r"\b(\d+)-(?:te|ta|to|ty|tą|tego|tej|tym|tych|tymi|szy|sza|sze)\b"
    )
    # letter-hyphen-digit is a hyphen, not a minus ("omega-3" -> "omega 3",
    # like digit-word "70-latek" -> "70 latek"); leading "-10" kept (17)
    _LETTER_HYPHEN_DIGIT_RE = re.compile(r"(?<=[a-ząćęłńóśźża-z])-(?=\d)")
    # unit full forms -> short, like bug 8 ul/nr/dr ("kilometrów" -> "km") (14)
    _UNIT_KM_RE = re.compile(r"\bkilometr\w*\b")
    # percent adjectives -> percent ("20-procentowy" -> "20%") (14)
    _PERCENT_ADJ_DIGIT_RE = re.compile(r"\b(\d+)\s*-?\s*procentow\w*\b")
    _PERCENT_GLUED_RE = re.compile(r"\b([a-ząćęłńóśźż]+?)procentow\w*\b")
    _PERCENT_ADJ_WORD_RE = re.compile(r"\bprocentow\w*\b")
    # "pół żartem/pół serio" are idioms, not 0.5 (15)
    _POL_IDIOM_RE = re.compile(r"\b(pół|pol)\s+(żartem|serio|zartem)\b")
    # magnitude abbreviations, like bug 8 ("50 tys." -> "50 tysięcy") (13)
    _ABBREV_TYS_RE = re.compile(r"\btys\.?(?=\s|$)")
    _ABBREV_MLN_RE = re.compile(r"\bmln\.?(?=\s|$)")
    _ABBREV_MLD_RE = re.compile(r"\bmld\.?(?=\s|$)")
    _SENTENCE_PERIOD_RE = re.compile(r"(?<![\d.])\.(?!\.)([^0-9]|$)")
    _ELLIPSIS_RE = re.compile(r"(?:\s*\.){2,}\s*|\s*…\s*")
    _NUMBER_SEPARATOR_RE = re.compile(r"[,;!?—–]")
    _TRAILING_DIGIT_PERIOD_RE = re.compile(r"(?<=\d)\.(?=\s|$)")
    _R_ABBREV_RE = re.compile(r"\br\.?\s*(?=\d{3,4}\b)")
    _DECIMAL_COMMA_RE = re.compile(r"(\d),(\d)")
    _MONTH_DAY_RE = re.compile(r"(\d+)\.?\s+([a-ząćęłńóśźż]+)\b")
    _FULL_DATE_ROKU_BEFORE_RE = re.compile(r"(\d+)\.\s*(\d+)[.\s]+\s*(?:roku|r\.?)\s+(\d+)\.?")
    _FULL_DATE_ROKU_AFTER_RE = re.compile(r"(\d+)\.\s*(\d+)[.\s]+(\d+)\.?\s+(?:roku|r\.?)\b")
    _FULL_DATE_RE = re.compile(r"(\d+)\.\s*(\d+)[.\s]+(\d{4})\.?")
    _TRAILING_R_RE1 = re.compile(r"(\d{2}\.\d{2}\.\d{4})\s+r\.?\b")
    _TRAILING_R_RE2 = re.compile(r"(\d{2}\.\d{2}\.\d{4})r\.?\b")
    # standalone year with trailing "r."/"roku" ("1860 r." -> "1860")
    _YEAR_TRAILING_R_RE = re.compile(r"\b(\d{3,4})\s+(?:r\.?|roku)\b")
    # day.month without year requires a space ("5. 5" -> "05.05") to avoid
    # turning decimals ("2.5", "3.14") into dates; full dates with 4-digit
    # year allow no space ("5.5.2026" -> "05.05.2026", safe: no decimal has year)
    _DAY_MONTH_RE = re.compile(r"(\d+)\.\s+(\d+)\b(?!\.)")
    _PERCENT_GLUE_RE = re.compile(r"(\d)\s+%")
    _PERCENT_RE = re.compile(r"([^0-9])%")
    _COLON_RE = re.compile(r"(?<!\d):|:(?!\d)")
    _SIGN_RE = re.compile(r"[-+](?!\d)")
    _WANTS_R_RE = re.compile(r"\br\.?\b")
    # Polish abbreviations <-> spoken forms (bug 8): contract full forms to
    # short (gender/case-neutral: "tak zwany/tak zwana" -> "tzw") except
    # "godz." which expands to "godzina" to enable time conversion.
    _ABBREV_TZW_RE = re.compile(r"\btak\s+zwan\w*\b")
    _ABBREV_NP_RE = re.compile(r"\bna\s+przyk[lł]ad\b")
    _ABBREV_ITD_RE = re.compile(r"\bi\s+tak\s+dalej\b")
    _ABBREV_ITP_RE = re.compile(r"\bi\s+(?:temu|tym)\s+podobn\w*\b")
    _ABBREV_MIN_SHORT_RE = re.compile(r"\bm\s*\.\s*in\s*\.?(?=\s|$)")
    # "tj."/"tzn." (i.e.) expand to full; "to jest"/"to znaczy" (it is)
    # stay full (don't contract copula "to jest złoty" -> must stay).
    _ABBREV_TJ_SHORT_RE = re.compile(r"\btj\s*\.?(?=\s|$)")
    _ABBREV_TZN_SHORT_RE = re.compile(r"\btzn\s*\.?(?=\s|$)")
    _ABBREV_TZW_SHORT_RE = re.compile(r"\btzw\s*\.?(?=\s|$)")
    _ABBREV_NP_SHORT_RE = re.compile(r"\bnp\s*\.?(?=\s|$)")
    _ABBREV_ITD_SHORT_RE = re.compile(r"\bitd\s*\.?(?=\s|$)")
    _ABBREV_ITP_SHORT_RE = re.compile(r"\bitp\s*\.?(?=\s|$)")
    _ABBREV_GODZ_RE = re.compile(r"\bgodz\.(?=\s|$)|\bgodz\b")
    _ABBREV_UL_FULL_RE = re.compile(r"\bulic(?:a|y|e|ę|o|ą|om|ami|ach)\b")
    _ABBREV_UL_SHORT_RE = re.compile(r"\bul\.(?=\s|$)")
    _ABBREV_NR_FULL_RE = re.compile(r"\bnumer(?:u|owi|em|ze|y|ów|om|ami|ach)?\b")
    _ABBREV_NR_SHORT_RE = re.compile(r"\bnr\.(?=\s|$)")
    _ABBREV_DR_FULL_RE = re.compile(r"\bdoktor(?:a|owi|em|ze|zy|ów|om|ami|ach)?\b")
    _ABBREV_DR_SHORT_RE = re.compile(r"\bdr\.(?=\s|$)")
    _ABBREV_PROF_FULL_RE = re.compile(r"\bprofesor(?:a|owi|em|ze|owie|ów|om|ami|ach)?\b")
    _ABBREV_PROF_SHORT_RE = re.compile(r"\bprof\.(?=\s|$)")
    _ABBREV_OK_RE = re.compile(
        r"\bok\.\s*(?=\d|\b(?:godz|południ|poludni|północ|polnoc|st|lut|mar|kwi|maj|cze|lip|sie|wrz|paź|paz|lis|gru|jed|dw|trz|czt|pię|pie|sze|sie|osi|dzi|sto|tys|mil|bil|pół|pol)\w*)"
    )

    def __init__(self, date_format: str = "{day:02d}.{month:02d}.{year}", **kwargs: object) -> None:
        """
        Args:
            date_format: How to render full dates (day month year). Supports
                Python format with {day}, {month}, {year} (e.g. "{day:02d}.{month:02d}.{year}"
                or "{day}/{month}/{year}") and strftime with %d/%m/%Y
                (e.g. "%d.%m.%Y", "%Y-%m-%d", "%d.%m.%Yr.").
                Default "{day:02d}.{month:02d}.{year}" -> "05.05.2026".
        """
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        self.ignore_patterns = r"\b(?:eee+|yyy+|hmm+|mhm+|mmm+|uh+|um+)\b"
        self.standardize_numbers = PolishNumberNormalizer()
        self.standardize_time = PolishTimeNormalizer()
        self.date_format: str = date_format

    def _format_date(self, day: int, month: int, year: int) -> str:
        fmt = self.date_format
        has_brace = "{" in fmt and "}" in fmt
        has_percent = "%" in fmt

        if has_brace:
            try:
                return fmt.format(day=day, month=month, year=year)
            except Exception as exc:
                logger.debug("brace date_format failed for %r: %s", fmt, exc)

        if has_percent:
            try:
                out = datetime.datetime(year, month, day).strftime(fmt)
                # unknown/invalid strftime codes leak "%" (e.g. "%Q" -> "%Q");
                # fall back to the default instead of leaking the format string
                if "%" in out:
                    raise ValueError(f"strftime leaked % for {fmt!r} -> {out!r}")
                return out
            except Exception as exc:
                logger.debug("strftime date_format failed for %r: %s", fmt, exc)

        return f"{day:02d}.{month:02d}.{year}"

    def _month_number(self, word: str) -> str | None:
        # direct map is fastest
        if word in self.standardize_numbers.month_lemmas:
            return self.standardize_numbers.month_lemmas[word]
        for base, pos in self.standardize_numbers.lemmatizer.analyse(word):
            if pos == "subst" and base in self.standardize_numbers.month_lemmas:
                return self.standardize_numbers.month_lemmas[base]
        # fallback for ASCII-folded month (e.g. "styczen" -> "styczeń")
        mapped = self.standardize_numbers._declined_ascii_map.get(word)
        if mapped is not None and mapped in self.standardize_numbers.month_lemmas:
            return self.standardize_numbers.month_lemmas[mapped]
        return None

    def __call__(self, s: str) -> str:
        if not isinstance(s, str):
            raise TypeError(f"Expected str, got {type(s).__name__}")
        # Roman numerals before lowercasing (case guard: only ALL-UPPERCASE
        # "XI wiek" -> "11 wiek"; "i", "Ci", "mi", lowercase stay) (11).
        # The lower() compare fuses the guard with already-needed work.
        if s.lower() != s:
            s = self._ROMAN_RE.sub(_roman_repl, s)
        s = normalize_vulgar_fractions(s.lower())
        s = self._FORMAT_RE.sub("", s)  # invisible formatting chars (10)
        # perf: literal guards below are necessary conditions — skipping the
        # scan never changes output (sub_if_present), ~85% of scans are no-ops
        if "-" in s:
            s = re.sub(r"\b(pół|pol|ćwierć|cwierc)-\s*([a-ząćęłńóśźż]+)\b", r"\1\2", s)
        # spaced "pół żartem/pół serio" are idioms, not 0.5 (15)
        s = sub_if_present(s, self._POL_IDIOM_RE, r"\1\2", "żartem", "serio", "zartem")

        # Polish abbreviations <-> spoken forms (bug 8): normalize both sides
        # to the same token ("tak zwany" <-> "tzw" -> "tzw", "np." -> "np",
        # "tj." -> "to jest" to preserve copula "to jest złoty").
        s = sub_if_present(s, self._ABBREV_TZW_RE, "tzw", "zwan")
        s = sub_if_present(s, self._ABBREV_NP_RE, "np", "przyk")
        s = sub_if_present(s, self._ABBREV_ITD_RE, "itd", "dalej")
        s = sub_if_present(s, self._ABBREV_ITP_RE, "itp", "podobn")
        s = sub_if_present(s, self._ABBREV_MIN_SHORT_RE, "między innymi", "in")
        s = sub_if_present(s, self._ABBREV_TJ_SHORT_RE, "to jest", "tj")
        s = sub_if_present(s, self._ABBREV_TZN_SHORT_RE, "to znaczy", "tzn")
        s = sub_if_present(s, self._ABBREV_TZW_SHORT_RE, "tzw", "tzw")
        s = sub_if_present(s, self._ABBREV_NP_SHORT_RE, "np", "np")
        s = sub_if_present(s, self._ABBREV_ITD_SHORT_RE, "itd", "itd")
        s = sub_if_present(s, self._ABBREV_ITP_SHORT_RE, "itp", "itp")
        s = sub_if_present(s, self._ABBREV_GODZ_RE, "godzina", "godz")
        s = sub_if_present(s, self._ABBREV_UL_FULL_RE, "ul", "ulic")
        s = sub_if_present(s, self._ABBREV_UL_SHORT_RE, "ul", "ul.")
        s = sub_if_present(s, self._ABBREV_NR_FULL_RE, "nr", "numer")
        s = sub_if_present(s, self._ABBREV_NR_SHORT_RE, "nr", "nr.")
        s = sub_if_present(s, self._ABBREV_DR_FULL_RE, "dr", "doktor")
        s = sub_if_present(s, self._ABBREV_DR_SHORT_RE, "dr", "dr.")
        s = sub_if_present(s, self._ABBREV_PROF_FULL_RE, "prof", "profesor")
        s = sub_if_present(s, self._ABBREV_PROF_SHORT_RE, "prof", "prof.")
        s = sub_if_present(s, self._ABBREV_OK_RE, "około", "ok.")
        # magnitude abbreviations, like bug 8 (13)
        s = sub_if_present(s, self._ABBREV_TYS_RE, "tysięcy", "tys")
        s = sub_if_present(s, self._ABBREV_MLN_RE, "milionów", "mln")
        s = sub_if_present(s, self._ABBREV_MLD_RE, "miliardów", "mld")
        # unit full forms -> short, like bug 8 ul/nr/dr (14)
        s = sub_if_present(s, self._UNIT_KM_RE, "km", "kilometr")
        # percent adjectives -> percent (14): digit ("20-procentowy" -> "20%"),
        # glued ("dwudziestoprocentowy" -> "20%"), spaced word
        # ("dwadzieścia procentowy" -> "dwadzieścia procent" -> "20%")
        s = sub_if_present(s, self._PERCENT_ADJ_DIGIT_RE, r"\1%", "procentow")
        s = sub_if_present(s, self._PERCENT_GLUED_RE, _percent_glued_repl, "procentow")
        s = sub_if_present(s, self._PERCENT_ADJ_WORD_RE, "procent", "procentow")
        # hyphenated ordinal suffix after digits ("70-te" -> "70", hyphenated
        # only) + letter-hyphen-digit split ("omega-3" -> "omega 3") (17)
        if "-" in s:
            s = self._DIGIT_ORD_SUFFIX_RE.sub(r"\1", s)
            s = self._LETTER_HYPHEN_DIGIT_RE.sub(" ", s)

        s = sub_if_present(s, self._IGNORE_RE, "", "eee", "yyy", "hmm", "mhm", "mmm", "uh", "um")

        # expand the year abbreviation "r." / "r" to "roku" (w r. 1860 -> w roku 1860)
        s = self._R_ABBREV_RE.sub("roku ", s)

        # remove sentence periods before digits are introduced by time/number
        # normalization; keep decimals ("3.14"), ordinal markers ("21.") and
        # ellipses (handled as boundaries below). A boundary "." is emitted so
        # separate numerals ("10. 500") are not merged, then dropped by numbers.
        s = self._SENTENCE_PERIOD_RE.sub(r" . \1", s)
        s = sub_if_present(s, self._ELLIPSIS_RE, " . ", "..", "…")

        s = self.standardize_time(s)

        # Polish dot thousands separator ("60.000" -> "60000", "1.000.000" ->
        # "1000000"); loop for stacked groups. BEFORE decimal comma so
        # "60,000" (Polish decimal 60) is untouched (12)
        if "." in s and contains_digit(s):
            _prev = None
            while _prev != s:
                _prev = s
                s = self._THOUSANDS_DOT_RE.sub(r"\1\2", s)

        s = sub_if_present(
            s, self._DECIMAL_COMMA_RE, r"\1.\2", ","
        )  # Polish decimal comma -> point
        # other punctuation separates numerals ("10, 500" -> "10 500") instead
        # of being erased (which would merge them into "10500")
        s = sub_if_present(s, self._NUMBER_SEPARATOR_RE, " . ", ",", ";", "!", "?", "—", "–")
        s = remove_symbols(
            s, keep=".:/%$€£¢+-"
        )  # keep numeric/time/sign/currency symbols + fraction slash
        # Strip non-time colons before number conversion so "drugi:" -> "drugi"
        # converts to "2" (bug 6b: punctuation suppressed ordinal conversion).
        # _COLON_RE preserves digit:digit times ("18:30").
        s = sub_if_present(s, self._COLON_RE, " ", ":")

        s = self.standardize_numbers(s)

        # conditional month mapping: only when preceded by day (with or without dot)
        # ("3. maja" -> "3. 5", "5 maja" -> "5. 5", but "maja" alone or "w maju" stays)
        # perf: date patterns all need a digit in the current string
        has_digit = contains_digit(s)

        def _date_repl(m: re.Match[str]) -> str:
            day = m.group(1)
            month_word = m.group(2)
            month_num = self._month_number(month_word)
            if month_num:
                return f"{day}. {month_num}"
            return m.group(0)

        if has_digit:
            s = self._MONTH_DAY_RE.sub(_date_repl, s)

        # full date formatting: "5. 5 roku 2026." -> "05.05.2026", "5. 5 2026" -> "05.05.2026" (uniform, no r)
        # digit dates without spaces ("5.5.2026") also converge; month 1-12
        # guard avoids turning decimals ("3.14", month 14) into dates.
        def _full_date_roku_before(m: re.Match[str]) -> str:
            day, month, year = m.group(1), m.group(2), m.group(3)
            if not 1 <= int(month) <= 12:
                return m.group(0)
            return self._format_date(int(day), int(month), int(year))

        def _full_date_roku_after(m: re.Match[str]) -> str:
            day, month, year = m.group(1), m.group(2), m.group(3)
            if not 1 <= int(month) <= 12:
                return m.group(0)
            return self._format_date(int(day), int(month), int(year))

        def _full_date(m: re.Match[str]) -> str:
            day, month, year = m.group(1), m.group(2), m.group(3)
            if not 1 <= int(month) <= 12:
                return m.group(0)
            return self._format_date(int(day), int(month), int(year))

        def _day_month(m: re.Match[str]) -> str:
            day, month = int(m.group(1)), int(m.group(2))
            # validate to avoid decimals ("3.14" month 14) becoming dates
            if not 1 <= month <= 12 or not 1 <= day <= 31:
                return m.group(0)
            return f"{day:02d}.{month:02d}"

        # order matters: most specific first (handle roku and r. uniformly)
        if has_digit:
            s = self._FULL_DATE_ROKU_BEFORE_RE.sub(_full_date_roku_before, s)
            s = self._FULL_DATE_ROKU_AFTER_RE.sub(_full_date_roku_after, s)
            s = self._FULL_DATE_RE.sub(_full_date, s)
        # uniform output: strip trailing r/r. only if date_format does not request it
        # detect literal 'r' in format (e.g. "%d.%m.%Yr." or "{day} r.")
        _wants_r = (
            bool(self._WANTS_R_RE.search(self.date_format.lower())) if self.date_format else False
        )
        # legacy check for formats ending with r/r.
        if not _wants_r:
            _wants_r = (
                self.date_format.strip().lower().endswith("r.")
                or self.date_format.strip().lower().endswith(" r")
                or " r." in self.date_format.lower()
            )
        if not _wants_r and has_digit:
            s = self._TRAILING_R_RE1.sub(r"\1", s)
            s = self._TRAILING_R_RE2.sub(r"\1", s)
            # standalone year with trailing r/roku ("1860 r." -> "1860")
            s = self._YEAR_TRAILING_R_RE.sub(r"\1", s)
        # day month without year -> 5. 5 -> 05.05 (as requested)
        # only when month is not ordinal (no trailing dot) – avoids "1. 2." -> "01.02."
        if has_digit:
            s = self._DAY_MONTH_RE.sub(_day_month, s)

        # digits may have been introduced by number conversion – recheck
        has_digit = contains_digit(s)

        # ordinal dots are dropped at the end of the pipeline; strip the same
        # dot from already-digit input ("15." -> "15") for consistency
        if "." in s and has_digit:
            s = self._TRAILING_DIGIT_PERIOD_RE.sub("", s)

        # glue digit-space-percent ("5 %" -> "5%") before dropping stray "%"
        # ("a%b" -> "a b", but "5%" stays)
        if "%" in s and has_digit:
            s = self._PERCENT_GLUE_RE.sub(r"\1%", s)
        # remove leftover symbols that are not part of a number/time
        s = sub_if_present(s, self._PERCENT_RE, r"\1 ", "%")
        s = sub_if_present(s, self._COLON_RE, " ", ":")
        s = sub_if_present(s, self._SIGN_RE, " ", "-", "+")

        s = self._WS_RE.sub(" ", s)  # replace successive whitespaces with a space
        return s.strip()
