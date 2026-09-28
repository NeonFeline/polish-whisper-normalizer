"""Full pipeline – time → numbers → dates → cleanup."""

from __future__ import annotations

import datetime
import logging
import re

from .basic import remove_symbols
from .numbers import PolishNumberNormalizer
from .time import PolishTimeNormalizer

logger = logging.getLogger(__name__)


class PolishTextNormalizer:
    # pre-compiled patterns shared across instances
    _BRACKETS_RE = re.compile(r"<[^>]*>|\[[^\]]*\]")
    _PAREN_RE = re.compile(r"\([^)]*\)")
    _WS_RE = re.compile(r"\s+")
    _IGNORE_RE = re.compile(r"\b(?:eee+|yyy+|hmm+|mhm+|mmm+|uh+|um+)\b")
    _SENTENCE_PERIOD_RE = re.compile(r"(?<![\d.])\.(?!\.)([^0-9]|$)")
    _ELLIPSIS_RE = re.compile(r"(?:\s*\.){2,}\s*|\s*…\s*")
    _NUMBER_SEPARATOR_RE = re.compile(r"[,;!?—–]")
    _TRAILING_DIGIT_PERIOD_RE = re.compile(r"(?<=\d)\.(?=\s|$)")
    _R_ABBREV_RE = re.compile(r"\br\.?\s*(?=\d{3,4}\b)")
    _DECIMAL_COMMA_RE = re.compile(r"(\d),(\d)")
    _MONTH_DAY_RE = re.compile(r"(\d+)\.?\s+([a-ząćęłńóśźż]+)\b")
    _FULL_DATE_ROKU_BEFORE_RE = re.compile(r"(\d+)\.\s+(\d+)\s+(?:roku|r\.?)\s+(\d+)\.?")
    _FULL_DATE_ROKU_AFTER_RE = re.compile(r"(\d+)\.\s+(\d+)\s+(\d+)\.?\s+(?:roku|r\.?)\b")
    _FULL_DATE_RE = re.compile(r"(\d+)\.\s+(\d+)\s+(\d{4})\.?")
    _TRAILING_R_RE1 = re.compile(r"(\d{2}\.\d{2}\.\d{4})\s+r\.?\b")
    _TRAILING_R_RE2 = re.compile(r"(\d{2}\.\d{2}\.\d{4})r\.?\b")
    _DAY_MONTH_RE = re.compile(r"(\d+)\.\s+(\d+)\b(?!\.)")
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
    _ABBREV_UL_FULL_RE = re.compile(r"\bulic(?:a|y|e|ę|o|om|ami|ach)\b")
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
                return datetime.datetime(year, month, day).strftime(fmt)
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
        s = s.lower()

        # Polish abbreviations <-> spoken forms (bug 8): normalize both sides
        # to the same token ("tak zwany" <-> "tzw" -> "tzw", "np." -> "np",
        # "tj." -> "to jest" to preserve copula "to jest złoty").
        s = self._ABBREV_TZW_RE.sub("tzw", s)
        s = self._ABBREV_NP_RE.sub("np", s)
        s = self._ABBREV_ITD_RE.sub("itd", s)
        s = self._ABBREV_ITP_RE.sub("itp", s)
        s = self._ABBREV_MIN_SHORT_RE.sub("między innymi", s)
        s = self._ABBREV_TJ_SHORT_RE.sub("to jest", s)
        s = self._ABBREV_TZN_SHORT_RE.sub("to znaczy", s)
        s = self._ABBREV_TZW_SHORT_RE.sub("tzw", s)
        s = self._ABBREV_NP_SHORT_RE.sub("np", s)
        s = self._ABBREV_ITD_SHORT_RE.sub("itd", s)
        s = self._ABBREV_ITP_SHORT_RE.sub("itp", s)
        s = self._ABBREV_GODZ_RE.sub("godzina", s)
        s = self._ABBREV_UL_FULL_RE.sub("ul", s)
        s = self._ABBREV_UL_SHORT_RE.sub("ul", s)
        s = self._ABBREV_NR_FULL_RE.sub("nr", s)
        s = self._ABBREV_NR_SHORT_RE.sub("nr", s)
        s = self._ABBREV_DR_FULL_RE.sub("dr", s)
        s = self._ABBREV_DR_SHORT_RE.sub("dr", s)
        s = self._ABBREV_PROF_FULL_RE.sub("prof", s)
        s = self._ABBREV_PROF_SHORT_RE.sub("prof", s)
        s = self._ABBREV_OK_RE.sub("około", s)

        s = self._BRACKETS_RE.sub("", s)  # remove words between brackets
        s = self._PAREN_RE.sub("", s)  # remove words between parenthesis
        s = self._IGNORE_RE.sub("", s)

        # expand the year abbreviation "r." / "r" to "roku" (w r. 1860 -> w roku 1860)
        s = self._R_ABBREV_RE.sub("roku ", s)

        # remove sentence periods before digits are introduced by time/number
        # normalization; keep decimals ("3.14"), ordinal markers ("21.") and
        # ellipses (handled as boundaries below). A boundary "." is emitted so
        # separate numerals ("10. 500") are not merged, then dropped by numbers.
        s = self._SENTENCE_PERIOD_RE.sub(r" . \1", s)
        s = self._ELLIPSIS_RE.sub(" . ", s)

        s = self.standardize_time(s)

        s = self._DECIMAL_COMMA_RE.sub(r"\1.\2", s)  # Polish decimal comma -> point
        # other punctuation separates numerals ("10, 500" -> "10 500") instead
        # of being erased (which would merge them into "10500")
        s = self._NUMBER_SEPARATOR_RE.sub(" . ", s)
        s = remove_symbols(
            s, keep=".:/%$€£¢+-"
        )  # keep numeric/time/sign/currency symbols + fraction slash
        # Strip non-time colons before number conversion so "drugi:" -> "drugi"
        # converts to "2" (bug 6b: punctuation suppressed ordinal conversion).
        # _COLON_RE preserves digit:digit times ("18:30").
        s = self._COLON_RE.sub(" ", s)

        s = self.standardize_numbers(s)

        # conditional month mapping: only when preceded by day (with or without dot)
        # ("3. maja" -> "3. 5", "5 maja" -> "5. 5", but "maja" alone or "w maju" stays)
        def _date_repl(m: re.Match[str]) -> str:
            day = m.group(1)
            month_word = m.group(2)
            month_num = self._month_number(month_word)
            if month_num:
                return f"{day}. {month_num}"
            return m.group(0)

        s = self._MONTH_DAY_RE.sub(_date_repl, s)

        # full date formatting: "5. 5 roku 2026." -> "05.05.2026", "5. 5 2026" -> "05.05.2026" (uniform, no r)
        def _full_date_roku_before(m: re.Match[str]) -> str:
            day, month, year = m.group(1), m.group(2), m.group(3)
            return self._format_date(int(day), int(month), int(year))

        def _full_date_roku_after(m: re.Match[str]) -> str:
            day, month, year = m.group(1), m.group(2), m.group(3)
            return self._format_date(int(day), int(month), int(year))

        def _full_date(m: re.Match[str]) -> str:
            day, month, year = m.group(1), m.group(2), m.group(3)
            return self._format_date(int(day), int(month), int(year))

        def _day_month(m: re.Match[str]) -> str:
            day, month = m.group(1), m.group(2)
            return f"{int(day):02d}.{int(month):02d}"

        # order matters: most specific first (handle roku and r. uniformly)
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
        if not _wants_r:
            s = self._TRAILING_R_RE1.sub(r"\1", s)
            s = self._TRAILING_R_RE2.sub(r"\1", s)
        # day month without year -> 5. 5 -> 05.05 (as requested)
        # only when month is not ordinal (no trailing dot) – avoids "1. 2." -> "01.02."
        s = self._DAY_MONTH_RE.sub(_day_month, s)

        # ordinal dots are dropped at the end of the pipeline; strip the same
        # dot from already-digit input ("15." -> "15") for consistency
        s = self._TRAILING_DIGIT_PERIOD_RE.sub("", s)

        # remove leftover symbols that are not part of a number/time
        s = self._PERCENT_RE.sub(r"\1 ", s)
        s = self._COLON_RE.sub(" ", s)
        s = self._SIGN_RE.sub(" ", s)

        s = self._WS_RE.sub(" ", s)  # replace successive whitespaces with a space
        return s.strip()
