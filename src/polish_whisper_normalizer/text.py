"""Full pipeline – time → numbers → dates → cleanup."""

from __future__ import annotations

import datetime
import re
import warnings

from .basic import remove_symbols
from .numbers import PolishNumberNormalizer
from .time import PolishTimeNormalizer


class PolishTextNormalizer:
    # pre-compiled patterns shared across instances
    _BRACKETS_RE = re.compile(r"<[^>]*>|\[[^\]]*\]")
    _PAREN_RE = re.compile(r"\([^)]*\)")
    _WS_RE = re.compile(r"\s+")

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
            if set(kwargs.keys()) == {"date_format"} and isinstance(kwargs["date_format"], str):
                warnings.warn(
                    "Passing date_format via **kwargs is deprecated, use positional arg",
                    DeprecationWarning,
                    stacklevel=2,
                )
                date_format = kwargs["date_format"]  # type: ignore[assignment]
            else:
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
        # prefer explicit brace-format; if both present try brace first
        if has_brace:
            try:
                return fmt.format(day=day, month=month, year=year)
            except Exception:
                if not has_percent:
                    return f"{day:02d}.{month:02d}.{year}"
        if has_percent:
            try:
                return datetime.datetime(year, month, day).strftime(fmt)
            except Exception:
                pass
        # fallback: brace format if not tried
        if not has_brace:
            try:
                return fmt.format(day=day, month=month, year=year)
            except Exception:
                pass
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

        s = self._BRACKETS_RE.sub("", s)  # remove words between brackets
        s = self._PAREN_RE.sub("", s)  # remove words between parenthesis
        s = re.sub(self.ignore_patterns, "", s)

        # remove sentence periods before digits are introduced by time/number
        # normalization; keep decimals ("3.14") and ordinal markers ("21.")
        s = re.sub(r"(?<!\d)\.([^0-9]|$)", r" \1", s)

        s = self.standardize_time(s)

        s = re.sub(r"(\d),(\d)", r"\1.\2", s)  # Polish decimal comma -> point
        s = remove_symbols(
            s, keep=".:/%$€£¢+-"
        )  # keep numeric/time/sign/currency symbols + fraction slash

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

        s = re.sub(r"(\d+)\.?\s+([a-ząćęłńóśźż]+)\b", _date_repl, s)

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
        s = re.sub(r"(\d+)\.\s+(\d+)\s+(?:roku|r\.?)\s+(\d+)\.?", _full_date_roku_before, s)
        s = re.sub(r"(\d+)\.\s+(\d+)\s+(\d+)\.?\s+(?:roku|r\.?)\b", _full_date_roku_after, s)
        s = re.sub(r"(\d+)\.\s+(\d+)\s+(\d{4})\.?", _full_date, s)
        # uniform output: strip trailing r/r. only if date_format does not request it
        # detect literal 'r' in format (e.g. "%d.%m.%Yr." or "{day} r.")
        _wants_r = (
            bool(re.search(r"\br\.?\b", self.date_format.lower())) if self.date_format else False
        )
        # legacy check for formats ending with r/r.
        if not _wants_r:
            _wants_r = (
                self.date_format.strip().lower().endswith("r.")
                or self.date_format.strip().lower().endswith(" r")
                or " r." in self.date_format.lower()
            )
        if not _wants_r:
            s = re.sub(r"(\d{2}\.\d{2}\.\d{4})\s+r\.?\b", r"\1", s)
            s = re.sub(r"(\d{2}\.\d{2}\.\d{4})r\.?\b", r"\1", s)
        # day month without year -> 5. 5 -> 05.05 (as requested)
        # only when month is not ordinal (no trailing dot) – avoids "1. 2." -> "01.02."
        s = re.sub(r"(\d+)\.\s+(\d+)\b(?!\.)", _day_month, s)

        # remove leftover symbols that are not part of a number/time
        s = re.sub(r"([^0-9])%", r"\1 ", s)
        s = re.sub(r"(?<!\d):|:(?!\d)", " ", s)
        s = re.sub(r"[-+](?!\d)", " ", s)

        s = self._WS_RE.sub(" ", s)  # replace successive whitespaces with a space
        return s.strip()
