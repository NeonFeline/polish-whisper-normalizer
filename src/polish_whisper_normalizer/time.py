"""Spoken time → HH:MM normalization.

Delegates cardinal lexicon building to Morfeusz via shared helpers; ASCII
variants are generated from ``utils`` so diacritic-less ASR is handled
without duplicating dictionaries.
"""

from __future__ import annotations

import copy
import re
import threading
import uuid

from .utils import with_ascii_variants


class PolishTimeNormalizer:
    """
    Convert spoken times into HH:MM format:

    - "piąta trzydzieści" -> "5:30"
    - "dwudziesta piętnaście" -> "20:15"
    - "wpół do ósmej" -> "7:30"
    - "za piętnaście ósma" -> "7:45"
    - "piętnaście po piątej" -> "5:15"
    - "godzina piętnasta trzydzieści" -> "15:30"
    - "północ" -> "0:00", "południe" -> "12:00"
    """

    # class-level cache: built once, reused by all instances
    _CACHE: dict[str, object] | None = None
    _CACHE_LOCK = threading.Lock()

    def __init__(self) -> None:
        if PolishTimeNormalizer._CACHE is not None:
            # deepcopy mutable containers to avoid cross-instance mutation
            cached = PolishTimeNormalizer._CACHE
            for key, value in cached.items():
                if isinstance(value, dict | set | list):
                    self.__dict__[key] = copy.deepcopy(value)
                elif hasattr(value, "pattern"):  # compiled regex
                    self.__dict__[key] = value
                else:
                    self.__dict__[key] = value
            return

        self.hours = {
            "pierwsza": 1,
            "druga": 2,
            "trzecia": 3,
            "czwarta": 4,
            "piąta": 5,
            "szósta": 6,
            "siódma": 7,
            "ósma": 8,
            "dziewiąta": 9,
            "dziesiąta": 10,
            "jedenasta": 11,
            "dwunasta": 12,
            "trzynasta": 13,
            "czternasta": 14,
            "piętnasta": 15,
            "szesnasta": 16,
            "siedemnasta": 17,
            "osiemnasta": 18,
            "dziewiętnasta": 19,
            "dwudziesta": 20,
            "dwudziesta pierwsza": 21,
            "dwudziesta druga": 22,
            "dwudziesta trzecia": 23,
            "dwudziesta czwarta": 24,
        }
        self.hours_gen = {
            "pierwszej": 1,
            "drugiej": 2,
            "trzeciej": 3,
            "czwartej": 4,
            "piątej": 5,
            "szóstej": 6,
            "siódmej": 7,
            "ósmej": 8,
            "dziewiątej": 9,
            "dziesiątej": 10,
            "jedenastej": 11,
            "dwunastej": 12,
            "trzynastej": 13,
            "czternastej": 14,
            "piętnastej": 15,
            "szesnastej": 16,
            "siedemnastej": 17,
            "osiemnastej": 18,
            "dziewiętnastej": 19,
            "dwudziestej": 20,
            "dwudziestej pierwszej": 21,
            "dwudziestej drugiej": 22,
            "dwudziestej trzeciej": 23,
        }

        self.minutes = self._build_minutes()
        # expand with ASCII variants for diacritic-less ASR
        self.hours = with_ascii_variants(self.hours)
        self.hours_gen = with_ascii_variants(self.hours_gen)
        self.minutes = with_ascii_variants(self.minutes)

        hours_alt = self._alternation(self.hours)
        hours_gen_alt = self._alternation(self.hours_gen)
        minutes_alt = self._alternation(self.minutes)
        # literals with diacritics also need ASCII variants (diacritic-less ASR)
        wpol_pat = r"(?:wpół|wpol)"
        polnoc_pat = r"(?:północ|polnoc)"
        poludnie_pat = r"(?:południe|poludnie)"

        self._polnoc_pat = polnoc_pat
        self._poludnie_pat = poludnie_pat

        self._re_wpol = re.compile(r"\b" + wpol_pat + r"\s+do\s+(" + hours_gen_alt + r")\b")
        self._re_za = re.compile(r"\bza\s+(" + minutes_alt + r")\s+(" + hours_alt + r")\b")
        self._re_po = re.compile(r"\b(" + minutes_alt + r")\s+po\s+(" + hours_gen_alt + r")\b")
        self._re_godzina_min = re.compile(
            r"\bgodzina\s+(" + hours_alt + r")\s+(" + minutes_alt + r")\b"
        )
        self._re_godzina = re.compile(r"\bgodzina\s+(" + hours_alt + r")\b")
        self._re_hour_min = re.compile(r"\b(" + hours_alt + r")\s+(" + minutes_alt + r")\b")
        self._re_hour_gen_min = re.compile(r"\b(" + hours_gen_alt + r")\s+(" + minutes_alt + r")\b")
        # "o piątej" -> "o 5:00" (genitive hour, not followed by a word)
        self._re_o_hour = re.compile(r"\bo\s+(" + hours_gen_alt + r")\b(?!\s*[a-ząćęłńóśźż])")
        # hour + time-of-day marker ("piąta rano" -> "5:00 rano")
        self._re_hour_rano = re.compile(r"\b(" + hours_alt + r")\s+rano\b")
        self._re_o_hour_rano = re.compile(r"\bo\s+(" + hours_gen_alt + r")\s+rano\b")
        # variants with an explicit "minut(ę/y)" word – include ASCII "minutę" -> "minute"
        self._re_za_minut = re.compile(
            r"\bza\s+(" + minutes_alt + r")\s+minut(?:a|ę|e|y)?\s+(" + hours_alt + r")\b"
        )
        self._re_po_minut = re.compile(
            r"\b(" + minutes_alt + r")\s+minut(?:a|ę|e|y)?\s+po\s+(" + hours_gen_alt + r")\b"
        )
        # "od piątej do szóstej" -> "od 5:00 do 6:00"
        self._re_range = re.compile(
            r"\bod\s+(" + hours_gen_alt + r")\s+do\s+(" + hours_gen_alt + r")\b"
        )

        # cache for next instance (thread-safe)
        with PolishTimeNormalizer._CACHE_LOCK:
            if PolishTimeNormalizer._CACHE is None:
                PolishTimeNormalizer._CACHE = dict(self.__dict__)

    @staticmethod
    def _ones_words() -> list[str]:
        return [
            "zero",
            "jeden",
            "dwa",
            "trzy",
            "cztery",
            "pięć",
            "sześć",
            "siedem",
            "osiem",
            "dziewięć",
        ]

    def _build_minutes(self) -> dict[str, int]:
        ones = self._ones_words()
        teens = {
            10: "dziesięć",
            11: "jedenaście",
            12: "dwanaście",
            13: "trzynaście",
            14: "czternaście",
            15: "piętnaście",
            16: "szesnaście",
            17: "siedemnaście",
            18: "osiemnaście",
            19: "dziewiętnaście",
        }
        tens = {
            20: "dwadzieścia",
            30: "trzydzieści",
            40: "czterdzieści",
            50: "pięćdziesiąt",
            60: "sześćdziesiąt",
            70: "siedemdziesiąt",
            80: "osiemdziesiąt",
            90: "dziewięćdziesiąt",
        }
        kwadrans = {"kwadrans": 15}

        def cardinal(n: int) -> str:
            if n < 10:
                return ones[n]
            if n < 20:
                return teens[n]
            t = n // 10 * 10
            o = n % 10
            if o == 0:
                return tens[t]
            return tens[t] + " " + ones[o]

        minutes: dict[str, int] = {}
        for m in range(60):
            minutes[cardinal(m)] = m
        for d in range(10):
            minutes["zero " + ones[d]] = d
        minutes.update(kwadrans)
        # feminine cardinal forms used with "minuta/minuty"
        minutes["jedna"] = 1
        minutes["dwie"] = 2
        return minutes

    @staticmethod
    def _norm_phrase(s: str) -> str:
        return " ".join(s.split())

    @staticmethod
    def _alternation(mapping: dict[str, int]) -> str:
        keys = sorted(mapping.keys(), key=lambda w: (-len(w), -w.count(" ")))

        def _escape_phrase(phrase: str) -> str:
            # allow flexible whitespace between words (single vs double spaces)
            return r"\s+".join(re.escape(part) for part in phrase.split())

        return "|".join(_escape_phrase(k) for k in keys)

    def __call__(self, s: str) -> str:
        # protect geographic "na północ/południe" (south/north) – keep as words
        # only convert time midnight/noon when not geographic
        _geo_map: dict[str, str] = {}

        def _protect_geo(m: re.Match[str]) -> str:
            # use UUID placeholder that cannot collide with user input
            # (no word chars that would match time patterns)
            key = f"__GEO_{uuid.uuid4().hex}__"
            _geo_map[key] = m.group(0)
            return key

        # geographic prepositions + północ/południe should stay (include ASCII variants)
        s = re.sub(
            r"\b(na|z|od|do|w|ku|kierunek|strona|część|czesc|północno|polnocno|południowo|poludniowo)\s+(północ|polnoc|południe|poludnie)\b",
            _protect_geo,
            s,
        )
        s = re.sub(r"\b(?:północ|polnoc)\b", "0:00", s)
        s = re.sub(r"\b(?:południe|poludnie)\b", "12:00", s)
        for k, v in _geo_map.items():
            s = s.replace(k, v)

        s = self._re_wpol.sub(
            lambda m: f"{(self.hours_gen[self._norm_phrase(m.group(1))] - 1) % 24}:30", s
        )
        # handle "o piątej rano" and "piąta rano" before generic "o piątej"
        s = self._re_o_hour_rano.sub(self._o_hour_rano_repl, s)
        s = self._re_hour_rano.sub(self._hour_rano_repl, s)
        s = self._re_za.sub(self._za_repl, s)
        s = self._re_po.sub(self._po_repl, s)
        s = self._re_godzina_min.sub(self._godzina_min_repl, s)
        s = self._re_godzina.sub(self._godzina_repl, s)
        s = self._re_hour_min.sub(self._hour_min_repl, s)
        s = self._re_hour_gen_min.sub(self._hour_gen_min_repl, s)
        s = self._re_o_hour.sub(self._o_hour_repl, s)
        s = self._re_za_minut.sub(self._za_repl, s)
        s = self._re_po_minut.sub(self._po_repl, s)
        s = self._re_range.sub(self._range_repl, s)
        return s

    def _za_repl(self, m: re.Match[str]) -> str:
        minute = self.minutes[self._norm_phrase(m.group(1))]
        hour = self.hours[self._norm_phrase(m.group(2))]
        if minute <= 0 or minute >= 60:
            return m.group(0)
        return f"{(hour - 1) % 24}:{60 - minute:02d}"

    def _po_repl(self, m: re.Match[str]) -> str:
        minute = self.minutes[self._norm_phrase(m.group(1))]
        hour = self.hours_gen[self._norm_phrase(m.group(2))]
        return f"{hour}:{minute:02d}"

    def _godzina_min_repl(self, m: re.Match[str]) -> str:
        hour = self.hours[self._norm_phrase(m.group(1))]
        minute = self.minutes[self._norm_phrase(m.group(2))]
        return f"{hour}:{minute:02d}"

    def _godzina_repl(self, m: re.Match[str]) -> str:
        hour = self.hours[self._norm_phrase(m.group(1))]
        return f"{hour}:00"

    def _hour_min_repl(self, m: re.Match[str]) -> str:
        hour = self.hours[self._norm_phrase(m.group(1))]
        minute = self.minutes[self._norm_phrase(m.group(2))]
        return f"{hour}:{minute:02d}"

    def _hour_gen_min_repl(self, m: re.Match[str]) -> str:
        hour = self.hours_gen[self._norm_phrase(m.group(1))]
        minute = self.minutes[self._norm_phrase(m.group(2))]
        return f"{hour}:{minute:02d}"

    def _o_hour_repl(self, m: re.Match[str]) -> str:
        hour = self.hours_gen[self._norm_phrase(m.group(1))]
        return f"o {hour}:00"

    def _o_hour_rano_repl(self, m: re.Match[str]) -> str:
        hour = self.hours_gen[self._norm_phrase(m.group(1))]
        return f"o {hour}:00 rano"

    def _hour_rano_repl(self, m: re.Match[str]) -> str:
        hour = self.hours[self._norm_phrase(m.group(1))]
        return f"{hour}:00 rano"

    def _range_repl(self, m: re.Match[str]) -> str:
        start = self.hours_gen[self._norm_phrase(m.group(1))]
        end = self.hours_gen[self._norm_phrase(m.group(2))]
        return f"od {start}:00 do {end}:00"
