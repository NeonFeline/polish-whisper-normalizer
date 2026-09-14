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

from .utils import strip_diacritics, with_ascii_variants


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
            "dwudziestej czwartej": 24,
        }

        self.minutes = self._build_minutes()
        self.minutes_ordinal = self._build_minutes_ordinal()
        # expand with ASCII variants for diacritic-less ASR
        self.hours = with_ascii_variants(self.hours)
        self.hours_gen = with_ascii_variants(self.hours_gen)
        self.minutes = with_ascii_variants(self.minutes)
        self.minutes_ordinal = with_ascii_variants(self.minutes_ordinal)

        hours_alt = self._alternation(self.hours)
        hours_gen_alt = self._alternation(self.hours_gen)
        minutes_alt = self._alternation(self.minutes)
        minutes_ordinal_alt = self._alternation(self.minutes_ordinal)
        # literals with diacritics also need ASCII variants (diacritic-less ASR)
        wpol_pat = r"(?:wpół|wpol)"
        # build midnight/noon forms via Morfeusz (all declensions) + ASCII
        polnoc_forms, poludnie_forms = self._build_midnight_forms()
        # fallback hardcoded if Morfeusz unavailable (should not happen in prod)
        if not polnoc_forms:
            polnoc_forms = {
                "północ",
                "północy",
                "północą",
                "północe",
                "północom",
                "północami",
                "północach",
            }
        if not poludnie_forms:
            poludnie_forms = {
                "południe",
                "południa",
                "południowi",
                "południu",
                "południem",
                "południach",
                "południom",
                "południami",
            }
        # expand ASCII for midnight sets (strip_diacritics)
        polnoc_forms_expanded: set[str] = set()
        for f in polnoc_forms:
            polnoc_forms_expanded.add(f)
            polnoc_forms_expanded.add(strip_diacritics(f))
        poludnie_forms_expanded: set[str] = set()
        for f in poludnie_forms:
            poludnie_forms_expanded.add(f)
            poludnie_forms_expanded.add(strip_diacritics(f))
        # build alternations sorted by length desc
        polnoc_alt = self._alternation_set(polnoc_forms_expanded)
        poludnie_alt = self._alternation_set(poludnie_forms_expanded)
        # keep for __call__ (geographic vs time)
        self._polnoc_forms = polnoc_forms_expanded
        self._poludnie_forms = poludnie_forms_expanded
        self._polnoc_alt = polnoc_alt
        self._poludnie_alt = poludnie_alt

        # legacy single-form patterns kept for reference
        self._polnoc_pat = r"(?:północ|polnoc)"
        self._poludnie_pat = r"(?:południe|poludnie)"

        self._re_wpol = re.compile(r"\b" + wpol_pat + r"\s+do\s+(" + hours_gen_alt + r")\b")
        self._re_za = re.compile(r"\bza\s+(" + minutes_alt + r")\s+(" + hours_alt + r")\b")
        self._re_po = re.compile(r"\b(" + minutes_alt + r")\s+po\s+(" + hours_gen_alt + r")\b")
        self._re_godzina_min = re.compile(
            r"\bgodzina\s+(" + hours_alt + r")\s+(" + minutes_alt + r")\b"
        )
        self._re_godzina = re.compile(r"\bgodzina\s+(" + hours_alt + r")\b")
        self._re_o_godzinie_ord = re.compile(
            r"\bo\s+godzinie\s+(" + hours_gen_alt + r")\s+(" + minutes_ordinal_alt + r")\b"
        )
        self._re_o_godzinie_hour = re.compile(
            r"\bo\s+godzinie\s+(" + hours_gen_alt + r")\b(?!\s*(?:" + minutes_ordinal_alt + r")\b)"
        )
        self._re_godzina_digits = re.compile(r"\bgodzina\s+(\d{1,2})[.:,](\d{2})\b")
        self._re_o_godzinie_digits = re.compile(r"\bo\s+godzinie\s+(\d{1,2})[.:,](\d{2})\b")
        self._re_godzina_digit_hour = re.compile(r"\bgodzina\s+(\d{1,2})\b(?!\s*[:.,]\d)")
        self._re_o_godzinie_digit_hour = re.compile(r"\bo\s+godzinie\s+(\d{1,2})\b(?!\s*[:.,]\d)")
        self._re_hour_min = re.compile(r"\b(" + hours_alt + r")\s+(" + minutes_alt + r")\b")
        self._re_hour_gen_min = re.compile(r"\b(" + hours_gen_alt + r")\s+(" + minutes_alt + r")\b")
        # "o piątej" -> "o 5:00" (genitive hour, not followed by a word)
        self._re_o_hour = re.compile(r"\bo\s+(" + hours_gen_alt + r")\b(?!\s*[a-ząćęłńóśźż])")
        # time-of-day markers (Morfeusz-inspired, covers rano + wieczorem/nocy/południu)
        marker_phrases = [
            "rano",
            "wieczorem",
            "w nocy",
            "nocą",
            "nad ranem",
            "w dzień",
        ]
        marker_set: set[str] = set()
        for phrase in marker_phrases:
            phrase_lc = phrase.lower()
            marker_set.add(phrase_lc)
            stripped = strip_diacritics(phrase_lc)
            if stripped != phrase_lc:
                marker_set.add(stripped)
        # also ensure ascii variants for phrases containing diacritics are added
        # e.g. "w dzień" -> "w dzien"
        marker_alt = self._alternation_set(marker_set)
        self._marker_alt = marker_alt
        # hour + marker ("piąta rano", "ósma wieczorem" -> "5:00 rano")
        # also genitive hour + marker ("o piątej rano" is handled separately)
        self._re_hour_marker = re.compile(r"\b(" + hours_alt + r")\s+(" + marker_alt + r")\b")
        self._re_o_hour_marker = re.compile(
            r"\bo\s+(" + hours_gen_alt + r")\s+(" + marker_alt + r")\b"
        )
        # keep legacy rano regexes for backwards compat (they are now subset of marker)
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
        # midnight/noon regexes: contextual (prep + declined) + standalone nominative
        # only declined forms with appropriate preposition should become time; bare "północy" stays word (see test)
        # include ASCII-folded variants for prepositions (około -> okolo, była -> byla, etc.)
        time_prep = r"(o|przed|po|do|od|około|okolo|w|jest|była|byla|było|bylo|był|byl|były|byly)"
        self._re_polnoc_context = re.compile(r"\b" + time_prep + r"\s+(?:" + polnoc_alt + r")\b")
        self._re_poludnie_context = re.compile(
            r"\b" + time_prep + r"\s+(?:" + poludnie_alt + r")\b"
        )
        self._re_polnoc_nominative = re.compile(r"\b(?:północ|polnoc)\b")
        self._re_poludnie_nominative = re.compile(r"\b(?:południe|poludnie)\b")
        # keep broad regex for any internal use but not in pipeline (avoid bare declined conversion)
        self._re_polnoc = re.compile(r"\b(?:" + polnoc_alt + r")\b")
        self._re_poludnie = re.compile(r"\b(?:" + poludnie_alt + r")\b")
        # geographic protection: preposition + północ/południe (any declined form)
        geo_preps = r"(?:na|z|od|do|w|ku|kierunek|strona|część|czesc|północno|polnocno|południowo|poludniowo|pod)"
        self._re_geo_polnoc = re.compile(r"\b" + geo_preps + r"\s+(?:" + polnoc_alt + r")\b")
        self._re_geo_poludnie = re.compile(r"\b" + geo_preps + r"\s+(?:" + poludnie_alt + r")\b")
        # broader geographic: "na północy", "z północy", "od północy" etc already covered,
        # but also "północny" adjectives should stay words – no conversion via midnight regex (word boundary)

        # cache for next instance (thread-safe)
        with PolishTimeNormalizer._CACHE_LOCK:
            if PolishTimeNormalizer._CACHE is None:
                PolishTimeNormalizer._CACHE = dict(self.__dict__)

    @staticmethod
    def _build_midnight_forms() -> tuple[set[str], set[str]]:
        """Collect midnight/noon declined forms via Morfeusz generate."""
        try:
            from .lemmatizer import PolishLemmatizer

            lem = PolishLemmatizer()
            polnoc: set[str] = set()
            poludnie: set[str] = set()
            if lem.morf is not None:
                for base, target in (("północ", polnoc), ("południe", poludnie)):
                    try:
                        forms = lem.generate(base)
                    except Exception:
                        continue
                    for surf, lemma, _tag in forms:
                        if lemma.split(":")[0] != base:
                            # allow "północ:..." but ignore abbreviations like "pn"
                            if len(surf) <= 2:
                                continue
                            # still check base
                            continue
                        if len(surf) <= 2:  # skip abbrev "pn", "pd"
                            continue
                        pol = surf.lower()
                        target.add(pol)
                # ensure base forms present even if generate failed partially
                polnoc.add("północ")
                poludnie.add("południe")
                return polnoc, poludnie
        except Exception:
            pass
        return set(), set()

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
    def _build_minutes_ordinal() -> dict[str, int]:
        """Genitive feminine ordinals used for minutes ("szesnastej piątej" -> 16:05)."""
        units = [
            "pierwszej",
            "drugiej",
            "trzeciej",
            "czwartej",
            "piątej",
            "szóstej",
            "siódmej",
            "ósmej",
            "dziewiątej",
            "dziesiątej",
            "jedenastej",
            "dwunastej",
            "trzynastej",
            "czternastej",
            "piętnastej",
            "szesnastej",
            "siedemnastej",
            "osiemnastej",
            "dziewiętnastej",
        ]
        tens = {
            20: "dwudziestej",
            30: "trzydziestej",
            40: "czterdziestej",
            50: "pięćdziesiątej",
        }
        result: dict[str, int] = {}
        for m in range(1, 60):
            if m <= 19:
                result[units[m - 1]] = m
            else:
                t = m // 10 * 10
                o = m % 10
                if o == 0:
                    result[tens[t]] = m
                else:
                    result[f"{tens[t]} {units[o - 1]}"] = m
        return result

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

    @staticmethod
    def _alternation_set(values: set[str]) -> str:
        keys = sorted(values, key=lambda w: (-len(w), -w.count(" ")))

        def _escape_phrase(phrase: str) -> str:
            return r"\s+".join(re.escape(part) for part in phrase.split())

        return "|".join(_escape_phrase(k) for k in keys)

    def __call__(self, s: str) -> str:
        s = self._re_godzina_digits.sub(self._godzina_digits_repl, s)
        s = self._re_o_godzinie_digits.sub(self._o_godzinie_digits_repl, s)
        s = self._re_godzina_digit_hour.sub(self._godzina_digit_hour_repl, s)
        s = self._re_o_godzinie_digit_hour.sub(self._o_godzinie_digit_hour_repl, s)
        s = self._re_o_godzinie_hour.sub(self._o_godzinie_hour_repl, s)
        s = self._re_o_godzinie_ord.sub(self._o_godzinie_ord_repl, s)
        s = self._re_wpol.sub(
            lambda m: f"{(self.hours_gen[self._norm_phrase(m.group(1))] - 1) % 24}:30", s
        )
        # handle time-of-day markers: generic marker before narrow "rano" to capture all
        s = self._re_o_hour_marker.sub(self._o_hour_marker_repl, s)
        s = self._re_hour_marker.sub(self._hour_marker_repl, s)
        # legacy rano handlers (kept for compat, now redundant but harmless)
        s = self._re_o_hour_rano.sub(self._o_hour_rano_repl, s)
        s = self._re_hour_rano.sub(self._hour_rano_repl, s)

        # protect geographic "na północ/południe" (south/north) – keep as words
        # only convert time midnight/noon when not geographic
        _geo_map: dict[str, str] = {}

        def _protect_geo(m: re.Match[str]) -> str:
            # use UUID placeholder that cannot collide with user input
            # (no word chars that would match time patterns)
            key = f"__GEO_{uuid.uuid4().hex}__"
            _geo_map[key] = m.group(0)
            return key

        # geographic prepositions + północ/południe should stay (include ASCII variants and declensions via Morfeusz)
        # need to protect both midnight forms
        # we use two regexes but need to protect combined: first try polnoc then poludnie
        s = self._re_geo_polnoc.sub(_protect_geo, s)
        s = self._re_geo_poludnie.sub(_protect_geo, s)
        # fallback legacy pattern (covers simple "na północ" if Morfeusz forms miss)
        s = re.sub(
            r"\b(na|z|od|do|w|ku|kierunek|strona|część|czesc|północno|polnocno|południowo|poludniowo)\s+(północ|polnoc|południe|poludnie)\b",
            _protect_geo,
            s,
        )
        # convert midnight/noon: contextual prep+form and standalone nominative; bare declined like "północy" stays word
        s = self._re_polnoc_context.sub(lambda m: f"{m.group(1)} 0:00", s)
        s = self._re_poludnie_context.sub(lambda m: f"{m.group(1)} 12:00", s)
        s = self._re_polnoc_nominative.sub("0:00", s)
        s = self._re_poludnie_nominative.sub("12:00", s)
        for k, v in _geo_map.items():
            s = s.replace(k, v)
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

    def _o_hour_marker_repl(self, m: re.Match[str]) -> str:
        hour = self.hours_gen[self._norm_phrase(m.group(1))]
        marker = self._norm_phrase(m.group(2))
        return f"o {hour}:00 {marker}"

    def _hour_marker_repl(self, m: re.Match[str]) -> str:
        hour = self.hours[self._norm_phrase(m.group(1))]
        marker = self._norm_phrase(m.group(2))
        return f"{hour}:00 {marker}"

    def _range_repl(self, m: re.Match[str]) -> str:
        start = self.hours_gen[self._norm_phrase(m.group(1))]
        end = self.hours_gen[self._norm_phrase(m.group(2))]
        return f"od {start}:00 do {end}:00"

    def _godzina_digits_repl(self, m: re.Match[str]) -> str:
        hour, minute = int(m.group(1)), int(m.group(2))
        if hour > 24 or minute > 59:
            return m.group(0)
        return f"{hour}:{minute:02d}"

    def _o_godzinie_digits_repl(self, m: re.Match[str]) -> str:
        hour, minute = int(m.group(1)), int(m.group(2))
        if hour > 24 or minute > 59:
            return m.group(0)
        return f"o {hour}:{minute:02d}"

    def _godzina_digit_hour_repl(self, m: re.Match[str]) -> str:
        hour = int(m.group(1))
        if hour > 24:
            return m.group(0)
        return f"{hour}:00"

    def _o_godzinie_digit_hour_repl(self, m: re.Match[str]) -> str:
        hour = int(m.group(1))
        if hour > 24:
            return m.group(0)
        return f"o {hour}:00"

    def _o_godzinie_ord_repl(self, m: re.Match[str]) -> str:
        hour = self.hours_gen[self._norm_phrase(m.group(1))]
        minute = self.minutes_ordinal[self._norm_phrase(m.group(2))]
        return f"o {hour}:{minute:02d}"

    def _o_godzinie_hour_repl(self, m: re.Match[str]) -> str:
        hour = self.hours_gen[self._norm_phrase(m.group(1))]
        return f"o {hour}:00"
