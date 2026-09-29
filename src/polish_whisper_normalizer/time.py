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

# invisible formatting chars (zero-width, bidi) are dropped (10)
_FORMAT_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]+")


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
            "zerowa": 0,
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
            "zerowej": 0,
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
        # spaced "w pół do ósmej" (Whisper often splits wpół) -> same as wpół
        self._re_wpol_spaced = re.compile(r"\bw\s+(?:pół|pol)\s+do\s+(" + hours_gen_alt + r")\b")
        self._re_za = re.compile(r"\bza\s+(" + minutes_alt + r")\s+(" + hours_alt + r")\b")
        self._re_po = re.compile(r"\b(" + minutes_alt + r")\s+po\s+(" + hours_gen_alt + r")\b")
        self._re_godzina_min = re.compile(
            r"\bgodzina\s+(" + hours_alt + r")\s+(" + minutes_alt + r")\b"
        )
        self._re_godzina = re.compile(r"\bgodzina\s+(" + hours_alt + r")\b")
        self._re_o_godzinie_ord = re.compile(
            r"\bo\s+godzinie\s+(" + hours_gen_alt + r")\s+(" + minutes_ordinal_alt + r")\b"
        )
        self._re_o_godzinie_card = re.compile(
            r"\bo\s+godzinie\s+(" + hours_gen_alt + r")\s+(" + minutes_alt + r")\b"
        )
        self._re_o_godzinie_hour = re.compile(
            r"\bo\s+godzinie\s+("
            + hours_gen_alt
            + r")\b(?!\s*(?:"
            + minutes_ordinal_alt
            + r"|"
            + minutes_alt
            + r")\b)"
        )
        # "godzina" + declined forms (godziny/godzinie/godzinę/…) + digits.
        # Whisper usually omits "godzina" but corpus hits like
        # "do godziny 18.00" need the genitive trigger (bug 2).
        self._re_godzina_digits = re.compile(r"\bgodzin\w*\s+(\d{1,2})[.:,](\d{2})\b")
        self._re_o_godzinie_digits = re.compile(r"\bo\s+godzin\w*\s+(\d{1,2})[.:,](\d{2})\b")
        self._re_godzina_digit_hour = re.compile(r"\bgodzin\w*\s+(\d{1,2})\b(?!\s*[:.,]\d)")
        self._re_o_godzinie_digit_hour = re.compile(r"\bo\s+godzin\w*\s+(\d{1,2})\b(?!\s*[:.,]\d)")
        # Bare digit dot-times: "18.30" -> "18:30" (Whisper omits "godzina").
        # Dot only (comma is Polish decimal, colon already time). Excludes dates
        # (DD.MM with MM 01-12) and currency/percent contexts; requires round
        # minutes (M%5==0) or PM hour (H>=13) to preserve decimals like 3.14.
        # The (?!\.\d) lookahead keeps DD.MM.YYYY dates intact.
        self._re_bare_dot_time = re.compile(r"\b(\d{1,2})\.(\d{2})\b(?!\.\d)")
        self._re_hour_min = re.compile(r"\b(" + hours_alt + r")\s+(" + minutes_alt + r")\b")
        self._re_hour_gen_min = re.compile(r"\b(" + hours_gen_alt + r")\s+(" + minutes_alt + r")\b")
        # genitive hour + ordinal minutes with leading "o"
        # ("o piątej trzydziestej" -> "o 5:30"). Bare (no "o") stays ordinal
        # to avoid hijacking compound ordinals ("dwudziestej pierwszej" -> 21).
        self._re_o_hour_gen_ord_min = re.compile(
            r"\bo\s+(" + hours_gen_alt + r")\s+(" + minutes_ordinal_alt + r")\b"
        )
        # "o piątej" -> "o 5:00" (genitive hour, not followed by a word)
        self._re_o_hour = re.compile(r"\bo\s+(" + hours_gen_alt + r")\b(?!\s*[a-ząćęłńóśźż])")
        # digit hours: "o 11" -> "o 11:00" (not followed by word/time suffix),
        # "o 11 rano"/"11 rano" with time-of-day markers (marker_alt defined below,
        # placeholders wired after marker construction).
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
        # digit hours with markers ("11 rano" -> "11:00 rano",
        # "o 11 rano" -> "o 11:00 rano"); lookbehind avoids matching minutes
        # inside existing times ("5:15 rano" must not match "15 rano").
        self._re_digit_marker = re.compile(
            r"(?<![\d:])(?<!\d\s)(\d{1,2})\s+(" + marker_alt + r")\b"
        )
        self._re_o_digit_marker = re.compile(r"\bo\s+(\d{1,2})\s+(" + marker_alt + r")\b")
        # digit hour after "o" ("o 11" -> "o 11:00", not followed by word)
        self._re_o_digit_hour = re.compile(r"\bo\s+(\d{1,2})\b(?!\s*[:.,]\d)(?!\s*[a-ząćęłńóśźż])")
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
        # 'w południe' is temporal (at 12:00), never geographic in Polish (geographic is 'na południu'/'na południe')
        geo_preps_polnoc = r"(?:na|z|od|do|w|ku|kierunek|strona|część|czesc|północno|polnocno|pod)"
        geo_preps_poludnie = (
            r"(?:na|z|od|do|ku|kierunek|strona|część|czesc|południowo|poludniowo|pod)"
        )
        self._re_geo_polnoc = re.compile(r"\b" + geo_preps_polnoc + r"\s+(?:" + polnoc_alt + r")\b")
        self._re_geo_poludnie = re.compile(
            r"\b" + geo_preps_poludnie + r"\s+(?:" + poludnie_alt + r")\b"
        )
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
        kwadrans = {"kwadrans": 15, "kwadransa": 15}

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
        if not isinstance(s, str):
            raise TypeError(f"Expected str, got {type(s).__name__}")
        s = _FORMAT_RE.sub("", s)  # invisible formatting chars (10)
        s = self._re_godzina_digits.sub(self._godzina_digits_repl, s)
        s = self._re_o_godzinie_digits.sub(self._o_godzinie_digits_repl, s)
        s = self._re_godzina_digit_hour.sub(self._godzina_digit_hour_repl, s)
        s = self._re_o_godzinie_digit_hour.sub(self._o_godzinie_digit_hour_repl, s)
        s = self._re_bare_dot_time.sub(self._bare_dot_time_repl, s)
        # "o godzinie" hour-alone before hour+minutes so compound hours
        # ("dwudziestej pierwszej" = 21) claim before short hour+minutes
        # ("dwudziestej" + "pierwszej" = 20:01); hour-alone lookahead blocks
        # both cardinal and ordinal minutes.
        s = self._re_o_godzinie_hour.sub(self._o_godzinie_hour_repl, s)
        s = self._re_o_godzinie_card.sub(self._o_godzinie_ord_repl, s)
        s = self._re_o_godzinie_ord.sub(self._o_godzinie_ord_repl, s)
        s = self._re_wpol_spaced.sub(
            lambda m: f"{(self.hours_gen[self._norm_phrase(m.group(1))] - 1) % 24}:30", s
        )
        s = self._re_wpol.sub(
            lambda m: f"{(self.hours_gen[self._norm_phrase(m.group(1))] - 1) % 24}:30", s
        )
        # handle time-of-day markers: generic marker before narrow "rano" to capture all
        # digit markers first (most specific), then word markers
        s = self._re_o_digit_marker.sub(self._o_digit_marker_repl, s)
        s = self._re_digit_marker.sub(self._digit_marker_repl, s)
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
            r"\b(na|z|od|do|ku|kierunek|strona|część|czesc|północno|polnocno|południowo|poludniowo)\s+(północ|polnoc|południe|poludnie)\b",
            _protect_geo,
            s,
        )
        # convert midnight/noon: contextual prep+form and standalone nominative; bare declined like "północy" stays word
        # protect redundant time phrases like "12:00 w południe" / "0:00 o północy"
        s = re.sub(r"\b(?:12:00|12)\s+w\s+(?:południe|poludnie)\b", _protect_geo, s)
        s = re.sub(
            r"\b(?:0:00|24:00|0|24)\s+(?:o\s+)?(?:północy|polnocy|północ|polnoc)\b",
            _protect_geo,
            s,
        )
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
        s = self._re_o_hour_gen_ord_min.sub(self._o_hour_gen_ord_min_repl, s)
        s = self._re_o_digit_hour.sub(self._o_digit_hour_repl, s)
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

    def _bare_dot_time_repl(self, m: re.Match[str]) -> str:
        hour, minute = int(m.group(1)), int(m.group(2))
        if hour > 24 or minute > 59:
            return m.group(0)

        # context for disambiguation (computed early so hour-0 and date
        # guards can be overridden by explicit time prepositions)
        full = m.string
        start, end = m.start(), m.end()
        after = full[end : end + 15].lower()
        before = full[max(0, start - 15) : start].lower()
        after_stripped = after.lstrip()
        before_stripped = before.rstrip()
        has_time_prep = bool(re.search(r"\b(?:o|od|do|około|okolo|koło|kolo)\s*$", before_stripped))
        # time-of-day marker after the number ("5.30 rano") also signals time
        has_marker_after = bool(
            re.match(
                r"(?:rano|wieczorem|w\s+nocy|nocą|noca|nad\s+ranem|w\s+dzień|w\s+dzien)\b",
                after_stripped,
            )
        )
        has_time_context = has_time_prep or has_marker_after

        # Preposition "o" specifically signals clock time ("o 5.05"), whereas "od", "do",
        # "około" frequently introduce date ranges ("od 5.05 do 10.05") where MM 01-12 must stay dates.
        has_o_prep = bool(re.search(r"\bo\s*$", before_stripped))
        has_date_override = has_o_prep or has_marker_after

        # Hour 0 needs explicit context ("godzina 0.00" handled elsewhere,
        # "o 0.00" here). Bare "0.00" stays decimal/number.
        if hour == 0 and not has_time_context:
            return m.group(0)
        # Preserve day.month dates: DD.MM with MM 01-12 stays dotted
        # (e.g. 21.05 May 21, 05.05, 01.01, od 5.05 do 10.05). Only explicit
        # time context ("o 5.05", "5.05 rano") overrides the date guard.
        if 1 <= hour <= 31 and 1 <= minute <= 12 and not has_date_override:
            return m.group(0)

        # Preserve units of measurement, quantities, and currencies

        # Check for currency or percent symbols
        if after_stripped[:1] in {"$", "€", "£", "¢", "%"}:
            return m.group(0)
        if before_stripped[-1:] in {"$", "€", "£", "¢"}:
            return m.group(0)

        # Check for currency and measurement units following the number
        units = (
            "zł",
            "gr",
            "euro",
            "eur",
            "dolar",
            "usd",
            "cent",
            "funt",
            "gbp",
            "pln",
            "złoty",
            "złotych",
            "złote",
            "złotego",
            "grosz",
            "groszy",
            "grosze",
            "grosza",
            "procent",
            "proc",
            "kg",
            "g",
            "mg",
            "dag",
            "t",
            "tona",
            "tony",
            "ton",
            "m",
            "km",
            "cm",
            "mm",
            "metr",
            "metra",
            "metry",
            "metrów",
            "l",
            "ml",
            "litr",
            "litra",
            "litry",
            "litrów",
            "s",
            "sek",
            "sekund",
            "sekundy",
            "sekunda",
            "stopni",
            "stopnie",
            "stopnia",
            "punkt",
            "punkty",
            "punktów",
            "pkt",
        )
        for u in units:
            if after_stripped.startswith(u):
                rest = after_stripped[len(u) :]
                if not rest or not rest[0].isalpha():
                    return m.group(0)

        # Bare times require round minutes (M % 5 == 0) and daytime/evening hour (H >= 6)
        # unless explicit time context (preposition or marker) is present.
        if not has_time_context:
            if hour < 6:
                return m.group(0)
            if minute % 5 != 0:
                return m.group(0)

        return f"{hour}:{minute:02d}"

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
        minute_phrase = self._norm_phrase(m.group(2))
        # shared by ordinal ("o godzinie szesnastej piątej") and cardinal
        # ("o godzinie piątej trzydzieści") minute variants
        minute = self.minutes_ordinal.get(minute_phrase)
        if minute is None:
            minute = self.minutes.get(minute_phrase)
        if minute is None:  # pragma: no cover – regex guarantees a hit
            return m.group(0)
        return f"o {hour}:{minute:02d}"

    def _hour_gen_ord_min_repl(self, m: re.Match[str]) -> str:
        # kept for backwards compat (bare genitive+ordinal now handled via
        # o-prefixed version to avoid ordinal hijack); not used in pipeline
        hour = self.hours_gen[self._norm_phrase(m.group(1))]
        minute = self.minutes_ordinal[self._norm_phrase(m.group(2))]
        return f"{hour}:{minute:02d}"

    def _o_hour_gen_ord_min_repl(self, m: re.Match[str]) -> str:
        hour = self.hours_gen[self._norm_phrase(m.group(1))]
        minute = self.minutes_ordinal[self._norm_phrase(m.group(2))]
        return f"o {hour}:{minute:02d}"

    def _o_digit_marker_repl(self, m: re.Match[str]) -> str:
        hour = int(m.group(1))
        if hour > 24:
            return m.group(0)
        marker = self._norm_phrase(m.group(2))
        return f"o {hour}:00 {marker}"

    def _digit_marker_repl(self, m: re.Match[str]) -> str:
        hour = int(m.group(1))
        if hour > 24:
            return m.group(0)
        marker = self._norm_phrase(m.group(2))
        return f"{hour}:00 {marker}"

    def _o_digit_hour_repl(self, m: re.Match[str]) -> str:
        hour = int(m.group(1))
        if hour > 24:
            return m.group(0)
        # Check context before "o <hour>":
        full = m.string
        start = m.start()
        before = full[max(0, start - 25) : start].lower()
        # Words indicating difference, increase/decrease, count, or prepositional usage:
        # e.g. "zwiększyć o 5", "mniej o 2", "chodzi o 1", "pomylił się o 3", "różnica o 4"
        if re.search(
            r"\b(?:mniej|więcej|zwiększ\w*|zmniejsz\w*|wzros\w*|spad\w*|podn\w*|obniż\w*|"
            r"pomyl\w*|spóźn\w*|różnic\w*|błąd\w*|chodz\w*|pyta\w*|prosi\w*|walcz\w*|"
            r"apel\w*|popraw\w*|zmień\w*|zmien\w*|skróć\w*|skroc\w*|wydłuż\w*|wydluz\w*|"
            r"wygra\w*|przegra\w*)\s*$",
            before,
        ):
            return m.group(0)
        # Hours 1-5 without explicit time markers (e.g. "rano", "godzina")
        # are almost exclusively delta/counting tokens in Polish ("o 1", "o 2", "o 3", "o 4", "o 5")
        if hour < 6:
            return m.group(0)
        return f"o {hour}:00"

    def _o_godzinie_hour_repl(self, m: re.Match[str]) -> str:
        hour = self.hours_gen[self._norm_phrase(m.group(1))]
        return f"o {hour}:00"
