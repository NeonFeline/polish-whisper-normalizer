import re
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union

from more_itertools import windowed

from .basic import remove_symbols


class PolishLemmatizer:
    """
    Thin, caching wrapper around Morfeusz 2 used to map declined number words
    back to their base (lemma) form. If Morfeusz is unavailable it degrades
    gracefully and simply returns no analyses.
    """

    def __init__(self):
        self._morf = None
        self._cache = {}

    @property
    def morf(self):
        if self._morf is None:
            try:
                import morfeusz2

                self._morf = morfeusz2.Morfeusz()
            except Exception:
                self._morf = False
        return self._morf if self._morf is not False else None

    def analyse(self, word: str):
        """Return a list of (base_lemma, part_of_speech) tuples."""
        morf = self.morf
        if morf is None:
            return []
        if word in self._cache:
            return self._cache[word]
        results = []
        try:
            for analysis in morf.analyse(word):
                if len(analysis) == 3 and isinstance(analysis[2], tuple):
                    datum = analysis[2]
                    lemma, morph = datum[1], datum[2]
                else:
                    lemma, morph = analysis[1], analysis[2]
                results.append((lemma.split(":")[0], morph.split(":")[0]))
        except Exception:
            results = []
        self._cache[word] = results
        return results


class PolishNumberNormalizer:
    """
    Convert spelled-out Polish numbers into arabic digits, while handling:

    - cardinal numbers ("sto dwadzieścia trzy" -> "123")
    - grammatical variants of big multipliers ("tysiąc/tysiące/tysięcy" -> 1000)
    - currency ("pięć złotych" -> "5 zł", "pięćdziesiąt groszy" -> "50 gr")
    - percents ("dwadzieścia procent" -> "20%")
    - decimals ("trzy przecinek czternaście" -> "3.14")
    - signs ("minus dziesięć" -> "-10")
    - ordinal numbers, including declension ("dwudziesty pierwszy" -> "21.",
      "pierwszego" -> "1.", "trzeciej" -> "3.")
    - declined cardinal forms ("pięciu" -> "5", "dwóm" -> "2", "tysiąca" -> "1000")
    """

    def __init__(self):
        super().__init__()

        self.zeros = {"zero"}
        self.ones = {
            "jeden": 1,
            "jedna": 1,
            "jedno": 1,
            "dwa": 2,
            "dwie": 2,
            "trzy": 3,
            "cztery": 4,
            "pięć": 5,
            "sześć": 6,
            "siedem": 7,
            "osiem": 8,
            "dziewięć": 9,
            "dziesięć": 10,
            "jedenaście": 11,
            "dwanaście": 12,
            "trzynaście": 13,
            "czternaście": 14,
            "piętnaście": 15,
            "szesnaście": 16,
            "siedemnaście": 17,
            "osiemnaście": 18,
            "dziewiętnaście": 19,
        }
        self.tens = {
            "dwadzieścia": 20,
            "trzydzieści": 30,
            "czterdzieści": 40,
            "pięćdziesiąt": 50,
            "sześćdziesiąt": 60,
            "siedemdziesiąt": 70,
            "osiemdziesiąt": 80,
            "dziewięćdziesiąt": 90,
        }
        self.hundreds = {
            "sto": 100,
            "dwieście": 200,
            "trzysta": 300,
            "czterysta": 400,
            "pięćset": 500,
            "sześćset": 600,
            "siedemset": 700,
            "osiemset": 800,
            "dziewięćset": 900,
        }
        self.multipliers = {
            "tysiąc": 1_000,
            "tysiące": 1_000,
            "tysięcy": 1_000,
            "milion": 1_000_000,
            "miliony": 1_000_000,
            "milionów": 1_000_000,
            "miliard": 1_000_000_000,
            "miliardy": 1_000_000_000,
            "miliardów": 1_000_000_000,
            "bilion": 1_000_000_000_000,
            "biliony": 1_000_000_000_000,
            "bilionów": 1_000_000_000_000,
            "biliard": 1_000_000_000_000_000,
            "biliardy": 1_000_000_000_000_000,
            "biliardów": 1_000_000_000_000_000,
            "trylion": 1_000_000_000_000_000_000,
            "tryliony": 1_000_000_000_000_000_000,
            "trylionów": 1_000_000_000_000_000_000,
        }

        # ordinal numbers (base masculine-nominative forms)
        self.ones_ordinal = {
            "pierwszy": 1,
            "drugi": 2,
            "trzeci": 3,
            "czwarty": 4,
            "piąty": 5,
            "szósty": 6,
            "siódmy": 7,
            "ósmy": 8,
            "dziewiąty": 9,
            "dziesiąty": 10,
            "jedenasty": 11,
            "dwunasty": 12,
            "trzynasty": 13,
            "czternasty": 14,
            "piętnasty": 15,
            "szesnasty": 16,
            "siedemnasty": 17,
            "osiemnasty": 18,
            "dziewiętnasty": 19,
        }
        self.tens_ordinal = {
            "dwudziesty": 20,
            "trzydziesty": 30,
            "czterdziesty": 40,
            "pięćdziesiąty": 50,
            "sześćdziesiąty": 60,
            "siedemdziesiąty": 70,
            "osiemdziesiąty": 80,
            "dziewięćdziesiąty": 90,
        }
        self.hundreds_ordinal = {
            "setny": 100,
            "dwusetny": 200,
            "trzysetny": 300,
            "czterysetny": 400,
            "pięćsetny": 500,
            "sześćsetny": 600,
            "siedemsetny": 700,
            "osiemsetny": 800,
            "dziewięćsetny": 900,
        }
        self.multipliers_ordinal = {
            "tysięczny": 1_000,
            "milionowy": 1_000_000,
            "miliardowy": 1_000_000_000,
            "bilionowy": 1_000_000_000_000,
        }

        self.decimals = {*self.ones, *self.tens, *self.zeros}

        self.preceding_prefixers = {
            "minus": "-",
            "plus": "+",
        }
        self.currencies = {
            # Polish złoty / grosz
            "złoty": "zł",
            "złote": "zł",
            "złotych": "zł",
            "złotego": "zł",
            "zł": "zł",
            "pln": "zł",
            "złotówka": "zł",
            "grosz": "gr",
            "grosze": "gr",
            "groszy": "gr",
            "grosza": "gr",
            "gr": "gr",
            # euro / cent
            "euro": "€",
            "eur": "€",
            "cent": "¢",
            "centy": "¢",
            "centów": "¢",
            "centa": "¢",
            "¢": "¢",
            # dolar
            "dolar": "$",
            "dolary": "$",
            "dolarów": "$",
            "dolara": "$",
            "usd": "$",
            # funt (pound sterling)
            "funt": "£",
            "funty": "£",
            "funtów": "£",
            "gbp": "£",
            # currency symbols (canonical)
            "€": "€",
            "$": "$",
            "£": "£",
        }
        self.suffixers = {
            "procent": "%",
            "procenta": "%",
        }
        self.specials = {"przecinek", "kropka"}
        self.conjunctions = {"i"}
        self.prefixes = set(self.preceding_prefixers.values())

        self.words = set(
            [
                key
                for mapping in [
                    self.zeros,
                    self.ones,
                    self.tens,
                    self.hundreds,
                    self.multipliers,
                    self.ones_ordinal,
                    self.tens_ordinal,
                    self.hundreds_ordinal,
                    self.multipliers_ordinal,
                    self.preceding_prefixers,
                    self.currencies,
                    self.suffixers,
                    self.specials,
                    self.conjunctions,
                ]
                for key in mapping
            ]
        )

        # lemma -> value lookups used to canonicalize declined forms
        self.cardinal_lemmas = set(self.ones) | set(self.tens) | set(self.hundreds)
        self.multiplier_lemmas = set(self.multipliers)
        self.ordinal_lemmas = (
            set(self.ones_ordinal)
            | set(self.tens_ordinal)
            | set(self.hundreds_ordinal)
            | set(self.multipliers_ordinal)
        )
        self.ordinal_values = {
            **self.ones_ordinal,
            **self.tens_ordinal,
            **self.hundreds_ordinal,
            **self.multipliers_ordinal,
        }

        # feminine/neuter cardinal forms used as fraction numerators
        self.fraction_numerators = {
            "jedna": 1,
            "dwie": 2,
            "trzy": 3,
            "cztery": 4,
            "pięć": 5,
            "sześć": 6,
            "siedem": 7,
            "osiem": 8,
            "dziewięć": 9,
            "dziesięć": 10,
        }
        # non-number words whose declined forms we canonicalize
        self.percent_lemmas = {"procent"}
        # colloquial currency nouns
        self.currency_lemmas = {"złotówka"}
        # month lemmas -> number string (for full-date normalization)
        self.month_lemmas = {
            "styczeń": "1",
            "luty": "2",
            "marzec": "3",
            "kwiecień": "4",
            "maj": "5",
            "czerwiec": "6",
            "lipiec": "7",
            "sierpień": "8",
            "wrzesień": "9",
            "październik": "10",
            "listopad": "11",
            "grudzień": "12",
        }

        self.lemmatizer = PolishLemmatizer()
        self._canon_cache = {}

    def _canonicalize(self, word: str) -> str:
        """Map a declined number word to its base lemma, if it is one."""
        if word in self.words:
            return word
        if not re.fullmatch(r"[a-ząćęłńóśźż]+", word):
            return word
        cached = self._canon_cache.get(word)
        if cached is not None:
            return cached

        candidates = {"num": None, "adj": None, "subst": None, "percent": None, "currency": None}
        for base, pos in self.lemmatizer.analyse(word):
            if pos == "num" and base in self.cardinal_lemmas:
                candidates["num"] = base
            elif pos == "adj" and base in self.ordinal_lemmas:
                candidates["adj"] = base
            elif pos == "subst" and base in self.multiplier_lemmas:
                candidates["subst"] = base
            elif pos == "subst" and base in self.percent_lemmas:
                candidates["percent"] = base
            elif pos == "subst" and base in self.currency_lemmas:
                candidates["currency"] = base

        result = word
        for key in ("num", "adj", "subst", "percent", "currency"):
            if candidates[key] is not None:
                result = candidates[key]
                break
        self._canon_cache[word] = result
        return result

    def process_words(self, words: List[str]) -> Iterator[str]:
        prefix: Optional[str] = None
        value: Optional[Union[str, int]] = None
        ordinal = False
        skip = False

        def to_fraction(s: str):
            try:
                return Fraction(s)
            except ValueError:
                return None

        def output(result: Union[str, int]):
            nonlocal prefix, value, ordinal
            result = str(result)
            if prefix is not None:
                result = prefix + result
            if ordinal:
                result += "."
            value = None
            prefix = None
            ordinal = False
            return result

        if len(words) == 0:
            return

        for prev, current, next in windowed([None] + words + [None], 3):
            if skip:
                skip = False
                continue

            next_is_numeric = next is not None and re.match(r"^\d+(\.\d+)?$", next)
            has_prefix = current[0] in self.prefixes
            current_without_prefix = current[1:] if has_prefix else current

            if re.match(r"^\d+(\.\d+)?$", current_without_prefix):
                # arabic numbers (potentially with signs/currency prefixes)
                f = to_fraction(current_without_prefix)
                assert f is not None
                if value is not None:
                    if isinstance(value, str) and value.endswith("."):
                        value = str(value) + str(current)
                        continue
                    else:
                        yield output(value)

                prefix = current[0] if has_prefix else prefix
                if f.denominator == 1:
                    value = f.numerator
                else:
                    value = current_without_prefix
            elif current not in self.words:
                # non-numeric words
                if value is not None:
                    yield output(value)
                yield output(current)
            elif current in self.zeros:
                value = str(value or "") + "0"
            elif current in self.ones:
                ones = self.ones[current]
                if value is None:
                    value = ones
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:
                        assert value[-1] == "0"
                        value = value[:-1] + str(ones)
                    else:
                        value = str(value) + str(ones)
                elif ones < 10:
                    if value % 10 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
                else:  # eleven to nineteen
                    if value % 100 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
            elif current in self.tens:
                tens = self.tens[current]
                if value is None:
                    value = tens
                elif isinstance(value, str):
                    value = str(value) + str(tens)
                else:
                    if value % 100 == 0:
                        value += tens
                    else:
                        value = str(value) + str(tens)
            elif current in self.hundreds:
                hundred = self.hundreds[current]
                if value is None:
                    value = hundred
                elif isinstance(value, str):
                    value = str(value) + str(hundred)
                else:
                    if value % 1000 == 0:
                        value += hundred
                    else:
                        value = str(value) + str(hundred)
            elif current in self.multipliers:
                multiplier = self.multipliers[current]
                if value is None:
                    value = multiplier
                elif isinstance(value, str) or value == 0:
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        value = p.numerator
                    else:
                        yield output(value)
                        value = multiplier
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
            elif current in self.ones_ordinal:
                ones = self.ones_ordinal[current]
                ordinal = True
                if value is None:
                    yield output(ones)
                elif isinstance(value, str):
                    yield output(str(value) + str(ones))
                elif ones < 10:
                    if value % 10 == 0:
                        yield output(str(value + ones))
                    else:
                        yield output(str(value) + str(ones))
                else:  # 11-19
                    if value % 100 == 0:
                        yield output(str(value + ones))
                    else:
                        yield output(str(value) + str(ones))
            elif current in self.tens_ordinal:
                tens = self.tens_ordinal[current]
                ordinal = True
                if value is None:
                    value = tens
                elif isinstance(value, str):
                    value = str(value) + str(tens)
                else:
                    if value % 100 == 0:
                        value += tens
                    else:
                        value = str(value) + str(tens)
            elif current in self.hundreds_ordinal:
                # ordinal hundred; composes with a preceding multiplier
                # ("tysiąc dziewięćsetny" -> "1900.")
                hundred = self.hundreds_ordinal[current]
                ordinal = True
                if value is None:
                    yield output(hundred)
                elif isinstance(value, str):
                    yield output(str(value) + str(hundred))
                else:
                    if value % 1000 == 0:
                        value += hundred
                    else:
                        value = str(value) + str(hundred)
            elif current in self.multipliers_ordinal:
                if value is not None:
                    yield output(value)
                ordinal = True
                yield output(self.multipliers_ordinal[current])
            elif current in self.preceding_prefixers:
                # apply prefix (minus, plus, etc.) if it precedes a number
                if value is not None:
                    yield output(value)
                if next in self.words or next_is_numeric:
                    prefix = self.preceding_prefixers[current]
                else:
                    yield output(current)
            elif current in self.currencies:
                # currency word/abbreviation follows the amount -> suffix
                if value is not None:
                    yield output(str(value) + " " + self.currencies[current])
                elif current.isalpha():
                    yield output(current)
                # else: stray currency symbol with no amount -> drop
            elif current in self.suffixers:
                # apply suffix symbols (procent -> '%')
                if value is not None:
                    yield output(str(value) + self.suffixers[current])
                else:
                    yield output(current)
            elif current in self.specials:
                # decimal separators ("przecinek", "kropka")
                if next in self.decimals or next_is_numeric:
                    value = str(value or "") + "."
                else:
                    if value is not None:
                        yield output(value)
                    yield output(current)
            elif current in self.conjunctions:
                # drop "i" only when it joins two numeric tokens
                # (e.g. "sto złotych i pięćdziesiąt groszy" -> "100 zł 50 gr")
                prev_numeric = prev is not None and (
                    prev in self.words or re.match(r"^\d", prev or "")
                )
                next_numeric = next in self.words or next_is_numeric
                if prev_numeric and next_numeric:
                    if value is not None:
                        yield output(value)
                else:
                    if value is not None:
                        yield output(value)
                    yield output(current)
            else:
                raise ValueError(f"Unexpected token: {current}")

        if value is not None:
            yield output(value)

    def preprocess(self, s: str):
        # "i pół" -> "przecinek pięć" (two and a half -> 2.5)
        s = re.sub(r"\bi\s+pół\b", "przecinek pięć", s)
        s = re.sub(r"\bpółtora\b", "jeden przecinek pięć", s)
        s = re.sub(r"\bpółtorej\b", "jeden przecinek pięć", s)
        # standalone "pół" (half) -> "0.5"
        s = re.sub(r"\bpół\b", "zero przecinek pięć", s)

        # normalize currency symbols to follow the amount ("€10" -> "10 €",
        # "10€" -> "10 €", "$1.50" -> "1.50 $")
        s = re.sub(r"([€$£¢])\s*(\d+(?:\.\d+)?)", r"\2 \1", s)
        s = re.sub(r"(\d+(?:\.\d+)?)\s*([€$£¢])", r"\1 \2", s)

        # put a space at number/letter boundary
        s = re.sub(r"([^\W\d_])([0-9])", r"\1 \2", s)
        s = re.sub(r"([0-9])([^\W\d_])", r"\1 \2", s)

        return s

    def postprocess(self, s: str):
        return s

    def _fraction_denominator(self, word: str):
        """Return the ordinal value of `word` if it is a fraction denominator."""
        for base, pos in self.lemmatizer.analyse(word):
            if pos == "adj" and base in self.ordinal_values:
                value = self.ordinal_values[base]
                if value >= 2:
                    return value
        return None

    def _convert_fractions(self, words: List[str]) -> List[str]:
        """Turn "jedna trzecia" -> "1/3", "trzy czwarte" -> "3/4", etc."""
        result = []
        i = 0
        n = len(words)
        while i < n:
            numerator = self.fraction_numerators.get(words[i])
            if numerator is not None and i + 1 < n:
                denominator = self._fraction_denominator(words[i + 1])
                if denominator is not None:
                    result.append(f"{numerator}/{denominator}")
                    i += 2
                    continue
            result.append(words[i])
            i += 1
        return result

    def __call__(self, s: str):
        s = self.preprocess(s)
        words = self._convert_fractions(s.split())
        words = [self._canonicalize(w) for w in words]
        s = " ".join(word for word in self.process_words(words) if word is not None)
        s = self.postprocess(s)
        return s


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

    def __init__(self):
        super().__init__()

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

        hours_alt = self._alternation(self.hours)
        hours_gen_alt = self._alternation(self.hours_gen)
        minutes_alt = self._alternation(self.minutes)

        self._re_wpol = re.compile(r"\bwpół\s+do\s+(" + hours_gen_alt + r")\b")
        self._re_za = re.compile(
            r"\bza\s+(" + minutes_alt + r")\s+(" + hours_alt + r")\b"
        )
        self._re_po = re.compile(
            r"\b(" + minutes_alt + r")\s+po\s+(" + hours_gen_alt + r")\b"
        )
        self._re_godzina_min = re.compile(
            r"\bgodzina\s+(" + hours_alt + r")\s+(" + minutes_alt + r")\b"
        )
        self._re_godzina = re.compile(r"\bgodzina\s+(" + hours_alt + r")\b")
        self._re_hour_min = re.compile(
            r"\b(" + hours_alt + r")\s+(" + minutes_alt + r")\b"
        )
        self._re_hour_gen_min = re.compile(
            r"\b(" + hours_gen_alt + r")\s+(" + minutes_alt + r")\b"
        )
        # "o piątej" -> "o 5:00" (genitive hour, not followed by a word)
        self._re_o_hour = re.compile(
            r"\bo\s+(" + hours_gen_alt + r")\b(?!\s*[a-ząćęłńóśźż])"
        )
        # hour + time-of-day marker ("piąta rano" -> "5:00 rano")
        self._re_hour_rano = re.compile(r"\b(" + hours_alt + r")\s+rano\b")
        self._re_o_hour_rano = re.compile(
            r"\bo\s+(" + hours_gen_alt + r")\s+rano\b"
        )
        # variants with an explicit "minut(ę/y)" word
        self._re_za_minut = re.compile(
            r"\bza\s+(" + minutes_alt + r")\s+minut(?:a|ę|y)?\s+(" + hours_alt + r")\b"
        )
        self._re_po_minut = re.compile(
            r"\b(" + minutes_alt + r")\s+minut(?:a|ę|y)?\s+po\s+(" + hours_gen_alt + r")\b"
        )
        # "od piątej do szóstej" -> "od 5:00 do 6:00"
        self._re_range = re.compile(
            r"\bod\s+(" + hours_gen_alt + r")\s+do\s+(" + hours_gen_alt + r")\b"
        )

    @staticmethod
    def _ones_words() -> List[str]:
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

    def _build_minutes(self):
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

        minutes = {}
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
    def _alternation(mapping) -> str:
        keys = sorted(mapping.keys(), key=lambda w: (-len(w), -w.count(" ")))
        return "|".join(re.escape(k) for k in keys)

    def __call__(self, s: str):
        s = re.sub(r"\bpółnoc\b", "0:00", s)
        s = re.sub(r"\bpołudnie\b", "12:00", s)

        s = self._re_wpol.sub(
            lambda m: f"{(self.hours_gen[m.group(1)] - 1) % 24}:30", s
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

    def _za_repl(self, m: Match) -> str:
        minute = self.minutes[m.group(1)]
        hour = self.hours[m.group(2)]
        if minute <= 0 or minute >= 60:
            return m.group(0)
        return f"{(hour - 1) % 24}:{60 - minute:02d}"

    def _po_repl(self, m: Match) -> str:
        minute = self.minutes[m.group(1)]
        hour = self.hours_gen[m.group(2)]
        return f"{hour}:{minute:02d}"

    def _godzina_min_repl(self, m: Match) -> str:
        hour = self.hours[m.group(1)]
        minute = self.minutes[m.group(2)]
        return f"{hour}:{minute:02d}"

    def _godzina_repl(self, m: Match) -> str:
        hour = self.hours[m.group(1)]
        return f"{hour}:00"

    def _hour_min_repl(self, m: Match) -> str:
        hour = self.hours[m.group(1)]
        minute = self.minutes[m.group(2)]
        return f"{hour}:{minute:02d}"

    def _hour_gen_min_repl(self, m: Match) -> str:
        hour = self.hours_gen[m.group(1)]
        minute = self.minutes[m.group(2)]
        return f"{hour}:{minute:02d}"

    def _o_hour_repl(self, m: Match) -> str:
        hour = self.hours_gen[m.group(1)]
        return f"o {hour}:00"

    def _o_hour_rano_repl(self, m: Match) -> str:
        hour = self.hours_gen[m.group(1)]
        return f"o {hour}:00 rano"

    def _hour_rano_repl(self, m: Match) -> str:
        hour = self.hours[m.group(1)]
        return f"{hour}:00 rano"

    def _range_repl(self, m: Match) -> str:
        start = self.hours_gen[m.group(1)]
        end = self.hours_gen[m.group(2)]
        return f"od {start}:00 do {end}:00"


class PolishTextNormalizer:
    def __init__(self):
        self.ignore_patterns = r"\b(?:eee+|yyy+|hmm+|mhm+|mmm+|uh+|um+)\b"
        self.standardize_numbers = PolishNumberNormalizer()
        self.standardize_time = PolishTimeNormalizer()

    def __call__(self, s: str):
        s = s.lower()

        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = re.sub(self.ignore_patterns, "", s)

        # remove sentence periods before digits are introduced by time/number
        # normalization; keep decimals ("3.14") and ordinal markers ("21.")
        s = re.sub(r"(?<!\d)\.([^0-9]|$)", r" \1", s)

        s = self.standardize_time(s)

        s = re.sub(r"(\d),(\d)", r"\1.\2", s)  # Polish decimal comma -> point
        s = remove_symbols(s, keep=".:%$€£¢+-")  # keep numeric/time/sign/currency symbols

        s = self.standardize_numbers(s)

        # remove leftover symbols that are not part of a number/time
        s = re.sub(r"([^0-9])%", r"\1 ", s)
        s = re.sub(r"(?<!\d):|:(?!\d)", " ", s)
        s = re.sub(r"[-+](?!\d)", " ", s)

        s = re.sub(r"\s+", " ", s)  # replace successive whitespaces with a space
        return s
