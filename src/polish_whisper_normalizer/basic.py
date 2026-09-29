import re
import unicodedata
from collections.abc import Callable

import regex

# non-ASCII letters that are not decomposed by NFKD (inherited from Whisper english.py).
# Polish requires ł/Ł; others kept for compatibility with mixed-language ASR.
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep: str = "") -> str:
    """
    Replace any other markers, symbols, and punctuations with a space,
    and drop any diacritics (category 'Mn' and some manual mappings).
    Uses NFKD to decompose diacritics; manual ADDITIONAL_DIACRITICS handles
    chars not decomposed (e.g. ł).
    """
    return "".join(
        (
            c
            if c in keep
            else (
                ADDITIONAL_DIACRITICS[c]
                if c in ADDITIONAL_DIACRITICS
                else (
                    ""
                    if unicodedata.category(c) == "Mn"
                    else " "
                    if unicodedata.category(c)[0] in "MSP"
                    else c
                )
            )
        )
        for c in unicodedata.normalize("NFKD", s)
    )


def remove_symbols(s: str, keep: str = "") -> str:
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    (and any characters listed in `keep`). Uses NFKC to keep composed diacritics intact.
    """
    s = unicodedata.normalize("NFKC", s).replace("⁄", "/")
    return "".join(
        c if c in keep else (" " if unicodedata.category(c)[0] in "MSP" else c) for c in s
    )


class BasicTextNormalizer:
    # pre-compiled for speed and clarity (avoid character-class confusion)
    # NOTE: brackets (), [], <>, {} are punctuation – only the bracket
    # characters are dropped (via clean/remove_symbols below), enclosed words kept.
    # Invisible formatting chars (zero-width, bidi) are dropped (10).
    _FORMAT_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]+")
    _WS_RE = re.compile(r"\s+")

    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False) -> None:
        self.clean: Callable[..., str] = (
            remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        )
        self.split_letters: bool = split_letters

    def __call__(self, s: str) -> str:
        if not isinstance(s, str):
            raise TypeError(f"Expected str, got {type(s).__name__}")
        s = s.lower()
        s = self._FORMAT_RE.sub("", s)  # invisible formatting chars (10)
        s = self.clean(s)

        if self.split_letters:
            s = " ".join(regex.findall(r"\X", s, regex.U))

        s = self._WS_RE.sub(" ", s)  # replace any successive whitespace characters with a space

        return s.strip()
