"""Compatibility shim – re-exports from split modules.

The implementation was split into ``lemmatizer.py``, ``numbers.py``,
``time.py``, ``text.py`` and ``utils.py`` to keep files focused and to
delegate all morphology to Morfeusz2 (see those modules). This file
remains for backwards compatibility: ``from polish_whisper_normalizer.polish
import PolishNumberNormalizer`` continues to work.
"""

from __future__ import annotations

from .lemmatizer import PolishLemmatizer
from .numbers import PolishNumberNormalizer
from .text import PolishTextNormalizer
from .time import PolishTimeNormalizer
from .utils import strip_diacritics, with_ascii_variants, with_ascii_variants_set

__all__ = [
    "PolishLemmatizer",
    "PolishNumberNormalizer",
    "PolishTextNormalizer",
    "PolishTimeNormalizer",
    "strip_diacritics",
    "with_ascii_variants",
    "with_ascii_variants_set",
]

# Backward compat aliases for private helpers that existed in the old monolith
_strip_diacritics = strip_diacritics  # type: ignore[unused-ignore]
_with_ascii_variants = with_ascii_variants  # type: ignore[unused-ignore]
_with_ascii_variants_set = with_ascii_variants_set  # type: ignore[unused-ignore]
