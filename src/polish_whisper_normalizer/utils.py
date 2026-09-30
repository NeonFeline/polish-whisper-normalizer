"""Shared utilities – ASCII-folding and helpers.

Keeps diacritic handling in one place so normalizers don't duplicate logic.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .basic import remove_symbols_and_diacritics


def strip_diacritics(word: str) -> str:
    """ASCII-fold a single word (like remove_symbols_and_diacritics but without adding spaces)."""
    return remove_symbols_and_diacritics(word)


_HAS_DIGIT_RE = re.compile(r"\d")


def contains_digit(s: str) -> bool:
    """C-level digit check to skip digit-dependent regexes (perf guard)."""
    return _HAS_DIGIT_RE.search(s) is not None


def sub_if_present(
    s: str,
    pattern: re.Pattern[str],
    repl: str | Callable[[re.Match[str]], str],
    *needles: str,
) -> str:
    """Apply ``pattern.sub`` only if a needle literal is present (perf guard).

    Every needle must be a *necessary* condition: if none is in ``s`` the
    pattern cannot match and the scan is skipped. Never changes output.
    """
    for needle in needles:
        if needle in s:
            return pattern.sub(repl, s)
    return s


def with_ascii_variants(mapping: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *mapping* extended with ASCII-folded keys for diacritic forms."""
    expanded: dict[str, Any] = dict(mapping)
    for key, value in list(mapping.items()):
        stripped = strip_diacritics(key)
        if stripped != key and stripped not in expanded:
            expanded[stripped] = value
    return expanded


def with_ascii_variants_set(values: set[str]) -> set[str]:
    """Return a copy of *values* extended with ASCII-folded variants."""
    expanded = set(values)
    for key in list(values):
        stripped = strip_diacritics(key)
        if stripped != key:
            expanded.add(stripped)
    return expanded


# vulgar fractions (single codepoint) -> "num/den" for WER convergence
# ("½" -> "1/2" which postprocess maps to "0.5" like "jedna druga").
_VULGAR_FRACTIONS: dict[str, str] = {
    "¼": "1/4",
    "½": "1/2",
    "¾": "3/4",
    "⅐": "1/7",
    "⅑": "1/9",
    "⅒": "1/10",
    "⅓": "1/3",
    "⅔": "2/3",
    "⅕": "1/5",
    "⅖": "2/5",
    "⅗": "3/5",
    "⅘": "4/5",
    "⅙": "1/6",
    "⅚": "5/6",
    "⅛": "1/8",
    "⅜": "3/8",
    "⅝": "5/8",
    "⅞": "7/8",
}

_FRACTION_SLASH = "⁄"  # U+2044 produced by NFKC("½") -> "1⁄2"


def normalize_vulgar_fractions(s: str) -> str:
    """Replace vulgar fractions with ASCII ``num/den`` forms.

    Digit-prefixed forms (``1½``) become mixed numbers (``1 i 1/2``) so the
    existing ``X i Y/Z`` postprocess yields ``1.5`` like ``dwa i pół``.
    Also normalizes U+2044 fraction slash to ``/``.
    """
    if _FRACTION_SLASH in s:
        s = s.replace(_FRACTION_SLASH, "/")
    if not any(v in s for v in _VULGAR_FRACTIONS):
        return s
    # digit + vulgar (no space required): "1½" -> "1 i 1/2"
    for vulg, frac in _VULGAR_FRACTIONS.items():
        if vulg in s:
            s = re.sub(r"(\d+)\s*" + re.escape(vulg), r"\1 i " + frac, s)
            s = s.replace(vulg, frac)
    return s
