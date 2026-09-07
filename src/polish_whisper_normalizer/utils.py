"""Shared utilities – ASCII-folding and helpers.

Keeps diacritic handling in one place so normalizers don't duplicate logic.
"""

from __future__ import annotations

from typing import Any

from .basic import remove_symbols_and_diacritics


def strip_diacritics(word: str) -> str:
    """ASCII-fold a single word (like remove_symbols_and_diacritics but without adding spaces)."""
    return remove_symbols_and_diacritics(word)


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
