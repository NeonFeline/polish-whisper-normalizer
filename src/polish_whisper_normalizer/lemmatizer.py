"""Thin caching wrapper around Morfeusz 2."""

from __future__ import annotations

from typing import Any


class PolishLemmatizer:
    """
    Thin, caching wrapper around Morfeusz 2 used to map declined number words
    back to their base (lemma) form. If Morfeusz is unavailable it degrades
    gracefully and simply returns no analyses.

    Morfeusz is the single source of truth for declension – we do not
    re-implement Polish morphology manually; instead we rely on
    ``analyse``/``generate`` to canonicalize declined forms.
    """

    def __init__(self) -> None:
        self._morf: object | None = None
        self._cache: dict[str, list[tuple[str, str]]] = {}

    @property
    def morf(self) -> Any | None:
        if self._morf is None:
            try:
                import morfeusz2

                self._morf = morfeusz2.Morfeusz()
            except Exception:
                self._morf = False
        return self._morf if self._morf is not False else None

    def analyse(self, word: str) -> list[tuple[str, str]]:
        """Return a list of (base_lemma, part_of_speech) tuples."""
        morf = self.morf
        if morf is None:
            return []
        if word in self._cache:
            return self._cache[word]
        results = []
        try:
            for analysis in morf.analyse(word):  # type: ignore[union-attr]
                if len(analysis) == 3 and isinstance(analysis[2], tuple):
                    datum = analysis[2]
                    lemma, morph = datum[1], datum[2]
                else:
                    lemma, morph = analysis[1], analysis[2]  # type: ignore[misc]
                results.append((lemma.split(":")[0], morph.split(":")[0]))
        except Exception:
            results = []
        self._cache[word] = results
        return results

    def generate(self, lemma: str) -> list[tuple[str, str, str]]:
        """Generate all surface forms for *lemma* (wrapper for Morfeusz.generate)."""
        morf = self.morf
        if morf is None:
            return []
        try:
            # Morfeusz.generate returns list of (form, lemma, tag, ..., ...)
            forms = morf.generate(lemma)  # type: ignore[union-attr]
            # Normalize to (surface, lemma, tag)
            out: list[tuple[str, str, str]] = []
            for entry in forms:
                if len(entry) >= 3:
                    surface, lem, tag = entry[0], entry[1], entry[2]
                    out.append((surface, lem, tag))
            return out
        except Exception:
            return []
