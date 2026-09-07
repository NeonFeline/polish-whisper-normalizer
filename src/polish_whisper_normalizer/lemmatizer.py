"""Thin caching wrapper around Morfeusz 2."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


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
        self._morf: Any | None = None
        self._morf_failed: bool = False
        self._cache: dict[str, list[tuple[str, str]]] = {}
        self._gen_cache: dict[str, list[tuple[str, str, str]]] = {}

    @property
    def morf(self) -> Any | None:
        if self._morf is not None:
            return self._morf
        if self._morf_failed:
            return None
        try:
            import morfeusz2

            self._morf = morfeusz2.Morfeusz()
        except ImportError:
            logger.debug("morfeusz2 not installed – lemmatizer disabled")
            self._morf_failed = True
            return None
        except Exception as exc:  # pragma: no cover – unexpected Morfeusz init error
            logger.warning("Failed to initialize Morfeusz: %s", exc)
            self._morf_failed = True
            return None
        return self._morf

    def analyse(self, word: str) -> list[tuple[str, str]]:
        """Return a list of (base_lemma, part_of_speech) tuples."""
        morf = self.morf
        if morf is None:
            return []
        if word in self._cache:
            return self._cache[word]
        results: list[tuple[str, str]] = []
        try:
            analyses = morf.analyse(word)  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover
            logger.debug("Morfeusz analyse failed for %r: %s", word, exc)
            self._cache[word] = []
            return []
        for analysis in analyses:
            try:
                if len(analysis) == 3 and isinstance(analysis[2], tuple):
                    datum = analysis[2]  # type: ignore[assignment]
                    lemma, morph = datum[1], datum[2]  # type: ignore[index]
                else:
                    lemma, morph = analysis[1], analysis[2]  # type: ignore[misc]
                results.append((lemma.split(":")[0], morph.split(":")[0]))
            except Exception:  # pragma: no cover – malformed entry
                continue
        self._cache[word] = results
        return results

    def generate(self, lemma: str) -> list[tuple[str, str, str]]:
        """Generate all surface forms for *lemma* (wrapper for Morfeusz.generate)."""
        if lemma in self._gen_cache:
            return self._gen_cache[lemma]
        morf = self.morf
        if morf is None:
            return []
        try:
            # Morfeusz.generate returns list of (form, lemma, tag, ..., ...)
            forms = morf.generate(lemma)  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover
            logger.debug("Morfeusz generate failed for %r: %s", lemma, exc)
            self._gen_cache[lemma] = []
            return []
        # Normalize to (surface, lemma, tag)
        out: list[tuple[str, str, str]] = []
        for entry in forms:
            if len(entry) >= 3:
                surface, lem, tag = entry[0], entry[1], entry[2]  # type: ignore[misc]
                if isinstance(surface, str) and isinstance(lem, str) and isinstance(tag, str):
                    out.append((surface, lem, tag))
        self._gen_cache[lemma] = out
        return out
