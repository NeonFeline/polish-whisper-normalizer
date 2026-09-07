"""Polish Whisper Normalizer – public API."""

from __future__ import annotations

from .basic import BasicTextNormalizer, remove_symbols, remove_symbols_and_diacritics
from .polish import (
    PolishLemmatizer,
    PolishNumberNormalizer,
    PolishTextNormalizer,
    PolishTimeNormalizer,
)

try:
    from .jiwer import PolishTransform, polish_transform
    from .jiwer import wer as polish_wer

    _HAS_JIWER = True
except ImportError:  # pragma: no cover - jiwer optional
    PolishTransform = None  # type: ignore[assignment,misc]
    polish_transform = None  # type: ignore[assignment,misc]
    polish_wer = None  # type: ignore[assignment,misc]
    _HAS_JIWER = False

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("polish-whisper-normalizer")
except Exception:
    __version__ = "0.0.0"

__all__ = [
    "BasicTextNormalizer",
    "PolishLemmatizer",
    "PolishNumberNormalizer",
    "PolishTextNormalizer",
    "PolishTimeNormalizer",
    "PolishTransform",
    "__version__",
    "polish_transform",
    "polish_wer",
    "remove_symbols",
    "remove_symbols_and_diacritics",
]
