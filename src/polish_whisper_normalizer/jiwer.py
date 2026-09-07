"""jiwer integration for Polish Whisper Normalizer."""

from __future__ import annotations

try:
    import jiwer
except ImportError as e:  # pragma: no cover
    raise ImportError("jiwer is required: uv pip install jiwer") from e

from .text import PolishTextNormalizer


class PolishTransform(jiwer.transforms.AbstractTransform):
    """jiwer transform that applies :class:`PolishTextNormalizer`.

    Use it **inside a Compose that ends with** ``ReduceToListOfListOfWords``,
    or use the ready-made :data:`polish_transform` / :func:`wer` helpers:

    Example:
        >>> import jiwer
        >>> from polish_whisper_normalizer.jiwer import PolishTransform
        >>> tr = jiwer.Compose([PolishTransform(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
        >>> jiwer.wer("piątego maja 2026", "05.05.2026",
        ...           reference_transform=tr, hypothesis_transform=tr)
        0.0
        >>> from polish_whisper_normalizer.jiwer import wer
        >>> wer("piątego maja 2026", "05.05.2026")
        0.0
    """

    def __init__(
        self,
        normalizer: PolishTextNormalizer | None = None,
        date_format: str | None = None,
        **kwargs: object,
    ) -> None:
        # Support legacy `PolishTransform(date_format="...")` and explicit normalizer
        if kwargs and date_format is None:
            # allow date_format via kwargs for backwards compat
            if "date_format" in kwargs and isinstance(kwargs["date_format"], str):
                date_format = kwargs["date_format"]  # type: ignore[assignment]
                kwargs.pop("date_format")
            if kwargs:
                unexpected = ", ".join(sorted(kwargs.keys()))
                raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        elif kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        self._normalizer: PolishTextNormalizer | None = normalizer
        self._date_format: str | None = date_format

    @property
    def normalizer(self) -> PolishTextNormalizer:
        if self._normalizer is None:
            if self._date_format is not None:
                self._normalizer = PolishTextNormalizer(date_format=self._date_format)
            else:
                self._normalizer = PolishTextNormalizer()
        return self._normalizer

    @normalizer.setter
    def normalizer(self, value: PolishTextNormalizer) -> None:
        self._normalizer = value

    def process_string(self, s: str) -> str:
        return self.normalizer(s)


# ready-made transform that already includes the required word reduction
# normalizer is lazy – no heavy Morfeusz work at import time
polish_transform = jiwer.Compose(
    [
        PolishTransform(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def wer(
    reference: str | list[str],
    hypothesis: str | list[str],
    date_format: str | None = None,
    **kwargs: object,
) -> float:
    """WER with Polish normalization (words→digits, dates, time, etc.).

    Wraps :func:`jiwer.wer` with :data:`polish_transform` on both sides.
    Pass ``date_format="..."`` to customize dates.

    Example:
        >>> from polish_whisper_normalizer.jiwer import wer
        >>> wer("piątego maja 2026", "05.05.2026")
        0.0
    """
    if kwargs and date_format is None and "date_format" in kwargs:
        date_format = kwargs.pop("date_format")  # type: ignore[assignment]
        if not isinstance(date_format, str):
            raise TypeError("date_format must be str")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    if date_format is not None:
        tr = jiwer.Compose(
            [
                PolishTransform(date_format=date_format),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )
        return jiwer.wer(reference, hypothesis, reference_transform=tr, hypothesis_transform=tr)
    return jiwer.wer(
        reference,
        hypothesis,
        reference_transform=polish_transform,
        hypothesis_transform=polish_transform,
    )


__all__ = ["PolishTransform", "polish_transform", "wer"]
