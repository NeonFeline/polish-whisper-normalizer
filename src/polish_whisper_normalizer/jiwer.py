"""jiwer integration for Polish Whisper Normalizer."""

from __future__ import annotations

try:
    import jiwer
except ImportError as e:  # pragma: no cover
    raise ImportError("jiwer is required: uv pip install jiwer") from e

from .polish import PolishTextNormalizer


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
        self, normalizer: PolishTextNormalizer | None = None, **kwargs: object
    ) -> None:
        # allow passing date_format etc. via kwargs
        if normalizer is None:
            if kwargs:
                normalizer = PolishTextNormalizer(**kwargs)  # type: ignore[arg-type]
            else:
                normalizer = PolishTextNormalizer()
        self.normalizer = normalizer

    def process_string(self, s: str) -> str:
        return self.normalizer(s)


# ready-made transform that already includes the required word reduction
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
    # allow custom normalizer via date_format kwarg
    if kwargs:
        tr = jiwer.Compose(
            [
                PolishTransform(**kwargs),  # type: ignore[arg-type]
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )
        return jiwer.wer(reference, hypothesis, reference_transform=tr, hypothesis_transform=tr)
    return jiwer.wer(reference, hypothesis, reference_transform=polish_transform, hypothesis_transform=polish_transform)


__all__ = ["PolishTransform", "polish_transform", "wer"]
