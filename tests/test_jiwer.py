import pytest

try:
    import jiwer  # noqa: F401

    from polish_whisper_normalizer.jiwer import PolishTransform, polish_transform, wer

    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False
    pytest.skip("jiwer not installed", allow_module_level=True)


def test_polish_wer_zero():
    assert wer("piątego maja 2026", "05.05.2026") == 0.0
    assert wer("piątego maja roku dwa tysiące dwudziestego szóstego", "05.05.2026") == 0.0
    assert wer("o piątej", "o 5:00") == 0.0
    assert wer("pięć złotówek", "5 zł") == 0.0
    assert wer("jedna trzecia", "1/3") == 0.0


def test_polish_wer_raw_vs_norm():
    # raw without normalization is 1.0, with is 0.0
    assert jiwer.wer("piątego maja 2026", "05.05.2026") == 1.0
    assert wer("piątego maja 2026", "05.05.2026") == 0.0


def test_polish_transform_compose():
    tr = PolishTransform()
    # direct string transform
    assert tr.process_string("piątego maja 2026") == "05.05.2026"
    # via jiwer
    assert (
        jiwer.wer(
            "piątego maja 2026",
            "05.05.2026",
            reference_transform=polish_transform,
            hypothesis_transform=polish_transform,
        )
        == 0.0
    )


def test_custom_date_format():
    assert wer("piątego maja 2026", "2026-05-05", date_format="%Y-%m-%d") == 0.0
