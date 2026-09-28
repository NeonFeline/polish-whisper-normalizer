"""Performance regression tests and benchmark sanity checks.

Ensures that throughput and latency invariants hold across versions,
and verifies that the benchmarking CLI and runner execute cleanly.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.datasets import CONVERSATIONAL_MEDIUM, SHORT_UTTERANCES
from benchmarks.runner import (
    BenchmarkResult,
    benchmark_callable,
    format_console_table,
    format_markdown_table,
    run_component_benchmarks,
    run_workload_benchmarks,
)
from polish_whisper_normalizer import PolishTextNormalizer


@pytest.fixture(scope="module")
def normalizer() -> PolishTextNormalizer:
    return PolishTextNormalizer()


def test_normalizer_throughput_regression(normalizer: PolishTextNormalizer) -> None:
    """Ensure throughput does not regress below minimum acceptable SLA."""
    result = benchmark_callable(
        normalizer,
        CONVERSATIONAL_MEDIUM,
        name="Regression Throughput Test",
        iterations=5,
        warmup=2,
    )
    # Conservative threshold: should easily exceed 300 texts/sec on any modern CPU
    msg_t = f"Throughput regressed below 300 texts/s: got {result.items_per_sec:.1f} texts/s"
    assert result.items_per_sec > 300.0, msg_t
    msg_l = f"p95 latency exceeded 15ms: got {result.latency_p95_ms:.2f}ms"
    assert result.latency_p95_ms < 15.0, msg_l


def test_short_utterance_latency(normalizer: PolishTextNormalizer) -> None:
    """Ensure short voice assistant turns normalize in sub-millisecond p95 latency."""
    result = benchmark_callable(
        normalizer,
        SHORT_UTTERANCES,
        name="Short Utterance Latency Test",
        iterations=5,
        warmup=2,
    )
    msg = f"Short turn p95 latency too high: got {result.latency_p95_ms:.2f}ms"
    assert result.latency_p95_ms < 2.0, msg


def test_workload_benchmarks_run() -> None:
    """Verify that all workloads can be benchmarked without crashing."""
    results = run_workload_benchmarks(iterations=1, warmup=1)
    assert len(results) >= 8
    for r in results:
        assert isinstance(r, BenchmarkResult)
        assert r.total_calls > 0
        assert r.items_per_sec > 0.0


def test_component_benchmarks_run() -> None:
    """Verify that isolated component benchmarks execute cleanly."""
    results = run_component_benchmarks(iterations=1, warmup=1)
    assert len(results) >= 7
    for r in results:
        assert r.total_calls > 0


def test_formatters_output() -> None:
    """Verify console and markdown formatters produce non-empty strings."""
    sample_result = BenchmarkResult(
        name="Sample Benchmark",
        total_calls=100,
        total_words=500,
        total_chars=3000,
        total_time_sec=0.1,
        items_per_sec=1000.0,
        words_per_sec=5000.0,
        chars_per_sec=30000.0,
        kb_per_sec=29.3,
        latency_mean_ms=1.0,
        latency_stddev_ms=0.2,
        latency_p50_ms=0.9,
        latency_p90_ms=1.2,
        latency_p95_ms=1.4,
        latency_p99_ms=1.8,
        latency_min_ms=0.5,
        latency_max_ms=2.5,
    )
    console_out = format_console_table([sample_result])
    assert "Sample Benchmark" in console_out
    assert "1,000.0" in console_out

    markdown_out = format_markdown_table([sample_result])
    assert "| Sample Benchmark |" in markdown_out

    as_dict = sample_result.to_dict()
    assert as_dict["items_per_sec"] == 1000.0
    json_str = json.dumps(as_dict)
    assert "Sample Benchmark" in json_str
