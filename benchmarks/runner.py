"""Benchmarking execution engine for Polish Whisper Normalizer.

Calculates throughput (texts/sec, words/sec, chars/sec, KB/sec) and
detailed latency percentiles (p50, p90, p95, p99, min, max, mean, stddev).
Supports single-call, batch, concurrent, and component-level benchmarks.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import math
import statistics
import time
from collections.abc import Callable
from typing import Any

from polish_whisper_normalizer import (
    BasicTextNormalizer,
    PolishLemmatizer,
    PolishNumberNormalizer,
    PolishTextNormalizer,
    PolishTimeNormalizer,
)
from polish_whisper_normalizer.basic import remove_symbols, remove_symbols_and_diacritics
from polish_whisper_normalizer.utils import strip_diacritics

try:
    from .datasets import (
        ALL_WORKLOADS,
        CONVERSATIONAL_MEDIUM,
        SHORT_UTTERANCES,
    )
except ImportError:
    from benchmarks.datasets import (  # type: ignore[no-redef]
        ALL_WORKLOADS,
        CONVERSATIONAL_MEDIUM,
        SHORT_UTTERANCES,
    )


@dataclasses.dataclass
class BenchmarkResult:
    """Metrics recorded during a benchmark run."""

    name: str
    total_calls: int
    total_words: int
    total_chars: int
    total_time_sec: float
    items_per_sec: float
    words_per_sec: float
    chars_per_sec: float
    kb_per_sec: float
    latency_mean_ms: float
    latency_stddev_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float
    extra_metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a dictionary for JSON serialization."""
        return {
            "name": self.name,
            "total_calls": self.total_calls,
            "total_words": self.total_words,
            "total_chars": self.total_chars,
            "total_time_sec": round(self.total_time_sec, 4),
            "items_per_sec": round(self.items_per_sec, 2),
            "words_per_sec": round(self.words_per_sec, 2),
            "chars_per_sec": round(self.chars_per_sec, 2),
            "kb_per_sec": round(self.kb_per_sec, 2),
            "latency_ms": {
                "mean": round(self.latency_mean_ms, 3),
                "stddev": round(self.latency_stddev_ms, 3),
                "p50": round(self.latency_p50_ms, 3),
                "p90": round(self.latency_p90_ms, 3),
                "p95": round(self.latency_p95_ms, 3),
                "p99": round(self.latency_p99_ms, 3),
                "min": round(self.latency_min_ms, 3),
                "max": round(self.latency_max_ms, 3),
            },
            "extra_metadata": self.extra_metadata,
        }


def _percentile(sorted_data: list[float], p: float) -> float:
    """Compute percentile from sorted ascending floats (0.0 to 100.0)."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1


def benchmark_callable(
    func: Callable[[str], Any],
    dataset: list[str],
    *,
    name: str = "Callable",
    iterations: int = 10,
    warmup: int = 2,
    extra_metadata: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Benchmark a normalizer callable over a dataset for N iterations.

    Args:
        func: The function to benchmark (e.g. PolishTextNormalizer()).
        dataset: List of input strings.
        name: Label for the benchmark.
        iterations: Number of full passes over dataset.
        warmup: Number of warmup passes to run before timing.
        extra_metadata: Additional metadata to attach.

    Returns:
        BenchmarkResult with calculated metrics.
    """
    # Warmup passes
    for _ in range(warmup):
        for text in dataset:
            func(text)

    latencies_sec: list[float] = []
    total_words = 0
    total_chars = 0

    t_start = time.perf_counter()
    for _ in range(iterations):
        for text in dataset:
            t0 = time.perf_counter()
            func(text)
            dt = time.perf_counter() - t0
            latencies_sec.append(dt)
            total_words += len(text.split())
            total_chars += len(text)
    t_total = time.perf_counter() - t_start

    total_calls = len(latencies_sec)
    latencies_sec.sort()
    latencies_ms = [t * 1000.0 for t in latencies_sec]

    mean_ms = statistics.mean(latencies_ms) if latencies_ms else 0.0
    stddev_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    p50_ms = _percentile(latencies_ms, 50.0)
    p90_ms = _percentile(latencies_ms, 90.0)
    p95_ms = _percentile(latencies_ms, 95.0)
    p99_ms = _percentile(latencies_ms, 99.0)
    min_ms = latencies_ms[0] if latencies_ms else 0.0
    max_ms = latencies_ms[-1] if latencies_ms else 0.0

    items_per_sec = total_calls / t_total if t_total > 0 else 0.0
    words_per_sec = total_words / t_total if t_total > 0 else 0.0
    chars_per_sec = total_chars / t_total if t_total > 0 else 0.0
    kb_per_sec = (total_chars / 1024.0) / t_total if t_total > 0 else 0.0

    return BenchmarkResult(
        name=name,
        total_calls=total_calls,
        total_words=total_words,
        total_chars=total_chars,
        total_time_sec=t_total,
        items_per_sec=items_per_sec,
        words_per_sec=words_per_sec,
        chars_per_sec=chars_per_sec,
        kb_per_sec=kb_per_sec,
        latency_mean_ms=mean_ms,
        latency_stddev_ms=stddev_ms,
        latency_p50_ms=p50_ms,
        latency_p90_ms=p90_ms,
        latency_p95_ms=p95_ms,
        latency_p99_ms=p99_ms,
        latency_min_ms=min_ms,
        latency_max_ms=max_ms,
        extra_metadata=extra_metadata or {},
    )


def run_workload_benchmarks(
    normalizer: PolishTextNormalizer | None = None,
    iterations: int = 15,
    warmup: int = 3,
) -> list[BenchmarkResult]:
    """Benchmark PolishTextNormalizer across all categorized workloads."""
    norm = normalizer or PolishTextNormalizer()
    results: list[BenchmarkResult] = []

    for workload_name, data in ALL_WORKLOADS.items():
        res = benchmark_callable(
            norm,
            data,
            name=f"Workload: {workload_name}",
            iterations=iterations,
            warmup=warmup,
            extra_metadata={"workload": workload_name, "sample_count": len(data)},
        )
        results.append(res)

    return results


def run_component_benchmarks(
    iterations: int = 15,
    warmup: int = 3,
) -> list[BenchmarkResult]:
    """Benchmark each individual component and utility in isolation."""
    data = CONVERSATIONAL_MEDIUM
    results: list[BenchmarkResult] = []

    components: list[tuple[str, Callable[[str], Any]]] = [
        ("Full Pipeline (PolishTextNormalizer)", PolishTextNormalizer()),
        ("Number Normalizer (PolishNumberNormalizer)", PolishNumberNormalizer()),
        ("Time Normalizer (PolishTimeNormalizer)", PolishTimeNormalizer()),
        (
            "Basic Whisper (BasicTextNormalizer composed)",
            BasicTextNormalizer(remove_diacritics=False),
        ),
        ("Basic Whisper (BasicTextNormalizer ascii)", BasicTextNormalizer(remove_diacritics=True)),
        ("Low-level: remove_symbols", remove_symbols),
        ("Low-level: remove_symbols_and_diacritics", remove_symbols_and_diacritics),
        ("Low-level: strip_diacritics", strip_diacritics),
    ]

    for label, comp in components:
        res = benchmark_callable(
            comp,
            data,
            name=label,
            iterations=iterations,
            warmup=warmup,
        )
        results.append(res)

    # Morfeusz Lemmatizer benchmark (warm vs synthetic cold words)
    lemmatizer = PolishLemmatizer()
    lem_sample = ["dwudziestu", "pięciu", "tysiącach", "pierwszego", "maja", "złotych"]
    lem_res = benchmark_callable(
        lemmatizer.analyse,
        lem_sample,
        name="Morfeusz Lemmatizer (analyse warm cache)",
        iterations=iterations * 10,
        warmup=warmup,
    )
    results.append(lem_res)

    # jiwer integration if available
    try:
        from polish_whisper_normalizer.jiwer import wer

        def wer_runner(pair: str) -> float:
            return wer(pair, pair)

        wer_res = benchmark_callable(
            wer_runner,
            SHORT_UTTERANCES,
            name="jiwer.wer with PolishTransform",
            iterations=iterations,
            warmup=warmup,
        )
        results.append(wer_res)
    except ImportError:
        pass

    return results


def run_scaling_benchmark(
    worker_counts: tuple[int, ...] = (1, 2, 4, 8),
    iterations_per_worker: int = 50,
) -> list[BenchmarkResult]:
    """Benchmark multi-threaded scaling with shared normalizer instance."""
    norm = PolishTextNormalizer()
    dataset = CONVERSATIONAL_MEDIUM
    results: list[BenchmarkResult] = []

    # Warmup
    for text in dataset:
        norm(text)

    for num_workers in worker_counts:
        t_start = time.perf_counter()

        def worker_task() -> int:
            count = 0
            for _ in range(iterations_per_worker):
                for text in dataset:
                    norm(text)
                    count += 1
            return count

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_task) for _ in range(num_workers)]
            total_items = sum(f.result() for f in futures)

        t_total = time.perf_counter() - t_start
        throughput = total_items / t_total if t_total > 0 else 0.0

        results.append(
            BenchmarkResult(
                name=f"Concurrent scaling: {num_workers} threads",
                total_calls=total_items,
                total_words=0,
                total_chars=0,
                total_time_sec=t_total,
                items_per_sec=throughput,
                words_per_sec=0.0,
                chars_per_sec=0.0,
                kb_per_sec=0.0,
                latency_mean_ms=1000.0 / throughput if throughput > 0 else 0.0,
                latency_stddev_ms=0.0,
                latency_p50_ms=0.0,
                latency_p90_ms=0.0,
                latency_p95_ms=0.0,
                latency_p99_ms=0.0,
                latency_min_ms=0.0,
                latency_max_ms=0.0,
                extra_metadata={"threads": num_workers, "items": total_items},
            )
        )

    return results


def run_stress_benchmarks(
    iterations: int = 10,
    warmup: int = 2,
) -> list[BenchmarkResult]:
    """Benchmark the normalizer under adversarial and stress test conditions."""
    from .datasets import ADVERSARIAL_STRESS

    norm = PolishTextNormalizer()
    results: list[BenchmarkResult] = []

    res_all = benchmark_callable(
        norm,
        ADVERSARIAL_STRESS,
        name="Stress: Mixed Adversarial Inputs",
        iterations=iterations,
        warmup=warmup,
    )
    results.append(res_all)

    # Specific adversarial patterns
    stress_patterns: list[tuple[str, list[str]]] = [
        ("Stress: Long Whitespace (1000 spaces)", ["początek" + " " * 1000 + "koniec"]),
        (
            "Stress: Punctuation Storm (1000 symbols)",
            ["start " + "!@#$%^&*()_+-=[]{}|;':\",./<>?`~" * 30 + " end"],
        ),
        ("Stress: Ellipsis & Dots (200 dots)", ["tekst" + "..." * 50 + " " + ". " * 50 + "koniec"]),
        (
            "Stress: Long Number Chain (100 numerals)",
            [" ".join(["jeden", "dwa", "trzy", "cztery", "pięć"] * 20)],
        ),
    ]

    for label, sample in stress_patterns:
        res = benchmark_callable(
            norm,
            sample,
            name=label,
            iterations=iterations * 5,
            warmup=warmup,
        )
        results.append(res)

    return results


def format_console_table(results: list[BenchmarkResult]) -> str:
    """Format benchmark results into a clean, aligned console table."""
    headers = [
        "Benchmark / Suite",
        "Calls",
        "Texts/s",
        "Words/s",
        "KB/s",
        "Mean (ms)",
        "p50 (ms)",
        "p95 (ms)",
        "p99 (ms)",
    ]
    rows = []
    for r in results:
        rows.append(
            [
                r.name,
                f"{r.total_calls:,}",
                f"{r.items_per_sec:,.1f}",
                f"{r.words_per_sec:,.1f}" if r.words_per_sec > 0 else "—",
                f"{r.kb_per_sec:,.1f}" if r.kb_per_sec > 0 else "—",
                f"{r.latency_mean_ms:.3f}",
                f"{r.latency_p50_ms:.3f}" if r.latency_p50_ms > 0 else "—",
                f"{r.latency_p95_ms:.3f}" if r.latency_p95_ms > 0 else "—",
                f"{r.latency_p99_ms:.3f}" if r.latency_p99_ms > 0 else "—",
            ]
        )

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    def fmt_row(items: list[str]) -> str:
        parts = []
        for i, (item, width) in enumerate(zip(items, col_widths, strict=True)):
            # Name left-aligned, numbers right-aligned
            if i == 0:
                parts.append(item.ljust(width))
            else:
                parts.append(item.rjust(width))
        return " | ".join(parts)

    separator = "-+-".join("-" * w for w in col_widths)
    header_str = fmt_row(headers)
    table_lines = [header_str, separator]
    table_lines.extend(fmt_row(row) for row in rows)
    return "\n".join(table_lines)


def format_markdown_table(results: list[BenchmarkResult]) -> str:
    """Format benchmark results into standard GitHub Flavored Markdown."""
    headers = [
        "Benchmark",
        "Calls",
        "Throughput (texts/s)",
        "Words/s",
        "Throughput (KB/s)",
        "Mean Latency (ms)",
        "p50 (ms)",
        "p95 (ms)",
        "p99 (ms)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|:---" + "|---:" * (len(headers) - 1) + "|",
    ]
    for r in results:
        cols = [
            r.name,
            f"{r.total_calls:,}",
            f"{r.items_per_sec:,.1f}",
            f"{r.words_per_sec:,.1f}" if r.words_per_sec > 0 else "—",
            f"{r.kb_per_sec:,.1f}" if r.kb_per_sec > 0 else "—",
            f"{r.latency_mean_ms:.3f}",
            f"{r.latency_p50_ms:.3f}" if r.latency_p50_ms > 0 else "—",
            f"{r.latency_p95_ms:.3f}" if r.latency_p95_ms > 0 else "—",
            f"{r.latency_p99_ms:.3f}" if r.latency_p99_ms > 0 else "—",
        ]
        lines.append("| " + " | ".join(cols) + " |")
    return "\n".join(lines)
