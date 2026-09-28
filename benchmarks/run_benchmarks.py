#!/usr/bin/env python3
"""CLI runner for Polish Whisper Normalizer benchmarks.

Usage:
    uv run python benchmarks/run_benchmarks.py
    uv run python benchmarks/run_benchmarks.py --suite workloads --markdown
    uv run python benchmarks/run_benchmarks.py --suite components --json results.json
    uv run python benchmarks/run_benchmarks.py --suite scaling
    uv run python benchmarks/run_benchmarks.py --suite all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from .runner import (
        BenchmarkResult,
        format_console_table,
        format_markdown_table,
        run_component_benchmarks,
        run_scaling_benchmark,
        run_stress_benchmarks,
        run_workload_benchmarks,
    )
except ImportError:
    from benchmarks.runner import (  # type: ignore[no-redef]
        BenchmarkResult,
        format_console_table,
        format_markdown_table,
        run_component_benchmarks,
        run_scaling_benchmark,
        run_stress_benchmarks,
        run_workload_benchmarks,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive Benchmark Suite for Polish Whisper Normalizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--suite",
        choices=["all", "workloads", "components", "scaling", "stress"],
        default="all",
        help="Which benchmark suite to execute",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of iterations per dataset",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup passes before measurement",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to output raw JSON results",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print report in GitHub-flavored Markdown table format",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 80)
    print("POLISH WHISPER NORMALIZER — BENCHMARK SUITE")
    print(f"Suite: {args.suite} | Iterations: {args.iterations} | Warmup: {args.warmup}")
    print("=" * 80)

    all_results: list[BenchmarkResult] = []
    t_start = time.perf_counter()

    if args.suite in ("all", "workloads"):
        print("\n>>> Running Workload Benchmarks...")
        workload_res = run_workload_benchmarks(
            iterations=args.iterations,
            warmup=args.warmup,
        )
        all_results.extend(workload_res)

    if args.suite in ("all", "components"):
        print("\n>>> Running Component Isolation Benchmarks...")
        comp_res = run_component_benchmarks(
            iterations=args.iterations,
            warmup=args.warmup,
        )
        all_results.extend(comp_res)

    if args.suite in ("all", "scaling"):
        print("\n>>> Running Multi-threaded Scaling Benchmarks...")
        scale_res = run_scaling_benchmark(
            worker_counts=(1, 2, 4, 8),
            iterations_per_worker=args.iterations,
        )
        all_results.extend(scale_res)

    if args.suite in ("all", "stress"):
        print("\n>>> Running Adversarial & Stress Benchmarks...")
        stress_res = run_stress_benchmarks(
            iterations=args.iterations,
            warmup=args.warmup,
        )
        all_results.extend(stress_res)

    t_total = time.perf_counter() - t_start

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    if args.markdown:
        print("\n" + format_markdown_table(all_results) + "\n")
    else:
        print("\n" + format_console_table(all_results) + "\n")

    print(f"Total benchmark execution time: {t_total:.2f}s across {len(all_results)} suites.")

    if args.json:
        output_data: dict[str, Any] = {
            "meta": {
                "timestamp": time.time(),
                "total_time_sec": round(t_total, 3),
                "suite": args.suite,
                "iterations": args.iterations,
                "warmup": args.warmup,
            },
            "results": [r.to_dict() for r in all_results],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved benchmark results to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
