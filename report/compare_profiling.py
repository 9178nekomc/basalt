#!/usr/bin/env python3
"""Measure cProfile overhead on the same model-inference workload shape as the Basalt JFR harness.

Two orchestration granularities are measured, because cProfile instruments every Python-level
call: a coarse pipeline (one vectorized step per request) and a fine-grained pipeline (one
function call per feature element). JFR's sampling approach does not depend on either.
"""

from __future__ import annotations

import csv
import cProfile
import pstats
import time
from pathlib import Path

import numpy as np
from sklearn.naive_bayes import GaussianNB

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "compare"
REQUESTS = 4_000
BATCH_ROWS = 16
FEATURES = 32


def train() -> GaussianNB:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(500, FEATURES))
    signal = (x * np.arange(FEATURES % 3 - 1)).sum(axis=1)
    y = (signal > 0).astype(float)
    return GaussianNB().fit(x, y)


def coarse_step(model: GaussianNB, batch: np.ndarray) -> float:
    scaled = batch * 0.5 + 0.1
    return float(model.predict(scaled).sum())


def clamp(value: float) -> float:
    return value * 0.5 + 0.1


def fine_step(model: GaussianNB, batch: np.ndarray) -> float:
    scaled = [[clamp(v) for v in row] for row in batch]
    return float(model.predict(np.asarray(scaled)).sum())


def run(model, batches, step) -> float:
    sink = 0.0
    for batch in batches:
        sink += step(model, batch)
    return sink


def timed(model, batches, step) -> float:
    run(model, batches[:200], step)
    start = time.perf_counter()
    run(model, batches, step)
    return (time.perf_counter() - start) * 1000.0


def profiled(model, batches, step) -> float:
    profiler = cProfile.Profile()
    profiler.enable()
    run(model, batches, step)
    profiler.disable()
    return float(pstats.Stats(profiler).total_calls)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = train()
    rng = np.random.default_rng(5)
    batches = [rng.normal(size=(BATCH_ROWS, FEATURES)) for _ in range(REQUESTS)]

    rows = []
    for name, step in (("coarse", coarse_step), ("fine", fine_step)):
        baseline = timed(model, batches, step)
        calls = profiled(model, batches, step)
        profiled_ms = timed(model, batches, step)
        # measure the profiled wall time properly: run once under the profiler
        start = time.perf_counter()
        profiler = cProfile.Profile()
        profiler.enable()
        run(model, batches, step)
        profiler.disable()
        profiled_ms = (time.perf_counter() - start) * 1000.0
        stats = pstats.Stats(profiler)
        overhead = (profiled_ms - baseline) / baseline * 100.0
        rows.append((name, baseline, profiled_ms, overhead, stats.total_calls))
        print(f"python cProfile ({name}): baseline={baseline:.1f} ms, "
              f"profiled={profiled_ms:.1f} ms, overhead=+{overhead:.1f}%, "
              f"calls={stats.total_calls}")

    path = OUT / "profiling_python.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "baseline_ms", "profiled_ms", "overhead_percent", "calls"])
        for name, baseline, profiled_ms, overhead, calls in rows:
            writer.writerow([f"Python + cProfile ({name})",
                             f"{baseline:.4f}", f"{profiled_ms:.4f}",
                             f"{overhead:.2f}", calls])
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
