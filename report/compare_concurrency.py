#!/usr/bin/env python3
"""Benchmark Python thread/process orchestration for the Basalt concurrency comparison.

CPU-bound Python needs processes because of the GIL; I/O-bound Python can use threads. The Basalt
side writes build/compare/vthreads.csv; this script writes python_concurrency.csv.
"""

from __future__ import annotations

import csv
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "compare"
TASKS = 4_000
PARALLELISM = 8


BATCH = np.random.default_rng(1).normal(size=(16, 32))
WEIGHTS = np.random.default_rng(2).normal(size=32)


def cpu_task() -> float:
    """Same shape as one Basalt batch: 16x32 features dotted with a 32-vector."""
    return float(np.dot(BATCH, WEIGHTS).sum())


def io_task() -> None:
    time.sleep(0.001)


def cpu_job(_index: int) -> float:
    return cpu_task()


def io_job(_index: int) -> None:
    io_task()


def measure(factory, task, workers: int) -> float:
    start = time.perf_counter()
    with factory(max_workers=workers) as pool:
        list(pool.map(task, range(TASKS)))
    return (time.perf_counter() - start) * 1000.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    # Warm-up
    for _ in range(200):
        cpu_task()
    io_task()

    wall = measure(ThreadPoolExecutor, cpu_job, PARALLELISM)
    rows.append(("cpu", "python-threads", TASKS, wall, TASKS / (wall / 1000.0)))
    print(f"python cpu threads: {wall:.1f} ms")

    wall = measure(ProcessPoolExecutor, cpu_job, PARALLELISM)
    rows.append(("cpu", "python-processes", TASKS, wall, TASKS / (wall / 1000.0)))
    print(f"python cpu processes: {wall:.1f} ms")

    wall = measure(ThreadPoolExecutor, io_job, 64)
    rows.append(("io", "python-threads", TASKS, wall, TASKS / (wall / 1000.0)))
    print(f"python io threads: {wall:.1f} ms")

    wall = measure(ProcessPoolExecutor, io_job, PARALLELISM)
    rows.append(("io", "python-processes", TASKS, wall, TASKS / (wall / 1000.0)))
    print(f"python io processes: {wall:.1f} ms")

    path = OUT / "python_concurrency.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "mode", "tasks", "wall_ms", "tasks_per_s"])
        for row in rows:
            writer.writerow([row[0], row[1], row[2], f"{row[3]:.3f}", f"{row[4]:.1f}"])
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
