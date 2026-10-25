#!/usr/bin/env python3
"""Generate the shared JSLD datasets and benchmark scikit-learn on them.

The Java harness (jsl.tests.SklearnComparison) reads the exact same JSLD files, so both
libraries see bit-identical features and labels. Results are written to build/compare/sklearn.csv.
"""

from __future__ import annotations

import csv
import os
import pickle
import struct
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "compare"
MAGIC = 0x4A534C44
VERSION = 1
REPETITIONS = 5


def write_jsld(path: Path, features: np.ndarray, labels: np.ndarray) -> None:
    rows, columns = features.shape
    payload = np.empty((rows, columns + 1), dtype=">f8")
    payload[:, :columns] = features
    payload[:, columns] = labels
    with path.open("wb") as handle:
        handle.write(struct.pack(">iiii", MAGIC, VERSION, columns, 0))
        payload.tofile(handle)


def make_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261024)
    rows, columns = 30_000, 32
    features = rng.normal(size=(rows, columns))
    weights = rng.normal(size=columns)
    logits = (
        3.0 * (features @ weights) / np.sqrt(columns)
        + 1.2 * np.sin(features[:, 0] * 1.5)
        + 1.0 * features[:, 1] * features[:, 2]
    )
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.random(rows) < probabilities).astype(np.float64)
    regression_weights = rng.normal(size=columns)
    targets = (
        2.5 * (features @ regression_weights) / np.sqrt(columns)
        + 0.7 * np.sin(features[:, 3])
        + 0.2 * rng.normal(size=rows)
    )
    split = 20_000
    return (
        features[:split], labels[:split], targets[:split],
        features[split:], labels[split:], targets[split:],
    )


def timed(factory, fit_args, predict_args, predict_method="predict"):
    factory().fit(*fit_args)  # warm-up, including lazy allocations
    fit_times: list[float] = []
    predict_times: list[float] = []
    for _ in range(REPETITIONS):
        model = factory()
        start = time.perf_counter()
        model.fit(*fit_args)
        fit_times.append((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        getattr(model, predict_method)(*predict_args)
        predict_times.append((time.perf_counter() - start) * 1000.0)
    fit_times.sort()
    predict_times.sort()
    return model, fit_times[len(fit_times) // 2], predict_times[len(predict_times) // 2]


def size_of(model) -> int:
    return len(pickle.dumps(model, protocol=5))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_x, train_y, train_t, test_x, test_y, test_t = make_data()
    write_jsld(OUT / "clf_train.jsld", train_x, train_y)
    write_jsld(OUT / "clf_test.jsld", test_x, test_y)
    write_jsld(OUT / "reg_train.jsld", train_x, train_t)
    write_jsld(OUT / "reg_test.jsld", test_x, test_t)
    print(f"data: train={train_x.shape}, test={test_x.shape}, "
          f"class balance={train_y.mean():.3f}")

    rows: list[dict[str, object]] = []

    def record(task: str, model: str, fitted, fit_ms: float, predict_ms: float,
               score: float, score_name: str) -> None:
        rows.append(
            {
                "task": task,
                "model": model,
                "fit_ms": f"{fit_ms:.4f}",
                "predict_ms": f"{predict_ms:.4f}",
                "score": f"{score:.8f}",
                "score_name": score_name,
                "model_bytes": size_of(fitted),
            }
        )
        print(f"sklearn {task:14s} {model:28s} fit={fit_ms:8.2f} ms "
              f"pred={predict_ms:7.2f} ms {score_name}={score:.5f}")

    model, fit_ms, predict_ms = timed(
        lambda: RandomForestClassifier(
            n_estimators=100, max_depth=8, max_features="sqrt", min_samples_leaf=1,
            n_jobs=8, random_state=42,
        ),
        (train_x, train_y), (test_x,),
    )
    record("clf", "RandomForest", model, fit_ms, predict_ms,
           accuracy_score(test_y, model.predict(test_x)), "accuracy")

    model, fit_ms, predict_ms = timed(
        lambda: HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.1, max_leaf_nodes=31, max_depth=6,
            max_bins=64, min_samples_leaf=20, l2_regularization=1.0, early_stopping=False,
            random_state=42,
        ),
        (train_x, train_y), (test_x,),
    )
    record("clf", "HistGradientBoosting", model, fit_ms, predict_ms,
           accuracy_score(test_y, model.predict(test_x)), "accuracy")

    model, fit_ms, predict_ms = timed(
        lambda: GaussianNB(),
        (train_x, train_y), (test_x,),
    )
    record("clf", "GaussianNB", model, fit_ms, predict_ms,
           accuracy_score(test_y, model.predict(test_x)), "accuracy")

    model, fit_ms, predict_ms = timed(
        lambda: KMeans(n_clusters=8, n_init=4, max_iter=300, random_state=42),
        (train_x,), (test_x,),
    )
    record("unsupervised", "KMeans", model, fit_ms, predict_ms,
           float(model.inertia_), "inertia")

    model, fit_ms, predict_ms = timed(
        lambda: PCA(n_components=8, svd_solver="full"),
        (train_x,), (test_x,), "transform",
    )
    ratio_sum = float(model.explained_variance_ratio_.sum())
    record("unsupervised", "PCA", model, fit_ms, predict_ms, ratio_sum, "evr_sum")

    model, fit_ms, predict_ms = timed(
        lambda: RandomForestRegressor(
            n_estimators=100, max_depth=8, max_features="sqrt", min_samples_leaf=1,
            n_jobs=8, random_state=42,
        ),
        (train_x, train_t), (test_x,),
    )
    record("reg", "RandomForestRegressor", model, fit_ms, predict_ms,
           r2_score(test_t, model.predict(test_x)), "r2")

    model, fit_ms, predict_ms = timed(
        lambda: Ridge(alpha=1.0),
        (train_x, train_t), (test_x,),
    )
    record("reg", "Ridge", model, fit_ms, predict_ms,
           r2_score(test_t, model.predict(test_x)), "r2")

    model, fit_ms, predict_ms = timed(
        lambda: Lasso(alpha=0.1, max_iter=2000),
        (train_x, train_t), (test_x,),
    )
    record("reg", "Lasso", model, fit_ms, predict_ms,
           r2_score(test_t, model.predict(test_x)), "r2")

    model, fit_ms, predict_ms = timed(
        lambda: LinearSVC(C=1.0, max_iter=2000),
        (train_x, train_y), (test_x,),
    )
    record("clf", "LinearSVM", model, fit_ms, predict_ms,
           accuracy_score(test_y, model.predict(test_x)), "accuracy")

    model, fit_ms, predict_ms = timed(
        lambda: HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.1, max_leaf_nodes=31, max_depth=6,
            max_bins=64, min_samples_leaf=20, l2_regularization=1.0, early_stopping=False,
            random_state=42,
        ),
        (train_x, train_t), (test_x,),
    )
    record("reg", "HistGradientBoostingRegressor", model, fit_ms, predict_ms,
           r2_score(test_t, model.predict(test_x)), "r2")

    path = OUT / "sklearn.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
