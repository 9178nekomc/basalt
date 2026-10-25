#!/usr/bin/env python3
"""Generate JSL benchmark figures and Markdown tables from build/bench/*.csv.

Usage:
    python report/make_figures.py

Requires matplotlib and numpy (standard scientific Python). No JSL dependency.
All figures use a light, print-friendly theme with a colour-blind-safe palette.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "build" / "bench"
OUT = ROOT / "report" / "figures"
TABLES = ROOT / "report" / "tables.md"

INK = "#1f2937"
MUTED = "#6b7280"
GRID = "#e5e7eb"
BLUE = "#2563eb"
TEAL = "#0d9488"
ORANGE = "#ea580c"
RED = "#dc2626"
PURPLE = "#7c3aed"
SLATE = "#475569"

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#d1d5db",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "figure.dpi": 160,
        "savefig.dpi": 160,
    }
)


def read(name: str) -> list[dict[str, str]]:
    path = BENCH / name
    if not path.exists():
        raise SystemExit(f"missing benchmark CSV: {path}; run JslBenchmark first")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(row: dict[str, str], key: str) -> float:
    return float(row[key])


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def annotate(ax, x, y, text, **kwargs) -> None:
    ax.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=(0, 7),
        ha="center",
        fontsize=8,
        color=MUTED,
        **kwargs,
    )


# --------------------------------------------------------------------------- matmul


def fig_matmul() -> None:
    rows = read("matmul.csv")
    sizes = [int(r["size"]) for r in rows]
    scalar = [num(r, "scalar_ms") for r in rows]
    simd = [num(r, "simd_ms") for r in rows]
    speedup = [num(r, "speedup") for r in rows]
    gflops = [num(r, "gflops") for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.9))
    ax1.plot(sizes, scalar, "o-", color=SLATE, lw=1.8, ms=5, label="标量三重循环")
    ax1.plot(sizes, simd, "o-", color=BLUE, lw=2.0, ms=5, label="SIMD + ForkJoin")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("方阵边长 n")
    ax1.set_ylabel("单次乘法耗时 (ms)")
    ax1.set_title("矩阵乘法：标量 vs SIMD 并行")
    ax1.legend(loc="upper left")

    bars = ax2.bar([str(s) for s in sizes], speedup, color=TEAL, width=0.55)
    for bar, value, gflop in zip(bars, speedup, gflops):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.25,
            f"{value:.1f}×\n{gflop:.0f} GF",
            ha="center",
            va="bottom",
            fontsize=7.6,
            color=MUTED,
        )
    ax2.set_xlabel("方阵边长 n")
    ax2.set_ylabel("相对标量加速比")
    ax2.set_title("加速比与吞吐")
    ax2.set_ylim(0, max(speedup) * 1.35)
    fig.tight_layout()
    save(fig, "fig_matmul_scaling.png")


def fig_matmul_threads() -> None:
    rows = read("matmul_threads.csv")
    threads = [int(r["threads"]) for r in rows]
    ms = [num(r, "ms") for r in rows]
    gflops = [num(r, "gflops") for r in rows]
    fig, ax1 = plt.subplots(figsize=(5.4, 3.7))
    ax1.plot(threads, ms, "o-", color=BLUE, lw=2, ms=6)
    ax1.set_xlabel("ForkJoin 并行度")
    ax1.set_ylabel("512×512 耗时 (ms)", color=BLUE)
    ax1.set_xticks(threads)
    ax2 = ax1.twinx()
    ax2.plot(threads, gflops, "s--", color=ORANGE, lw=1.6, ms=5)
    ax2.set_ylabel("吞吐 (GFLOP/s)", color=ORANGE)
    ax2.grid(False)
    ax1.set_title("矩阵乘法线程扩展（512×512）")
    fig.tight_layout()
    save(fig, "fig_matmul_threads.png")


# --------------------------------------------------------------------------- loader


def fig_loader() -> None:
    rows = read("loader.csv")
    modes = ["sequential", "mapped", "mapped-shuffle"]
    labels = {"sequential": "顺序通道读取", "mapped": "内存映射", "mapped-shuffle": "映射 + 洗牌"}
    batches = sorted({int(r["batch_size"]) for r in rows})
    colors = [SLATE, TEAL, ORANGE]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.9))
    width = 0.8 / len(modes)
    for index, mode in enumerate(modes):
        values = []
        for batch in batches:
            match = [r for r in rows if r["mode"] == mode and int(r["batch_size"]) == batch]
            values.append(num(match[0], "mb_per_s") if match else 0.0)
        positions = [i + (index - (len(modes) - 1) / 2) * width for i in range(len(batches))]
        ax1.bar(positions, values, width=width, color=colors[index], label=labels[mode])
        for position, value in zip(positions, values):
            ax1.text(position, value, f"{value:.0f}", ha="center", va="bottom",
                     fontsize=7.2, color=MUTED)
    ax1.set_xticks(range(len(batches)))
    ax1.set_xticklabels([str(b) for b in batches])
    ax1.set_xlabel("批次大小")
    ax1.set_ylabel("吞吐 (MiB/s)")
    ax1.set_title("JSLD 数据管线吞吐")
    ax1.legend(loc="upper left", ncol=3, fontsize=8)

    for index, mode in enumerate(modes):
        values = []
        for batch in batches:
            match = [r for r in rows if r["mode"] == mode and int(r["batch_size"]) == batch]
            values.append(num(match[0], "bytes_per_row") if match else 0.0)
        positions = [i + (index - (len(modes) - 1) / 2) * width for i in range(len(batches))]
        ax2.bar(positions, values, width=width, color=colors[index], label=labels[mode])
    ax2.set_xticks(range(len(batches)))
    ax2.set_xticklabels([str(b) for b in batches])
    ax2.set_xlabel("批次大小")
    ax2.set_ylabel("每行分摊分配字节 (B/row)")
    ax2.set_title("每轮迭代的堆分配（越低越好）")
    fig.tight_layout()
    save(fig, "fig_loader.png")


# --------------------------------------------------------------------------- trees


def fig_boosting() -> None:
    rows = read("boosting.csv")
    algorithms = sorted({r["algorithm"] for r in rows})
    colors = {"XGBOOST": BLUE, "LIGHTGBM": ORANGE}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.9))
    for algorithm in algorithms:
        subset = sorted([r for r in rows if r["algorithm"] == algorithm],
                        key=lambda r: int(r["threads"]))
        threads = [int(r["threads"]) for r in subset]
        times = [num(r, "ms") for r in subset]
        ax1.plot(threads, times, "o-", color=colors.get(algorithm, PURPLE), lw=2, ms=5,
                 label=algorithm)
    ax1.set_xticks([1, 2, 4, 8])
    ax1.set_xlabel("线程数")
    ax1.set_ylabel("30 棵树训练耗时 (ms)")
    ax1.set_title("小规模 5k×16：同步开销主导")
    ax1.legend()

    scale = read("boosting_scale.csv")
    threads = [int(r["threads"]) for r in scale]
    times = [num(r, "ms") for r in scale]
    ax2.plot(threads, times, "o-", color=TEAL, lw=2, ms=6)
    base = times[0]
    for thread, value in zip(threads, times):
        ax2.annotate(f"{value:.0f} ms\n{base / value:.1f}×", (thread, value),
                     textcoords="offset points", xytext=(0, 8), ha="center",
                     fontsize=7.6, color=MUTED)
    ax2.set_xticks(threads)
    ax2.set_xlabel("线程数")
    ax2.set_ylabel("20 棵树训练耗时 (ms)")
    ax2.set_title("大规模 20k×64：并行开始生效")
    fig.tight_layout()
    save(fig, "fig_boosting.png")


def fig_forest() -> None:
    rows = read("forest.csv")
    thread_rows = sorted([r for r in rows if int(r["trees"]) == 25],
                         key=lambda r: int(r["threads"]))
    tree_rows = sorted([r for r in rows if int(r["threads"]) == 8 and int(r["trees"]) != 25],
                       key=lambda r: int(r["trees"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.9))
    threads = [int(r["threads"]) for r in thread_rows]
    times = [num(r, "ms") for r in thread_rows]
    ax1.plot(threads, times, "o-", color=TEAL, lw=2, ms=6)
    for thread, value in zip(threads, times):
        annotate(ax1, thread, value, f"{value:.0f} ms")
    ax1.set_xticks(threads)
    ax1.set_xlabel("线程数")
    ax1.set_ylabel("25 棵树训练耗时 (ms)")
    ax1.set_title("随机森林：线程扩展（5k×16）")

    trees = [int(r["trees"]) for r in tree_rows]
    times = [num(r, "ms") for r in tree_rows]
    ax2.plot(trees, times, "s-", color=ORANGE, lw=2, ms=6)
    for tree, value in zip(trees, times):
        annotate(ax2, tree, value, f"{value:.0f} ms")
    ax2.set_xticks(trees)
    ax2.set_xlabel("树数量")
    ax2.set_ylabel("训练耗时 (ms)")
    ax2.set_title("随机森林：树数量扩展（8 线程）")
    fig.tight_layout()
    save(fig, "fig_forest.png")


# --------------------------------------------------------------------------- unsupervised


def fig_kmeans_pca() -> None:
    rows = read("kmeans.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.9))
    row_counts = sorted({int(r["rows"]) for r in rows})
    colors = [BLUE, TEAL]
    for index, row_count in enumerate(row_counts):
        subset = sorted([r for r in rows if int(r["rows"]) == row_count],
                        key=lambda r: int(r["clusters"]))
        clusters = [int(r["clusters"]) for r in subset]
        times = [num(r, "ms") for r in subset]
        ax1.plot(clusters, times, "o-", color=colors[index % len(colors)], lw=2, ms=5,
                 label=f"{row_count:,} 行")
    ax1.set_xticks(sorted({int(r["clusters"]) for r in rows}))
    ax1.set_xlabel("簇数量 k")
    ax1.set_ylabel("单次拟合耗时 (ms)")
    ax1.set_title("K-Means++：规模扩展（8 线程）")
    ax1.legend()

    pca_rows = read("pca.csv")
    names = [r["solver"] for r in pca_rows]
    times = [num(r, "ms") for r in pca_rows]
    bars = ax2.barh(names, times, color=[PURPLE, TEAL, BLUE][: len(names)], height=0.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("拟合耗时 (ms, 对数轴)")
    ax2.set_title("PCA：协方差 Jacobi vs Gram 对偶")
    for bar, row in zip(bars, pca_rows):
        ax2.text(bar.get_width() * 1.08, bar.get_y() + bar.get_height() / 2,
                 f"{row['rows']}×{row['features']}, EVR={float(row['ratio_sum']):.2f}",
                 va="center", fontsize=7.6, color=MUTED)
    fig.tight_layout()
    save(fig, "fig_kmeans_pca.png")


# --------------------------------------------------------------------------- sequence


def fig_sequence() -> None:
    rows = read("sequence.csv")
    layers = ["RNN", "GRU", "LSTM", "Attention"]
    colors = {"RNN": SLATE, "GRU": ORANGE, "LSTM": BLUE, "Attention": TEAL}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for layer in layers:
        subset = sorted([r for r in rows if r["layer"] == layer], key=lambda r: int(r["seq"]))
        if not subset:
            continue
        sequences = [int(r["seq"]) for r in subset]
        times = [num(r, "ms_per_step") for r in subset]
        ax.plot(sequences, times, "o-", color=colors[layer], lw=2, ms=5, label=layer)
    ax.set_yscale("log")
    ax.set_xlabel("序列长度 T")
    ax.set_ylabel("单步训练耗时 (ms, 对数轴)")
    ax.set_title("序列层前向+反向：B=16, H=32")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_sequence.png")


# --------------------------------------------------------------------------- serialization


def fig_serialization() -> None:
    rows = read("serialization.csv")
    names = [r["model"] for r in rows]
    sizes = [num(r, "bytes") / 1024.0 for r in rows]
    save_ms = [num(r, "save_ms") for r in rows]
    load_ms = [num(r, "load_ms") for r in rows]
    infer_us = [num(r, "infer_us") for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))
    colors = [BLUE, TEAL, ORANGE, PURPLE, RED][: len(names)]
    axes[0].bar(names, sizes, color=colors, width=0.55)
    axes[0].set_ylabel("文件大小 (KiB)")
    axes[0].set_title(".jsl 体积")
    axes[0].tick_params(axis="x", rotation=25)

    width = 0.38
    positions = range(len(names))
    axes[1].bar([p - width / 2 for p in positions], save_ms, width, color=BLUE, label="保存")
    axes[1].bar([p + width / 2 for p in positions], load_ms, width, color=TEAL, label="加载")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("耗时 (ms, 对数轴)")
    axes[1].set_title("存盘与加载")
    axes[1].set_xticks(list(positions))
    axes[1].set_xticklabels(names, rotation=25, ha="right")
    axes[1].legend()

    axes[2].bar(names, infer_us, color=ORANGE, width=0.55)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("批量推理 (µs, 对数轴)")
    axes[2].set_title("加载后推理延迟")
    axes[2].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save(fig, "fig_serialization.png")


# --------------------------------------------------------------------------- convergence


def fig_convergence() -> None:
    rows = read("convergence_gbdt.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.9))
    colors = {"XGBOOST": BLUE, "LIGHTGBM": ORANGE}
    for algorithm in sorted({r["algorithm"] for r in rows}):
        subset = sorted([r for r in rows if r["algorithm"] == algorithm],
                        key=lambda r: int(r["iteration"]))
        iterations = [int(r["iteration"]) for r in subset]
        train = [num(r, "train_loss") for r in subset]
        valid = [num(r, "val_loss") for r in subset]
        color = colors.get(algorithm, PURPLE)
        ax1.plot(iterations, train, "-", color=color, lw=1.8, label=f"{algorithm} 训练")
        ax1.plot(iterations, valid, "--", color=color, lw=1.4, alpha=0.75,
                 label=f"{algorithm} 验证")
    ax1.set_yscale("log")
    ax1.set_xlabel("提升轮次")
    ax1.set_ylabel("MSE (对数轴)")
    ax1.set_title("GBDT 收敛：训练 vs 验证")
    ax1.legend(fontsize=8)

    sequence = read("convergence_sequence.csv")
    epochs = [int(r["epoch"]) for r in sequence]
    losses = [num(r, "loss") for r in sequence]
    ax2.plot(epochs, losses, "-", color=TEAL, lw=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("交叉熵 (对数轴)")
    ax2.set_title("LSTM 序列分类收敛（记忆首 token）")
    ax2.annotate(f"final={losses[-1]:.4f}", (epochs[-1], losses[-1]),
                 textcoords="offset points", xytext=(-8, 12), ha="right", fontsize=8,
                 color=MUTED)
    fig.tight_layout()
    save(fig, "fig_convergence.png")


def fig_gradient_decay() -> None:
    rows = read("gradient_decay.csv")
    steps = [int(r["timestep"]) for r in rows]
    lstm = [max(num(r, "lstm_norm"), 1e-300) for r in rows]
    rnn = [max(num(r, "rnn_norm"), 1e-300) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.semilogy(steps, lstm, "-", color=BLUE, lw=2, label="LSTM（遗忘门 ≈ 0.993）")
    ax.semilogy(steps, rnn, "-", color=RED, lw=2, label="Vanilla RNN（U=0.1）")
    ax.set_xlabel("序列长度 T")
    ax.set_ylabel("t=0 处输入梯度范数（对数轴）")
    ax.set_title("长程梯度传播：常量误差传送带 vs 连乘雅可比")
    ax.legend()
    ratio = lstm[-1] / rnn[-1]
    ax.annotate(f"T={steps[-1]} 时相差 {ratio:.1e}×", (steps[-1], lstm[-1]),
                textcoords="offset points", xytext=(-12, 14), ha="right", fontsize=8.5,
                color=MUTED)
    fig.tight_layout()
    save(fig, "fig_gradient_decay.png")


# --------------------------------------------------------------------------- tables


def markdown_tables() -> None:
    lines: list[str] = ["# JSL 基准数据表（自动生成）", ""]

    rows = read("matmul.csv")
    lines += ["## 矩阵乘法", "", "| n | 标量 (ms) | SIMD+FJ (ms) | 加速比 | GFLOP/s |",
              "|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['size']} | {float(row['scalar_ms']):.3f} | "
                     f"{float(row['simd_ms']):.3f} | {float(row['speedup']):.2f}× | "
                     f"{float(row['gflops']):.2f} |")
    lines.append("")

    rows = read("matmul_threads.csv")
    lines += ["## 矩阵乘法线程扩展（512×512）", "", "| 线程 | 耗时 (ms) | GFLOP/s |",
              "|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['threads']} | {float(row['ms']):.3f} | "
                     f"{float(row['gflops']):.2f} |")
    lines.append("")

    rows = read("loader.csv")
    lines += ["## 数据管线", "", "| 模式 | 批次 | MiB/s | 行/s | B/row |",
              "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['mode']} | {row['batch_size']} | {float(row['mb_per_s']):.1f} | "
                     f"{float(row['rows_per_s']):.0f} | {float(row['bytes_per_row']):.2f} |")
    lines.append("")

    rows = read("boosting.csv")
    lines += ["## 直方图提升（5k×16，30 棵树）", "", "| 算法 | 线程 | 耗时 (ms) |",
              "|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['algorithm']} | {row['threads']} | {float(row['ms']):.1f} |")
    lines.append("")

    if (BENCH / "boosting_scale.csv").exists():
        rows = read("boosting_scale.csv")
        lines += ["## 直方图提升大规模扩展（20k×64，20 棵树，LightGBM）", "",
                  "| 线程 | 耗时 (ms) | 相对 1 线程 |", "|---:|---:|---:|"]
        base = float(rows[0]["ms"])
        for row in rows:
            lines.append(f"| {row['threads']} | {float(row['ms']):.1f} | "
                         f"{base / float(row['ms']):.2f}× |")
        lines.append("")

    rows = read("forest.csv")
    lines += ["## 随机森林（5k×16）", "", "| 线程 | 树 | 耗时 (ms) |", "|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['threads']} | {row['trees']} | {float(row['ms']):.1f} |")
    lines.append("")

    rows = read("kmeans.csv")
    lines += ["## K-Means++（8 线程）", "", "| 行 | k | 耗时 (ms) | inertia | 迭代 |",
              "|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['rows']} | {row['clusters']} | {float(row['ms']):.1f} | "
                     f"{float(row['inertia']):.1f} | {row['iterations']} |")
    lines.append("")

    rows = read("pca.csv")
    lines += ["## PCA", "", "| 求解器 | 形状 | 耗时 (ms) | 解释方差和 |", "|---|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['solver']} | {row['rows']}×{row['features']} | "
                     f"{float(row['ms']):.1f} | {float(row['ratio_sum']):.4f} |")
    lines.append("")

    rows = read("sequence.csv")
    lines += ["## 序列层（B=16, H=32）", "", "| 层 | T | ms/step |", "|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['layer']} | {row['seq']} | {float(row['ms_per_step']):.3f} |")
    lines.append("")

    rows = read("serialization.csv")
    lines += ["## 序列化", "", "| 模型 | 字节 | 保存 (ms) | 加载 (ms) | 批量推理 (µs) |",
              "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['bytes']} | {float(row['save_ms']):.3f} | "
                     f"{float(row['load_ms']):.3f} | {float(row['infer_us']):.1f} |")
    lines.append("")

    TABLES.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {TABLES.relative_to(ROOT)}")


def main() -> None:
    if not BENCH.exists():
        raise SystemExit(f"benchmark directory not found: {BENCH}")
    OUT.mkdir(parents=True, exist_ok=True)
    fig_matmul()
    fig_matmul_threads()
    fig_loader()
    fig_boosting()
    fig_forest()
    fig_kmeans_pca()
    fig_sequence()
    fig_serialization()
    fig_convergence()
    fig_gradient_decay()
    markdown_tables()


if __name__ == "__main__":
    main()
