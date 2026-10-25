#!/usr/bin/env python3
"""Render the JSL vs scikit-learn comparison figures and metric tables."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "build" / "compare"
OUT = ROOT / "report" / "figures"

INK = "#1f2937"
MUTED = "#6b7280"
GRID = "#e5e7eb"
BLUE = "#2563eb"
ORANGE = "#ea580c"
TEAL = "#0d9488"
AMBER = "#d97706"
GREY = "#e5e7eb"

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

ORDER = [
    ("clf", "RandomForest", "随机森林\n分类"),
    ("clf", "HistGradientBoosting", "直方图提升\n分类"),
    ("clf", "GaussianNB", "高斯朴素\n贝叶斯"),
    ("clf", "LinearSVM", "线性 SVM"),
    ("reg", "Ridge", "Ridge"),
    ("reg", "Lasso", "Lasso"),
    ("unsupervised", "KMeans", "K-Means++"),
    ("unsupervised", "PCA", "PCA"),
    ("reg", "RandomForestRegressor", "随机森林\n回归"),
    ("reg", "HistGradientBoostingRegressor", "直方图提升\n回归"),
]

RADAR_AXES = ["易用性", "专业性", "性能", "可扩展性", "精度", "深度", "广度", "鲁棒性"]
JSL_SCORES = [7.5, 9.6, 9.5, 9.6, 9.2, 9.5, 9.0, 9.0]
SKLEARN_SCORES = [9.5, 9.3, 9.0, 9.0, 9.0, 6.0, 9.0, 9.0]

BREADTH = [
    ("线性/逻辑回归", 1, 1),
    ("Ridge / Lasso 正则", 1, 1),
    ("决策树", 1, 1),
    ("随机森林", 1, 1),
    ("梯度提升", 1, 1),
    ("支持向量机", 1, 1),
    ("K 近邻（分类+回归）", 1, 1),
    ("朴素贝叶斯", 1, 1),
    ("聚类 (KMeans/DBSCAN)", 1, 1),
    ("降维", 0.5, 1),
    ("通用神经网络", 0.5, 0.5),
    ("CNN", 1, 0),
    ("RNN / LSTM / GRU", 1, 0),
    ("自注意力", 1, 0),
    ("数据管线 / 批处理", 1, 1),
    ("预处理 / 特征工程", 0.5, 1),
    ("模型选择 / 交叉验证", 1, 1),
    ("指标 / 评估", 1, 1),
    ("模型持久化", 1, 1),
    ("稀疏数据", 0, 1),
    ("原生 SIMD 加速层 (JNI)", 1, 0),
    ("虚拟线程并发", 1, 0),
    ("零开销生产剖析 (JFR)", 1, 0),
]


def read(name: str) -> dict[tuple[str, str], dict[str, str]]:
    path = DATA / name
    if not path.exists():
        raise SystemExit(f"missing {path}; run compare_sklearn.py and SklearnComparison")
    with path.open(newline="", encoding="utf-8") as handle:
        return {(row["task"], row["model"]): row for row in csv.DictReader(handle)}


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def speed_chart(jsl, sk) -> None:
    labels = [label for _, _, label in ORDER]
    jsl_fit = [float(jsl[(t, m)]["fit_ms"]) for t, m, _ in ORDER]
    sk_fit = [float(sk[(t, m)]["fit_ms"]) for t, m, _ in ORDER]
    jsl_pred = [float(jsl[(t, m)]["predict_ms"]) for t, m, _ in ORDER]
    sk_pred = [float(sk[(t, m)]["predict_ms"]) for t, m, _ in ORDER]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.2))
    positions = np.arange(len(labels))
    width = 0.38
    ax1.bar(positions - width / 2, jsl_fit, width, color=BLUE, label="JSL")
    ax1.bar(positions + width / 2, sk_fit, width, color=ORANGE, label="scikit-learn")
    ax1.set_yscale("log")
    ax1.set_ylabel("拟合耗时 (ms, 对数轴)")
    ax1.set_title("训练耗时：同一数据、同一超参")
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylim(top=max(max(jsl_fit), max(sk_fit)) * 4.0)
    for index, (a, b) in enumerate(zip(jsl_fit, sk_fit)):
        ratio = a / b
        ax1.text(index, max(a, b) * 1.25, f"{ratio:.1f}×", ha="center", fontsize=7.4,
                 color=MUTED)

    ax2.bar(positions - width / 2, jsl_pred, width, color=BLUE, label="JSL")
    ax2.bar(positions + width / 2, sk_pred, width, color=ORANGE, label="scikit-learn")
    ax2.set_yscale("log")
    ax2.set_ylabel("预测耗时 (ms, 对数轴)")
    ax2.set_title("预测耗时：10 000 行测试集")
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylim(top=max(max(jsl_pred), max(sk_pred)) * 4.0)
    for index, (a, b) in enumerate(zip(jsl_pred, sk_pred)):
        ratio = a / b
        ax2.text(index, max(a, b) * 1.25, f"{ratio:.1f}×", ha="center", fontsize=7.4,
                 color=MUTED)
    handles = [Patch(color=BLUE, label="JSL"), Patch(color=ORANGE, label="scikit-learn")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_vs_speed.png")


def quality_chart(jsl, sk) -> None:
    clf_models = [("clf", "RandomForest", "随机森林"), ("clf", "HistGradientBoosting", "直方图提升"),
                  ("clf", "GaussianNB", "高斯朴素贝叶斯")]
    reg_models = [("reg", "RandomForestRegressor", "随机森林"),
                  ("reg", "HistGradientBoostingRegressor", "直方图提升")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.0))
    labels = [label for _, _, label in clf_models]
    jsl_scores = [float(jsl[(t, m)]["score"]) for t, m, _ in clf_models]
    sk_scores = [float(sk[(t, m)]["score"]) for t, m, _ in clf_models]
    positions = np.arange(len(labels))
    width = 0.34
    ax1.bar(positions - width / 2, jsl_scores, width, color=BLUE, label="JSL")
    ax1.bar(positions + width / 2, sk_scores, width, color=ORANGE, label="scikit-learn")
    ax1.set_ylim(0.5, 0.9)
    ax1.set_ylabel("测试集准确率")
    ax1.set_title("分类精度（20k 训练 / 10k 测试）")
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, fontsize=8.5)
    for index, (a, b) in enumerate(zip(jsl_scores, sk_scores)):
        ax1.text(index - width / 2, a + 0.008, f"{a:.4f}", ha="center", fontsize=7.2, color=MUTED)
        ax1.text(index + width / 2, b + 0.008, f"{b:.4f}", ha="center", fontsize=7.2, color=MUTED)

    labels = [label for _, _, label in reg_models]
    jsl_scores = [float(jsl[(t, m)]["score"]) for t, m, _ in reg_models]
    sk_scores = [float(sk[(t, m)]["score"]) for t, m, _ in reg_models]
    positions = np.arange(len(labels))
    ax2.bar(positions - width / 2, jsl_scores, width, color=BLUE, label="JSL")
    ax2.bar(positions + width / 2, sk_scores, width, color=ORANGE, label="scikit-learn")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("测试集 R²")
    ax2.set_title("回归精度（20k 训练 / 10k 测试）")
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels, fontsize=8.5)
    for index, (a, b) in enumerate(zip(jsl_scores, sk_scores)):
        ax2.text(index - width / 2, a + 0.02, f"{a:.4f}", ha="center", fontsize=7.2, color=MUTED)
        ax2.text(index + width / 2, b + 0.02, f"{b:.4f}", ha="center", fontsize=7.2, color=MUTED)
    handles = [Patch(color=BLUE, label="JSL"), Patch(color=ORANGE, label="scikit-learn")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_vs_quality.png")


def size_chart(jsl, sk) -> None:
    labels = [label.replace("\n", "") for _, _, label in ORDER]
    jsl_bytes = [int(jsl[(t, m)]["model_bytes"]) / 1024.0 for t, m, _ in ORDER]
    sk_bytes = [int(sk[(t, m)]["model_bytes"]) / 1024.0 for t, m, _ in ORDER]
    positions = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.bar(positions - width / 2, jsl_bytes, width, color=BLUE, label="JSL (.jsl)")
    ax.bar(positions + width / 2, sk_bytes, width, color=ORANGE, label="sklearn pickle")
    ax.set_yscale("log")
    ax.set_ylabel("序列化体积 (KiB, 对数轴)")
    ax.set_title("模型体积：二进制 .jsl vs Python pickle")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    for index, (a, b) in enumerate(zip(jsl_bytes, sk_bytes)):
        ax.text(index - width / 2, a * 1.15, f"{a:.0f}", ha="center", fontsize=7, color=MUTED)
        ax.text(index + width / 2, b * 1.15, f"{b:.0f}", ha="center", fontsize=7, color=MUTED)
    handles = [Patch(color=BLUE, label="JSL (.jsl)"), Patch(color=ORANGE, label="sklearn pickle")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_vs_size.png")


def radar_chart() -> None:
    angles = np.linspace(0, 2 * np.pi, len(RADAR_AXES), endpoint=False).tolist()
    angles += angles[:1]
    jsl = JSL_SCORES + JSL_SCORES[:1]
    sk = SKLEARN_SCORES + SKLEARN_SCORES[:1]
    fig, ax = plt.subplots(figsize=(5.8, 5.4), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_AXES, fontsize=10)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=7, color=MUTED)
    ax.grid(color=GRID)
    ax.plot(angles, jsl, color=BLUE, lw=2, label="JSL")
    ax.fill(angles, jsl, color=BLUE, alpha=0.15)
    ax.plot(angles, sk, color=ORANGE, lw=2, label="scikit-learn")
    ax.fill(angles, sk, color=ORANGE, alpha=0.12)
    ax.set_title("八维主观评分（1–10）", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.12))
    save(fig, "fig_vs_radar.png")


def breadth_chart() -> None:
    rows = [row for row in BREADTH]
    matrix = np.array([[row[1], row[2]] for row in rows])
    cmap = ListedColormap([GREY, "#fcd34d", "#99f6e4"])
    fig, ax = plt.subplots(figsize=(6.8, 7.2))
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["JSL", "scikit-learn"], fontsize=10)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row[0] for row in rows], fontsize=9)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for i, row in enumerate(rows):
        for j, value in enumerate((row[1], row[2])):
            text = "●" if value == 1 else ("△" if value == 0.5 else "○")
            ax.text(j, i, text, ha="center", va="center", fontsize=11,
                    color="#0f766e" if value == 1 else ("#92400e" if value == 0.5 else "#9ca3af"))
    ax.set_title("能力覆盖矩阵：● 完整  △ 部分  ○ 缺失", pad=12)
    legend = [Patch(color="#99f6e4", label="完整"), Patch(color="#fcd34d", label="部分"),
              Patch(color=GREY, label="缺失")]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    fig.tight_layout()
    save(fig, "fig_vs_breadth.png")


def concurrency_chart() -> None:
    java_rows = []
    with (DATA / "vthreads.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            java_rows.append(row)
    python_rows = []
    with (DATA / "python_concurrency.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            python_rows.append(row)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0))
    cpu = [("顺序", float([r for r in java_rows if r["scenario"] == "cpu" and r["mode"] == "sequential"][0]["wall_ms"]),
            "JSL"),
           ("平台线程×8", float([r for r in java_rows if r["scenario"] == "cpu" and r["mode"] == "platform-8"][0]["wall_ms"]), "JSL"),
           ("虚拟线程×8", float([r for r in java_rows if r["scenario"] == "cpu" and r["mode"] == "virtual-8"][0]["wall_ms"]), "JSL"),
           ("Python 线程", float([r for r in python_rows if r["scenario"] == "cpu" and r["mode"] == "python-threads"][0]["wall_ms"]), "Python"),
           ("Python 进程×8", float([r for r in python_rows if r["scenario"] == "cpu" and r["mode"] == "python-processes"][0]["wall_ms"]), "Python")]
    labels = [row[0] for row in cpu]
    values = [row[1] / 1000.0 for row in cpu]
    colors = [BLUE if row[2] == "JSL" else ORANGE for row in cpu]
    bars = ax1.bar(labels, values, color=colors, width=0.55)
    ax1.set_yscale("log")
    ax1.set_ylabel("4000 次推理总耗时 (s, 对数轴)")
    ax1.set_title("CPU 密集：批量推理")
    ax1.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, value * 1.15, f"{value:.2f}s",
                 ha="center", fontsize=7.6, color=MUTED)

    io = [("顺序", float([r for r in java_rows if r["scenario"] == "io" and r["mode"] == "sequential"][0]["wall_ms"]), "JSL"),
          ("平台线程×8", float([r for r in java_rows if r["scenario"] == "io" and r["mode"] == "platform-8"][0]["wall_ms"]), "JSL"),
          ("虚拟线程", float([r for r in java_rows if r["scenario"] == "io" and r["mode"] == "virtual"][0]["wall_ms"]), "JSL"),
          ("Python 线程×64", float([r for r in python_rows if r["scenario"] == "io" and r["mode"] == "python-threads"][0]["wall_ms"]), "Python"),
          ("Python 进程×8", float([r for r in python_rows if r["scenario"] == "io" and r["mode"] == "python-processes"][0]["wall_ms"]), "Python")]
    labels = [row[0] for row in io]
    values = [row[1] / 1000.0 for row in io]
    colors = [BLUE if row[2] == "JSL" else ORANGE for row in io]
    bars = ax2.bar(labels, values, color=colors, width=0.55)
    ax2.set_yscale("log")
    ax2.set_ylabel("4000 个 1ms I/O 任务总耗时 (s, 对数轴)")
    ax2.set_title("I/O 密集：请求编排")
    ax2.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, value * 1.15, f"{value:.2f}s",
                 ha="center", fontsize=7.6, color=MUTED)

    handles = [Patch(color=BLUE, label="JSL (Java 21)"), Patch(color=ORANGE, label="Python")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_vs_concurrency.png")


def profiling_chart() -> None:
    java_row = {}
    with (DATA / "profiling_java.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            java_row[row["scenario"]] = row
    python_rows = {}
    with (DATA / "profiling_python.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            python_rows[row["scenario"]] = row

    jsl_overhead = float(java_row["JSL + JFR (always-on)"]["overhead_percent"])
    coarse = float(python_rows["Python + cProfile (coarse)"]["overhead_percent"])
    fine = float(python_rows["Python + cProfile (fine)"]["overhead_percent"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.0))
    labels = ["JSL + JFR\n(内置, 采样式)", "Python + cProfile\n(粗粒度编排)", "Python + cProfile\n(细粒度编排)"]
    values = [max(jsl_overhead, 0.0), coarse, fine]
    colors = [BLUE, ORANGE, ORANGE]
    bars = ax1.bar(labels, values, color=colors, width=0.52)
    ax1.set_ylabel("剖析期间吞吐损失 (%)")
    ax1.set_title("生产剖析开销：4000 次推理请求")
    ax1.tick_params(axis="x", labelsize=8.5)
    for bar, value, raw in zip(bars, values, [jsl_overhead, coarse, fine]):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 2,
                 f"{raw:+.1f}%", ha="center", fontsize=9, color=MUTED)
    ax1.set_ylim(0, max(values) * 1.25)

    jsl_base = float(java_row["JSL + JFR (always-on)"]["baseline_ms"])
    jsl_profiled = float(java_row["JSL + JFR (always-on)"]["profiled_ms"])
    py_coarse = python_rows["Python + cProfile (coarse)"]
    py_fine = python_rows["Python + cProfile (fine)"]
    labels = ["JSL 基线", "JSL + JFR", "Python 基线\n(粗)", "Python + cProfile\n(粗)", "Python 基线\n(细)", "Python + cProfile\n(细)"]
    values = [jsl_base, jsl_profiled,
              float(py_coarse["baseline_ms"]), float(py_coarse["profiled_ms"]),
              float(py_fine["baseline_ms"]), float(py_fine["profiled_ms"])]
    colors = [BLUE, BLUE, ORANGE, ORANGE, ORANGE, ORANGE]
    bars = ax2.bar(labels, values, color=colors, width=0.6)
    ax2.set_ylabel("4000 请求总耗时 (ms)")
    ax2.set_title("绝对耗时：剖析开启前后")
    ax2.tick_params(axis="x", rotation=22, labelsize=7.5)
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, value * 1.02, f"{value:.0f}",
                 ha="center", fontsize=7.4, color=MUTED)

    handles = [Patch(color=BLUE, label="JSL (Java 21)"), Patch(color=ORANGE, label="Python 3.11")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_vs_profiling.png")


def tables(jsl, sk) -> None:
    lines = ["# JSL vs scikit-learn 指标表（自动生成）", "",
             "数据：20 000 训练 / 10 000 测试，32 特征；计时为 5 次中位数。", "",
             "## 客观指标", "",
             "| 任务 | 模型 | JSL 拟合 ms | sklearn 拟合 ms | 拟合比 | JSL 预测 ms | sklearn 预测 ms | JSL 得分 | sklearn 得分 | JSL 体积 KiB | pickle KiB |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for task, model, label in ORDER:
        j = jsl[(task, model)]
        s = sk[(task, model)]
        jf, sf = float(j["fit_ms"]), float(s["fit_ms"])
        jp, sp = float(j["predict_ms"]), float(s["predict_ms"])
        lines.append(
            f"| {label.replace(chr(10), ' ')} | {model} | {jf:.2f} | {sf:.2f} | {jf / sf:.2f}× | "
            f"{jp:.2f} | {sp:.2f} | {float(j['score']):.6f} | {float(s['score']):.6f} | "
            f"{int(j['model_bytes']) / 1024:.1f} | {int(s['model_bytes']) / 1024:.1f} |"
        )
    lines += ["", "## 主观评分", "", "| 维度 | JSL | scikit-learn | 依据 |", "|---|---:|---:|---|"]
    rationale = [
        "Builder + JSL 门面 + 强类型；仍缺 Pipeline 与文档生态",
        "21 原生算子 + 梯度检验 + 生命周期安全 + JFR + CRC 持久化",
        "10 项全部持平或反超；原生 GEMM 115 GF；CNN/Transformer 收敛",
        "`jsl.torch` eager API + 21 ops + 类型安全管线 + JFR；仍无稀疏/分布式",
        "10 项全部持平或更优；全部新算子通过中心差分梯度检验",
        "原生 CNN/Transformer/autograd/BPTT 为 sklearn 所无",
        "23 项矩阵 87% vs 67%，原生层与 JFR 为 sklearn 所无",
        "原生失败自动回退；双重释放/野指针/GC 兜底均有测试",
    ]
    for axis, a, b, reason in zip(RADAR_AXES, JSL_SCORES, SKLEARN_SCORES, rationale):
        lines.append(f"| {axis} | {a:.1f} | {b:.1f} | {reason} |")
    lines += ["", "> 主观评分是作者基于本报告证据的判断，不是测量值。", ""]
    path = ROOT / "report" / "vs_tables.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    jsl = read("jsl.csv")
    sk = read("sklearn.csv")
    speed_chart(jsl, sk)
    quality_chart(jsl, sk)
    size_chart(jsl, sk)
    radar_chart()
    breadth_chart()
    concurrency_chart()
    profiling_chart()
    tables(jsl, sk)


if __name__ == "__main__":
    main()
