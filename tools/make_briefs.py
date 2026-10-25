#!/usr/bin/env python3
"""Split build/api/api.json into per-group documentation briefs.

Each brief is a skeleton that mirrors the extracted API exactly.  A writer fills in
the ``doc`` objects; the renderer merges skeleton + doc and never trusts the writer
for signatures.  Keeping the split mechanical is what makes "documented API" and
"compiled API" impossible to drift apart.

Usage:
    python tools/make_briefs.py --api build/api/api.json --out build/api/briefs
"""

from __future__ import annotations

import argparse
import json
import os

# group id -> (title, module -> list of type names, or "*" for the whole module)
GROUPS: list[tuple[str, str, dict[str, list[str] | str]]] = [
    ("g01", "门面与并发", {"jsl": "*", "jsl.concurrent": "*"}),
    ("g02a", "核心数据：布局与矩阵", {"jsl.core": [
        "Layout", "BC", "BCHW", "BHW", "BSF",
        "Matrix", "MatMul", "MatMul.RowTask", "Tensor", "VectorMath",
    ]}),
    ("g02b", "核心数据：原生核与模型 I/O", {"jsl.core": [
        "NativeKernels", "ModelKind", "ModelReader", "ModelWriter",
        "Persistable", "ModelFormatException",
    ]}),
    ("g03", "数据管线与持久化", {"jsl.data": "*", "jsl.io": "*"}),
    ("g04", "线性模型 · 近邻 · 指标", {"jsl.ml": [
        "Estimator", "LinearRegression", "LinearRegression.Builder",
        "LogisticRegression", "LogisticRegression.Builder",
        "Ridge", "Ridge.Builder", "Lasso", "Lasso.Builder",
        "KNearestNeighbors", "KNearestNeighbors.Builder", "KNearestNeighbors.Neighbor",
        "KNeighborsRegressor", "KNeighborsRegressor.Builder", "KNeighborsRegressor.Neighbor",
        "DistanceMetric", "LinearSVM", "LinearSVM.Builder",
        "Metrics", "DataSplit", "DataSplit.Fold", "DataSplit.Holdout",
        "MlChecks", "JSL", "JSL.Algorithm", "JSL.Metric", "JSL.Task",
    ]}),
    ("g05", "树模型与梯度提升", {"jsl.ml": [
        "DecisionTree", "DecisionTree.Builder", "DecisionTree.Workspace",
        "RandomForest", "RandomForest.Builder",
        "TreeModel", "TreeTask", "ImpurityCriterion", "PackedTrees", "BinnedFeatures",
        "HistogramGradientBoosting", "HistogramGradientBoosting.Builder",
        "HistogramGradientBoosting.HistogramWorkspace",
        "HistogramGradientBoosting.LeafCandidate",
        "HistogramGradientBoosting.NodeHistogram",
        "HistogramGradientBoosting.SplitDecision",
        "BoostingAlgorithm", "BoostingObjective",
    ]}),
    ("g06", "聚类 · 降维 · 朴素贝叶斯", {"jsl.ml": [
        "KMeans", "KMeans.Builder", "KMeans.Run",
        "DBSCAN", "DBSCAN.Builder",
        "PCA", "PCA.Builder", "PCA.EigenDecomposition", "EigenSolver",
        "GaussianNaiveBayes", "GaussianNaiveBayes.Builder",
    ]}),
    ("g07a", "神经网络：层与卷积", {"jsl.nn": [
        "Layer", "Activation", "Dense", "Dense.Builder",
        "Conv2D", "Conv2D.Builder", "MaxPooling2D", "MaxPooling2D.Builder",
        "Flatten", "Flatten.Builder", "Padding",
    ]}),
    ("g07b", "神经网络：损失、优化器与训练", {"jsl.nn": [
        "Loss", "LossEvaluation", "Optimizer", "OptimizerConfig", "TrainingHistory",
        "Parameter", "ParameterRef",
        "NeuralNetwork", "NeuralNetwork.ImageBuilder", "NeuralNetwork.VectorBuilder",
        "CNN", "ModelSerializer", "GradientCheck", "GradientCheck.Report",
    ]}),
    ("g08", "序列模型与注意力", {"jsl.nn": [
        "LSTM", "LSTM.Builder", "GRU", "GRU.Builder", "VanillaRNN", "VanillaRNN.Builder",
        "SelfAttention", "SelfAttention.Builder",
        "SequenceNetwork", "SequenceNetwork.Builder",
        "SequencePool", "SequencePool.Pooling",
    ]}),
    ("g09a", "原生桥接：上下文与句柄", {"jsl.nativebridge": [
        "NativeContext", "NativeContext.Guard", "NativeContext.State",
        "NativeTensorHandle", "NativeTensorHandle.State",
        "NativeLibrary", "NativeBindings",
    ]}),
    ("g09b", "原生桥接：指令队列与参考实现", {"jsl.nativebridge": [
        "CommandQueueBuilder", "NativeMLP", "NativeMLP.Activation",
        "NativeMLP.OutputActivation",
    ]}),
    ("g10a", "eager 张量：核心与自动求导", {"jsl.torch": [
        "Tensor", "Module", "Sequential", "Graph", "NativeQueueEvent",
        "Optimizer", "AdamW", "SGD", "Loss", "MSELoss", "CrossEntropyLoss",
    ]}),
    ("g10b", "eager 张量：模块与 Transformer", {"jsl.torch": [
        "Linear", "Conv2d", "MaxPool2d", "BatchNorm2d", "LayerNorm", "Dropout",
        "Embedding", "Flatten", "Activation", "Activation.Kind",
        "MultiheadAttention", "TransformerBlock", "SelfAttention",
    ]}),
]


def build_brief(group_id: str, title: str, selection: dict, api: dict) -> dict:
    modules = []
    for module in api["modules"]:
        name = module["name"]
        if name not in selection:
            continue
        wanted = selection[name]
        types = []
        for t in module["types"]:
            if wanted != "*" and t["simple_qualified"] not in wanted:
                continue
            members = []
            for m in t["members"]:
                if "private" in m["modifiers"]:
                    continue
                if m["kind"] == "constructor" and "public" not in m["modifiers"] \
                        and "protected" not in m["modifiers"]:
                    continue
                members.append({
                    "kind": m["kind"],
                    "key": m["key"],
                    "signature": m["signature"],
                    "params": m["params"],
                    "throws": m["throws"],
                    "javadoc": (m["javadoc"] or {}).get("text", ""),
                })
            types.append({
                "qualified": t["qualified"],
                "simple_qualified": t["simple_qualified"],
                "kind": t["kind"],
                "modifiers": t["modifiers"],
                "generic": t["generic"],
                "extends": t["extends"],
                "implements": t["implements"],
                "file": t["file"],
                "line": t["line"],
                "javadoc": (t["javadoc"] or {}).get("text", ""),
                "nested": t["nested"],
                "members": members,
            })
        if types:
            modules.append({"name": name, "types": types})
    return {"group": group_id, "title": title, "modules": modules}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="build/api/api.json")
    ap.add_argument("--out", default="build/api/briefs")
    args = ap.parse_args()

    api = json.load(open(args.api, encoding="utf-8"))
    os.makedirs(args.out, exist_ok=True)

    covered: set[str] = set()
    total_types = total_members = 0
    for group_id, title, selection in GROUPS:
        brief = build_brief(group_id, title, selection, api)
        path = os.path.join(args.out, f"{group_id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(brief, fh, ensure_ascii=False, indent=1)
        types = sum(len(m["types"]) for m in brief["modules"])
        members = sum(len(t["members"]) for m in brief["modules"] for t in m["types"])
        total_types += types
        total_members += members
        for m in brief["modules"]:
            for t in m["types"]:
                covered.add(t["qualified"])
        print(f"{group_id}  {title:22s} types={types:3d} members={members:4d}")

    all_types = {t["qualified"] for m in api["modules"] for t in m["types"]}
    missing = sorted(all_types - covered)
    print(f"\ntotal types={total_types} members={total_members}")
    if missing:
        print(f"NOT ASSIGNED ({len(missing)}): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
