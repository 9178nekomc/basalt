#!/usr/bin/env python3
"""Split build/api/api.json into per-group documentation briefs.

Each brief is a skeleton that mirrors the extracted API exactly.  A writer fills in
the ``doc`` objects; the renderer merges skeleton + doc and never trusts the writer
for signatures.  Keeping the split mechanical is what makes "documented API" and
"compiled API" impossible to drift apart.

Groups are produced automatically: every module is cut into slices of at most
``--size`` public types, so the split keeps working as the API grows (2.4.0 had
154 types, 3.4.0 has 248).

Usage:
    python tools/make_briefs.py --api build/api/api.json --out build/api/briefs
"""

from __future__ import annotations

import argparse
import json
import os

MODULE_ORDER = [
    "basalt", "basalt.core", "basalt.data", "basalt.io", "basalt.ml", "basalt.nn",
    "basalt.nativebridge", "basalt.torch", "basalt.concurrent",
]

MODULE_TITLE = {
    "basalt": "门面与异常",
    "basalt.core": "核心数据与模型 I/O",
    "basalt.data": "数据管线",
    "basalt.io": "持久化与 ONNX",
    "basalt.ml": "经典机器学习",
    "basalt.nn": "神经网络层与序列",
    "basalt.nativebridge": "原生桥接层",
    "basalt.torch": "eager 张量与模块",
    "basalt.concurrent": "并发推理",
}


def build_brief(group_id: str, title: str, module: dict, types: list[dict]) -> dict:
    kept = []
    for t in types:
        members = []
        for m in t["members"]:
            mods = set(m.get("modifiers") or [])
            if "private" in mods:
                continue
            if m["kind"] == "constructor" and not (mods & {"public", "protected"}):
                continue
            if m["kind"] in ("method", "field") and not (mods & {"public", "protected"}):
                continue
            members.append({
                "kind": m["kind"],
                "key": m["key"],
                "signature": m["signature"],
                "params": m["params"],
                "throws": m["throws"],
                "javadoc": (m["javadoc"] or {}).get("text", ""),
            })
        kept.append({
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
    return {"group": group_id, "title": title,
            "modules": [{"name": module["name"], "types": kept}]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="build/api/api.json")
    ap.add_argument("--out", default="build/api/briefs")
    ap.add_argument("--size", type=int, default=12,
                    help="max public types per group")
    args = ap.parse_args()

    api = json.load(open(args.api, encoding="utf-8"))
    os.makedirs(args.out, exist_ok=True)
    for stale in os.listdir(args.out):
        if stale.endswith(".json"):
            os.unlink(os.path.join(args.out, stale))

    modules = sorted(api["modules"],
                     key=lambda m: (MODULE_ORDER.index(m["name"])
                                    if m["name"] in MODULE_ORDER else 99, m["name"]))

    index = 0
    total_types = total_members = 0
    plan = []
    for module in modules:
        public = [t for t in module["types"] if "public" in t["modifiers"]]
        if not public:
            continue
        chunks = (len(public) + args.size - 1) // args.size
        for start in range(0, len(public), args.size):
            chunk = public[start:start + args.size]
            index += 1
            gid = f"g{index:02d}"
            title = MODULE_TITLE.get(module["name"], module["name"])
            if chunks > 1:
                title = f"{title}（{start // args.size + 1}/{chunks}）"
            brief = build_brief(gid, title, module, chunk)
            with open(os.path.join(args.out, f"{gid}.json"), "w", encoding="utf-8") as fh:
                json.dump(brief, fh, ensure_ascii=False, indent=1)
            members = sum(len(t["members"]) for t in chunk)
            total_types += len(chunk)
            total_members += members
            plan.append((gid, module["name"], len(chunk), members, title))

    print(f"groups={len(plan)} types={total_types} members={total_members}")
    for gid, mod, nt, nm, title in plan:
        print(f"  {gid}  {mod:20s} {nt:3d} types {nm:4d} members  {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
