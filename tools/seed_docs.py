#!/usr/bin/env python3
"""Seed 3.4 documentation from the 2.4.0 corpus.

The 2.4.0 docs_data files were keyed by ``(qualified type, member signature)``.
Those keys are stable across releases, so unchanged API can be carried over
verbatim; only genuinely new or changed members need writing.

Reads   build/api/briefs/*.json   (skeleton for the current release)
        --seed <dir>              (previous release's docs_data)
Writes  tools/docs_data/*.json    (skeleton + every doc that could be matched)

Usage:
    python tools/seed_docs.py --seed ../seed-2.4 --out tools/docs_data
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def load_corpus(pattern: str) -> tuple[dict, dict]:
    types: dict[str, dict] = {}
    members: dict[tuple[str, str], dict] = {}
    for path in sorted(glob.glob(pattern)):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"WARN skip {path}: {exc}", file=sys.stderr)
            continue
        for module in data.get("modules", []):
            for t in module.get("types", []):
                if t.get("doc"):
                    types[t["qualified"]] = t["doc"]
                for m in t.get("members", []):
                    if m.get("doc"):
                        members[(t["qualified"], m["key"])] = m["doc"]
    return types, members


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--briefs", default="build/api/briefs")
    ap.add_argument("--seed", required=True)
    ap.add_argument("--out", default="tools/docs_data")
    args = ap.parse_args()

    old_types, old_members = load_corpus(os.path.join(args.seed, "g*.json"))
    print(f"seed corpus: {len(old_types)} types, {len(old_members)} members")

    os.makedirs(args.out, exist_ok=True)
    for stale in os.listdir(args.out):
        if stale.endswith(".json"):
            os.unlink(os.path.join(args.out, stale))

    tot_t = tot_m = hit_t = hit_m = 0
    missing_types: list[str] = []
    missing_members: list[tuple[str, str]] = []
    per_group = []

    for path in sorted(glob.glob(os.path.join(args.briefs, "g*.json"))):
        brief = json.load(open(path, encoding="utf-8"))
        gt = gm = gth = ghm = 0
        for module in brief["modules"]:
            for t in module["types"]:
                gt += 1
                if t["qualified"] in old_types:
                    t["doc"] = old_types[t["qualified"]]
                    gth += 1
                else:
                    missing_types.append(t["qualified"])
                for m in t["members"]:
                    gm += 1
                    doc = old_members.get((t["qualified"], m["key"]))
                    if doc:
                        m["doc"] = doc
                        ghm += 1
                    else:
                        missing_members.append((t["qualified"], m["key"]))
        tot_t += gt
        tot_m += gm
        hit_t += gth
        hit_m += ghm
        per_group.append((os.path.basename(path)[:-5], brief["title"], gt, gth, gm, ghm))
        with open(os.path.join(args.out, os.path.basename(path)), "w", encoding="utf-8") as fh:
            json.dump(brief, fh, ensure_ascii=False, indent=1)

    print(f"\n{'group':6s} {'types':>5s} {'reused':>6s} {'members':>7s} {'reused':>6s}  title")
    for gid, title, gt, gth, gm, ghm in per_group:
        print(f"{gid:6s} {gt:5d} {gth:6d} {gm:7d} {ghm:6d}  {title}")

    print(f"\ntotal: types {hit_t}/{tot_t} ({hit_t / tot_t:.1%}), "
          f"members {hit_m}/{tot_m} ({hit_m / tot_m:.1%})")
    print(f"needs writing: {len(missing_types)} types, {len(missing_members)} members")

    with open("build/api/missing.json", "w", encoding="utf-8") as fh:
        json.dump({"types": missing_types,
                   "members": [{"type": t, "key": k} for t, k in missing_members]},
                  fh, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
