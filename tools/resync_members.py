#!/usr/bin/env python3
"""Re-align every docs_data type with the current API dump.

Members are rebuilt in API order; existing ``doc`` objects are carried over by
``(type, signature)``.  Types that appeared since the last sync are appended to the
shard that already documents their enclosing class (falling back to the shard that
documents the same module).

Run after changing public signatures, then fill the newly reported gaps.

Usage:
    python tools/resync_members.py --api build/api/api.json --docs tools/docs_data
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def visible(member: dict) -> bool:
    mods = set(member.get("modifiers") or [])
    if "private" in mods:
        return False
    if member["kind"] == "constructor" and not (mods & {"public", "protected"}):
        return False
    if member["kind"] in ("method", "field") and not (mods & {"public", "protected"}):
        return False
    return True


def skeleton_member(m: dict) -> dict:
    entry = {
        "kind": m["kind"],
        "key": m["key"],
        "signature": m["signature"],
        "params": m["params"],
        "throws": m["throws"],
    }
    if m.get("javadoc"):
        entry["javadoc"] = m["javadoc"]
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="build/api/api.json")
    ap.add_argument("--docs", default="tools/docs_data")
    args = ap.parse_args()

    api = json.load(open(args.api, encoding="utf-8"))
    shards: dict[str, dict] = {}
    owner: dict[str, str] = {}          # qualified type -> shard file
    existing: dict[str, dict] = {}      # qualified type -> type object

    for path in sorted(glob.glob(os.path.join(args.docs, "g*.json"))):
        data = json.load(open(path, encoding="utf-8"))
        shards[path] = data
        for module in data["modules"]:
            for t in module["types"]:
                owner[t["qualified"]] = path
                existing[t["qualified"]] = t

    by_module: dict[str, str] = {}
    for path, data in shards.items():
        for module in data["modules"]:
            by_module.setdefault(module["name"], path)

    api_types = [(module["name"], t) for module in api["modules"] for t in module["types"]]

    added_types = 0
    added_members = 0
    for module_name, t in api_types:
        qualified = t["qualified"]
        if qualified in existing:
            target = existing[qualified]
            old = {m["key"]: m for m in target["members"]}
            members = []
            for m in t["members"]:
                if not visible(m):
                    continue
                if m["key"] in old:
                    members.append(old[m["key"]])
                else:
                    members.append(skeleton_member(m))
                    added_members += 1
            target["members"] = members
            continue

        # new type: attach it next to its enclosing class, else next to its module
        outer = qualified.rsplit(".", 1)[0]
        path = owner.get(outer) or by_module.get(module_name)
        if path is None:
            print(f"WARN no shard for new type {qualified}", file=sys.stderr)
            continue
        shard = shards[path]
        module_entry = next((m for m in shard["modules"] if m["name"] == module_name), None)
        if module_entry is None:
            module_entry = {"name": module_name, "types": []}
            shard["modules"].append(module_entry)
        new_type = {
            "qualified": qualified,
            "simple_qualified": t["simple_qualified"],
            "kind": t["kind"],
            "modifiers": t["modifiers"],
            "generic": t["generic"],
            "extends": t["extends"],
            "implements": t["implements"],
            "file": t["file"],
            "line": t["line"],
            "nested": t["nested"],
            "members": [skeleton_member(m) for m in t["members"] if visible(m)],
        }
        module_entry["types"].append(new_type)
        owner[qualified] = path
        existing[qualified] = new_type
        added_types += 1
        added_members += len(new_type["members"])
        print(f"NEW type {qualified} -> {os.path.basename(path)}")

    dropped = 0
    keep = {t["qualified"] for _, t in api_types}
    for path, data in shards.items():
        for module in data["modules"]:
            before = len(module["types"])
            module["types"] = [t for t in module["types"] if t["qualified"] in keep]
            dropped += before - len(module["types"])

    for path, data in shards.items():
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)

    missing_members = sum(1 for q in existing for m in existing[q]["members"] if not m.get("doc"))
    missing_types = sum(1 for q in existing if not existing[q].get("doc"))
    print(f"new types {added_types}, new members {added_members}, dropped types {dropped}")
    print(f"still undocumented: {missing_types} types, {missing_members} members")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
