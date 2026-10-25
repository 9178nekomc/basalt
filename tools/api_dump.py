#!/usr/bin/env python3
"""Extract the public API surface of the Basalt Java sources into a JSON model.

The extractor is dependency-free (stdlib only) and deliberately not a full Java
parser.  It masks comments and string literals (preserving byte offsets), then
splits class bodies at depth-1 member boundaries.  That is enough to recover exact
signatures, parameter names, declared exceptions, nesting relationships and the raw
Javadoc block attached to every declaration.

Usage:
    python tools/api_dump.py --src src/main/java --out build/api/api.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict

MODIFIERS = {
    "public", "protected", "private", "static", "final", "abstract", "native",
    "synchronized", "transient", "volatile", "strictfp", "default", "sealed",
    "non-sealed",
}


# --------------------------------------------------------------------------- #
# source masking
# --------------------------------------------------------------------------- #
def mask_source(src: str) -> str:
    """Blank out comments and string/char literals, preserving offsets and newlines."""
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j == -1 else j
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            j = n if j == -1 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        elif src.startswith('"""', i):
            j = src.find('"""', i + 3)
            j = n if j == -1 else j + 3
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        elif c in "\"'":
            quote = c
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == quote:
                    j += 1
                    break
                j += 1
            for k in range(i, min(j, n)):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        else:
            i += 1
    return "".join(out)


def match_brace(masked: str, open_index: int) -> int:
    depth = 0
    for i in range(open_index, len(masked)):
        ch = masked[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def split_top_level(text: str, separator: str = ",") -> list[str]:
    parts, buf = [], []
    angle = paren = bracket = 0
    for ch in text:
        if ch == "<":
            angle += 1
        elif ch == ">":
            angle -= 1
        elif ch == "(":
            paren += 1
        elif ch == ")":
            paren -= 1
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket -= 1
        if ch == separator and angle <= 0 and paren == 0 and bracket == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


# --------------------------------------------------------------------------- #
# javadoc
# --------------------------------------------------------------------------- #
TAG_RE = re.compile(r"^@(\w+)\s*(.*)$")


def parse_javadoc(raw: str) -> dict:
    body = re.sub(r"^/\*\*", "", raw)
    body = re.sub(r"\*/$", "", body)
    lines = [re.sub(r"^\s*\*+\s?", "", line).rstrip() for line in body.splitlines()]

    text_lines: list[str] = []
    params: dict[str, str] = {}
    throws: dict[str, str] = {}
    see: list[str] = []
    returns = ""
    since = ""
    deprecated = ""
    tag = None
    key = None

    for line in lines:
        stripped = line.strip()
        m = TAG_RE.match(stripped) if stripped.startswith("@") else None
        if m:
            tag = m.group(1)
            rest = m.group(2).strip()
            key = None
            if tag == "param":
                pieces = rest.split(None, 1)
                key = pieces[0] if pieces else ""
                params[key] = pieces[1].strip() if len(pieces) > 1 else ""
            elif tag == "return":
                returns = rest
            elif tag in ("throws", "exception"):
                pieces = rest.split(None, 1)
                key = pieces[0] if pieces else ""
                throws[key] = pieces[1].strip() if len(pieces) > 1 else ""
            elif tag == "see":
                see.append(rest)
            elif tag == "since":
                since = rest
            elif tag == "deprecated":
                deprecated = rest
            continue
        if not stripped:
            text_lines.append("")
            continue
        if tag == "param" and key is not None:
            params[key] = (params[key] + " " + stripped).strip()
        elif tag == "return" and returns:
            returns = (returns + " " + stripped).strip()
        elif tag in ("throws", "exception") and key is not None:
            throws[key] = (throws[key] + " " + stripped).strip()
        elif tag is None:
            text_lines.append(stripped)

    while text_lines and not text_lines[0].strip():
        text_lines.pop(0)
    while text_lines and not text_lines[-1].strip():
        text_lines.pop()

    return {
        "text": "\n".join(text_lines).strip(),
        "params": params,
        "returns": returns,
        "throws": throws,
        "see": see,
        "since": since,
        "deprecated": deprecated,
    }


def find_javadoc(src: str, decl_start: int) -> dict | None:
    head = src[:decl_start]
    idx = head.rfind("/**")
    if idx == -1:
        return None
    end = src.find("*/", idx)
    if end == -1 or end + 2 > decl_start:
        return None
    between = src[end + 2:decl_start]
    if re.sub(r"@[\w.]+(\s*\((?:[^()]|\([^()]*\))*\))?", "", between).strip():
        return None
    return parse_javadoc(src[idx:end + 2])


# --------------------------------------------------------------------------- #
# member parsing
# --------------------------------------------------------------------------- #
ANNOTATION_RE = re.compile(r"@[\w.]+(?:\s*\((?:[^()]|\([^()]*\))*\))?")


def strip_annotations(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = ANNOTATION_RE.sub(" ", text)
    return text


def parse_params(param_text: str) -> list[dict]:
    params = []
    for raw in split_top_level(param_text, ","):
        cleaned = re.sub(r"^final\s+", "", strip_annotations(raw).strip()).strip()
        if not cleaned:
            continue
        tokens = cleaned.split()
        if len(tokens) == 1:
            params.append({"type": cleaned, "name": cleaned})
            continue
        name = tokens[-1]
        ptype = " ".join(tokens[:-1])
        while name.endswith("[]"):
            name = name[:-2]
            ptype += "[]"
        params.append({"type": ptype.strip(), "name": name})
    return params


@dataclass
class Member:
    kind: str
    name: str
    signature: str = ""
    key: str = ""
    return_type: str = ""
    type: str = ""
    params: list = field(default_factory=list)
    throws: list = field(default_factory=list)
    modifiers: list = field(default_factory=list)
    static: bool = False
    abstract: bool = False
    javadoc: dict | None = None
    value: str = ""


def build_signature(name: str, params: list[dict], ret: str, kind: str) -> str:
    rendered = [f"{p['type']} {p['name']}".strip() for p in params]
    head = f"{ret} {name}" if kind == "method" else name
    return f"{head}({', '.join(rendered)})"


def build_key(name: str, params: list[dict]) -> str:
    return f"{name}({', '.join(p['type'] for p in params)})"


def parse_member(src: str, masked: str, start: int, end: int) -> Member | None:
    header_masked = masked[start:end]
    header_src = src[start:end]

    brace, depth = -1, 0
    for i, ch in enumerate(header_masked):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "{" and depth == 0:
            brace = i
            break
    header_masked = header_masked[:brace] if brace != -1 else header_masked.rstrip().rstrip(";")
    header_src = header_src[:brace] if brace != -1 else header_src.rstrip().rstrip(";")

    javadoc = find_javadoc(src, start)
    body = re.sub(r"\s+", " ", strip_annotations(header_masked)).strip()
    body_src = re.sub(r"\s+", " ", strip_annotations(header_src)).strip()
    if not body:
        return None

    generic = ""
    if body.startswith("<"):
        depth = 0
        for i, ch in enumerate(body):
            if ch == "<":
                depth += 1
            elif ch == ">":
                depth -= 1
                if depth == 0:
                    generic = body[:i + 1]
                    body = body[i + 1:].strip()
                    body_src = body_src[len(generic):].strip()
                    break

    tokens = body.split()
    modifiers = []
    while tokens and tokens[0] in MODIFIERS:
        modifiers.append(tokens.pop(0))
    remainder = " ".join(tokens)

    paren = remainder.find("(")
    if paren != -1:
        prefix = remainder[:paren].strip()
        type_and_name = prefix.split()
        if not type_and_name:
            return None
        name = type_and_name[-1]
        ret = " ".join(type_and_name[:-1]).strip()
        depth, close = 0, -1
        for i in range(paren, len(remainder)):
            if remainder[i] == "(":
                depth += 1
            elif remainder[i] == ")":
                depth -= 1
                if depth == 0:
                    close = i
                    break
        if close == -1:
            return None
        param_text = remainder[paren + 1:close]
        tail = remainder[close + 1:].strip()
        throws = []
        if tail.startswith("throws"):
            throws = [t.strip() for t in split_top_level(tail[len("throws"):].strip(), ",")]
        params = parse_params(param_text)
        kind = "constructor" if not ret else "method"
        return Member(
            kind=kind, name=name,
            signature=build_signature(name, params, ret, kind),
            key=build_key(name, params),
            return_type=ret, params=params, throws=throws, modifiers=modifiers,
            static="static" in modifiers, abstract="abstract" in modifiers,
            javadoc=javadoc,
        )

    # `int MAGIC = 0x42534C54` -> the declaration is everything before '=',
    # otherwise the initialiser would be taken as the field name.
    declaration = remainder.split("=", 1)[0].strip()
    field_tokens = declaration.split()
    if len(field_tokens) < 2:
        return None
    type_text = " ".join(field_tokens[:-1]).strip()
    name = field_tokens[-1]
    if not name or not type_text:
        return None
    value = remainder_src_value(header_src, name)
    signature = f"{type_text} {name}"
    if value:
        signature += f" = {value}"
    return Member(
        kind="field", name=name, type=type_text, key=name,
        signature=signature, modifiers=modifiers,
        static="static" in modifiers, javadoc=javadoc, value=value,
    )


def remainder_src_value(header_src: str, name: str) -> str:
    if "=" not in header_src:
        return ""
    return header_src.split("=", 1)[1].strip()


# --------------------------------------------------------------------------- #
# file / class extraction
# --------------------------------------------------------------------------- #
CLASS_RE = re.compile(
    r"(?P<mods>(?:(?:public|protected|private|static|final|abstract|sealed|non-sealed|strictfp)\s+)*)"
    r"(?P<kind>class|interface|enum|record)\s+"
    r"(?P<name>[A-Za-z_$][\w$]*)"
)


@dataclass
class JClass:
    name: str
    qualified: str
    simple_qualified: str
    module: str
    kind: str
    modifiers: list
    generic: str
    extends: str
    implements: list
    javadoc: dict | None
    members: list
    nested: list
    owner: str
    file: str
    line: int
    depth: int


def depth_map(masked: str) -> list[int]:
    depths = [0] * (len(masked) + 1)
    d = 0
    for i, ch in enumerate(masked):
        depths[i] = d
        if ch == "{":
            d += 1
        elif ch == "}":
            d -= 1
    depths[len(masked)] = d
    return depths


def scan_classes(src: str, masked: str, path: str, root: str, package: str) -> list[dict]:
    found = []
    for m in CLASS_RE.finditer(masked):
        prefix = masked[max(0, m.start() - 40):m.start()]
        if re.search(r"\bnew\s+$", prefix):
            continue
        brace = masked.find("{", m.end())
        if brace == -1:
            continue
        tail = masked[m.end():brace]
        if ";" in tail:
            continue
        body_end = match_brace(masked, brace)
        if body_end == -1:
            continue
        found.append({
            "start": m.start(), "name": m.group("name"), "kind": m.group("kind"),
            "mods": [x for x in m.group("mods").split() if x],
            "tail": tail.strip(), "open": brace, "end": body_end,
        })
    return found


def parse_file(path: str, root: str) -> list[JClass]:
    src = open(path, encoding="utf-8").read()
    masked = mask_source(src)
    package_match = re.search(r"^\s*package\s+([\w.]+)\s*;", masked, re.M)
    package = package_match.group(1) if package_match else ""
    depths = depth_map(masked)
    raw = scan_classes(src, masked, path, root, package)
    if not raw:
        return []

    # parent = innermost enclosing class body
    for item in raw:
        item["parent"] = None
        for other in raw:
            if other is item:
                continue
            if other["open"] < item["start"] < other["end"]:
                if item["parent"] is None or other["open"] > item["parent"]["open"]:
                    item["parent"] = other

    classes = []
    by_start = {item["start"]: item for item in raw}
    for item in raw:
        parent = item["parent"]
        owner = parent["name"] if parent else ""
        name = item["name"]
        if parent:
            qualified = f"{package}.{parent['name']}.{name}"
            simple = f"{parent['name']}.{name}"
        else:
            qualified = f"{package}.{name}" if package else name
            simple = name

        # header tail: generics / extends / implements / record components
        tail = item["tail"]
        generic = ""
        if tail.startswith("<"):
            depth = 0
            for i, ch in enumerate(tail):
                if ch == "<":
                    depth += 1
                elif ch == ">":
                    depth -= 1
                    if depth == 0:
                        generic = tail[:i + 1]
                        tail = tail[i + 1:].strip()
                        break
        extends, implements = "", []
        if "extends" in tail:
            after = tail.split("extends", 1)[1]
            if "implements" in after:
                ext, impl = after.split("implements", 1)
                extends = ext.strip().rstrip(",").strip()
                implements = [x.strip() for x in split_top_level(impl, ",")]
            else:
                extends = after.strip().rstrip(",").strip()
        elif "implements" in tail:
            implements = [x.strip() for x in split_top_level(tail.split("implements", 1)[1], ",")]

        children = [c for c in raw if c["parent"] is item]
        skip = [(c["start"], c["end"] + 1) for c in children]
        members = parse_body(src, masked, item["open"], item["end"], item["kind"], skip)

        # record components become documented fields
        if item["kind"] == "record":
            comp = re.search(r"\(([^)]*)\)", tail)
            if comp:
                for p in parse_params(comp.group(1)):
                    members.insert(0, Member(
                        kind="field", name=p["name"], type=p["type"], key=p["name"],
                        signature=f"{p['type']} {p['name']}", modifiers=["private", "final"],
                        javadoc=None,
                    ))

        line = src.count("\n", 0, item["start"]) + 1
        classes.append(JClass(
            name=name, qualified=qualified, simple_qualified=simple,
            module=package or "(default)", kind=item["kind"],
            modifiers=item["mods"], generic=generic, extends=extends,
            implements=implements, javadoc=find_javadoc(src, item["start"]),
            members=members, nested=[c["name"] for c in children], owner=owner,
            file=os.path.relpath(path, root).replace("\\", "/"), line=line,
            depth=depths[item["start"]],
        ))
    return classes


ENUM_CONST_RE = re.compile(r"^([A-Za-z_$][\w$]*)\s*(\((.*)\))?$", re.S)


def append_enum_constants(members: list, const_text: str) -> None:
    """Parse an enum constant list terminated by ';' or by the closing brace."""
    if not const_text.strip():
        return
    for piece in split_top_level(const_text, ","):
        piece = strip_annotations(piece).strip()
        if not piece:
            continue
        mm = ENUM_CONST_RE.match(piece)
        if mm:
            members.append(Member(kind="enum_constant", name=mm.group(1),
                                  key=mm.group(1), signature=piece, javadoc=None))


def parse_body(src: str, masked: str, brace: int, body_end: int, kind: str,
               skip: list[tuple[int, int]] | None = None) -> list[Member]:
    skip = sorted(skip or [])
    members: list[Member] = []
    i = brace + 1
    seg_start = i
    depth = 0
    enum_done = False

    def flush(end_index: int, with_semicolon: bool) -> None:
        nonlocal seg_start
        if end_index <= seg_start:
            return
        if not masked[seg_start:end_index].strip():
            return
        member = parse_member(src, masked, seg_start, end_index + (1 if with_semicolon else 0))
        if member is not None:
            members.append(member)

    while i < body_end:
        for s_start, s_end in skip:
            if i == s_start:
                i = s_end
                seg_start = i
                break
        else:
            ch = masked[i]
            if ch == "{":
                depth += 1
                i += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    flush(i, False)
                    i += 1
                    seg_start = i
                    continue
                i += 1
                continue
            if depth == 0 and ch == ";":
                if kind == "enum" and not enum_done:
                    append_enum_constants(members, masked[seg_start:i])
                    enum_done = True
                    i += 1
                    seg_start = i
                    continue
                flush(i, True)
                i += 1
                seg_start = i
                continue
            i += 1
            continue
        continue
    if kind == "enum" and not enum_done:
        # No ';' inside the body: the whole body is the constant list, and the
        # closing brace sits at body_end so the loop above never reaches it.
        append_enum_constants(members, masked[brace + 1:body_end])
    return members


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="src/main/java")
    ap.add_argument("--out", default="build/api/api.json")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print(f"source directory not found: {args.src}", file=sys.stderr)
        return 2

    files = []
    for base, _dirs, names in os.walk(args.src):
        for n in sorted(names):
            if n.endswith(".java"):
                files.append(os.path.join(base, n))
    files.sort()

    classes: list[JClass] = []
    for path in files:
        classes.extend(parse_file(path, args.src))

    modules: dict[str, list] = {}
    for cls in classes:
        modules.setdefault(cls.module, []).append(cls)

    def sort_key(c: JClass):
        return (c.owner, c.name)

    model = {
        "generated_by": "tools/api_dump.py",
        "source_root": args.src,
        "file_count": len(files),
        "type_count": len(classes),
        "modules": [
            {"name": name, "types": [asdict(c) for c in sorted(mods, key=sort_key)]}
            for name, mods in sorted(modules.items())
        ],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(model, fh, ensure_ascii=False, indent=1)

    methods = sum(1 for c in classes for m in c.members if m.kind in ("method", "constructor"))
    print(f"files={len(files)} types={len(classes)} methods={methods}")
    for name, mods in sorted(modules.items()):
        pub = sum(1 for c in mods if "public" in c.modifiers)
        print(f"  {name:24s} {len(mods):3d} types ({pub} public)")
    print(f"types without javadoc: {sum(1 for c in classes if not c.javadoc)}")
    dupes = {}
    for c in classes:
        if c.qualified in dupes:
            print(f"DUPLICATE qualified name: {c.qualified}", file=sys.stderr)
        dupes[c.qualified] = c
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
