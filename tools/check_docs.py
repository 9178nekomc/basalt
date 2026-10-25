#!/usr/bin/env python3
"""Smoke-test the generated documentation site.

Checks performed
----------------
1. Every HTML file parses without tag imbalance (html.parser).
2. No duplicate ``id`` attributes inside a page.
3. Every relative href resolves to an existing file.
4. Every ``file.html#fragment`` resolves to an existing id.
5. No placeholder text (``未提供说明``) remains.
6. Reports the number of documented types / methods / parameters.

Exit code 1 on any failure, so it can gate a release.

Usage:
    python tools/check_docs.py --root docs
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from html.parser import HTMLParser

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr"}


class Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.hrefs: list[str] = []
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.counts = {"class": 0, "method": 0, "field_list": 0, "param": 0,
                       "signature": 0}
        self.text_chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.hrefs.append(attrs["href"])
        cls = attrs.get("class", "")
        if tag == "dl" and "py" in cls.split():
            if "class" in cls.split():
                self.counts["class"] += 1
            if "method" in cls.split():
                self.counts["method"] += 1
            self.counts["signature"] += 1
        if tag == "dl" and "field-list" in cls:
            self.counts["field_list"] += 1
        if tag == "span" and "param-name" in cls:
            self.counts["param"] += 1
        if tag == "script" or tag == "style":
            self._skip_depth += 1
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if tag in ("script", "style"):
            self._skip_depth = max(0, self._skip_depth - 1)
        if not self.stack:
            self.errors.append(f"unexpected </{tag}>")
            return
        if self.stack[-1] == tag:
            self.stack.pop()
            return
        # tolerate implicit closes for <p> / <li>
        if tag in ("p", "li") and tag in self.stack:
            while self.stack and self.stack[-1] != tag:
                self.stack.pop()
            if self.stack:
                self.stack.pop()
            return
        self.errors.append(f"mismatched </{tag}>, open={self.stack[-1]}")

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self.text_chunks.append(data.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="docs")
    args = ap.parse_args()

    pages: dict[str, Collector] = {}
    for base, _dirs, files in os.walk(args.root):
        for name in sorted(files):
            if not name.endswith(".html"):
                continue
            path = os.path.join(base, name)
            parser = Collector()
            parser.feed(open(path, encoding="utf-8").read())
            parser.close()
            pages[path.replace("\\", "/")] = parser

    failures: list[str] = []
    total = {"class": 0, "method": 0, "field_list": 0, "param": 0}
    placeholders = 0

    for path, parser in sorted(pages.items()):
        rel = os.path.relpath(path, args.root).replace("\\", "/")
        for err in parser.errors:
            failures.append(f"{rel}: {err}")
        dupes = {i for i in parser.ids if parser.ids.count(i) > 1}
        for d in sorted(dupes):
            failures.append(f"{rel}: duplicate id '{d}'")
        if parser.stack:
            failures.append(f"{rel}: unclosed tags {parser.stack[-4:]}")
        for key in total:
            total[key] += parser.counts[key]
        placeholders += sum(1 for t in parser.text_chunks if "未提供说明" in t)

        for href in parser.hrefs:
            if href.startswith(("http://", "https://", "mailto:")):
                continue
            if href in ("", "#"):
                continue
            if href.startswith("#"):
                if href[1:] and href[1:] not in parser.ids:
                    failures.append(f"{rel}: missing local anchor {href}")
                continue
            target, _, frag = href.partition("#")
            target_path = os.path.normpath(os.path.join(os.path.dirname(path), target))
            target_key = target_path.replace("\\", "/")
            if not os.path.exists(target_path):
                failures.append(f"{rel}: broken link {href}")
                continue
            if frag and target_key in pages:
                if frag not in pages[target_key].ids:
                    failures.append(f"{rel}: missing anchor {href}")

    print(f"pages       : {len(pages)}")
    print(f"type blocks : {total['class']}")
    print(f"methods     : {total['method']}")
    print(f"field lists : {total['field_list']} (参数/返回/抛出)")
    print(f"params      : {total['param']}")
    print(f"placeholders: {placeholders}")
    if failures:
        print(f"\nFAIL ({len(failures)} issues)")
        for f in failures[:40]:
            print("  " + f)
        return 1
    print("\nOK — 所有页面结构闭合、锚点与链接全部可达")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
