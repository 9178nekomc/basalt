#!/usr/bin/env python3
"""Minimal, dependency-free Markdown renderer for the project README.

Only the subset used by this repository is implemented, but each construct is
handled properly (no regex-only shortcuts that break on nesting):

    ATX headings, fenced code blocks, GFM tables, ordered/unordered lists with
    one nesting level, blockquotes, horizontal rules, paragraphs, and inline
    code / bold / italic / strikethrough / links.

HTML is escaped before any inline markup is applied, so a README containing
``<T>`` or ``&&`` cannot inject tags into the generated page.

Usage:
    from md_render import render_markdown
    html = render_markdown(open("README.md", encoding="utf-8").read())
"""

from __future__ import annotations

import html
import re

FENCE_RE = re.compile(r"^```([\w+-]*)\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
HR_RE = re.compile(r"^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$")
UL_RE = re.compile(r"^(\s*)([-*+])\s+(.*)$")
OL_RE = re.compile(r"^(\s*)(\d+)\.\s+(.*)$")
TABLE_SEP_RE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?$")
QUOTE_RE = re.compile(r"^\s{0,3}>\s?(.*)$")

CODE_SPAN_RE = re.compile(r"`([^`]+)`")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
BOLD_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
ITALIC_RE = re.compile(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])")
STRIKE_RE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
AUTOLINK_RE = re.compile(r"(?<![\"'=])(https?://[^\s<>()]+)")

JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class",
    "const", "continue", "default", "do", "double", "else", "enum", "extends", "final",
    "finally", "float", "for", "goto", "if", "implements", "import", "instanceof", "int",
    "interface", "long", "native", "new", "package", "private", "protected", "public",
    "return", "short", "static", "strictfp", "super", "switch", "synchronized", "this",
    "throw", "throws", "transient", "try", "void", "volatile", "while", "var", "true",
    "false", "null",
}


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def inline(text: str) -> str:
    """Escape then apply the inline markup subset."""
    out = esc(text)

    # code spans first: their content must not be touched by later rules
    spans: list[str] = []

    def stash(m: re.Match) -> str:
        spans.append(f"<code>{m.group(1)}</code>")
        return f"\x00{len(spans) - 1}\x00"

    out = CODE_SPAN_RE.sub(stash, out)
    out = LINK_RE.sub(
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', out)
    out = BOLD_RE.sub(lambda m: f"<strong>{m.group(1)}</strong>", out)
    out = ITALIC_RE.sub(lambda m: f"<em>{m.group(1)}</em>", out)
    out = STRIKE_RE.sub(lambda m: f"<del>{m.group(1)}</del>", out)
    out = AUTOLINK_RE.sub(
        lambda m: f'<a href="{m.group(1)}">{m.group(1)}</a>', out)

    def unstash(m: re.Match) -> str:
        return spans[int(m.group(1))]

    return re.sub(r"\x00(\d+)\x00", unstash, out)


CJK_TAIL = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef\u2018-\u201d]$")
CJK_HEAD = re.compile(r"^[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef\u2018-\u201d]")


def join_lines(parts: list[str]) -> str:
    """Join source lines the way Markdown does, without inserting spaces between CJK."""
    html_parts: list[str] = []
    prev = ""
    for raw in parts:
        hard_break = raw.endswith("  ")
        text = raw.rstrip()
        if not html_parts:
            html_parts.append(inline(text))
        elif hard_break:
            html_parts.append("<br />" + inline(text))
        elif CJK_TAIL.search(prev) and CJK_HEAD.match(text):
            html_parts.append(inline(text))
        else:
            html_parts.append(" " + inline(text))
        prev = text
    return "".join(html_parts)


def split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    cells, buf, in_code = [], [], False
    for ch in line:
        if ch == "`":
            in_code = not in_code
        if ch == "|" and not in_code:
            cells.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    cells.append("".join(buf).strip())
    return cells


JAVA_HIGHLIGHTER = None


def set_java_highlighter(fn) -> None:
    """Let the site generator reuse its Java highlighter for ```java fences."""
    global JAVA_HIGHLIGHTER
    JAVA_HIGHLIGHTER = fn


def render_code(lang: str, lines: list[str]) -> str:
    body = "\n".join(lines)
    cls = f' class="language-{esc(lang)}"' if lang else ""
    if lang in ("java", "kotlin") and JAVA_HIGHLIGHTER is not None:
        return f'<pre class="code"><code{cls}>{JAVA_HIGHLIGHTER(body)}</code></pre>'
    return f'<pre class="code"><code{cls}>{esc(body)}</code></pre>'


def render_list(items: list[tuple[int, str, list[str]]]) -> str:
    """Render an indented list block; deeper items nest inside the previous <li>."""
    out: list[str] = []
    stack: list[tuple[int, str]] = []   # (indent, kind)
    for indent, kind, parts in items:
        while stack and indent < stack[-1][0]:
            out.append(f"</li></{stack.pop()[1]}>")
        if not stack or indent > stack[-1][0]:
            out.append(f"<{kind}>")
            stack.append((indent, kind))
        else:
            out.append("</li>")
            if stack[-1][1] != kind:
                out.append(f"</{stack.pop()[1]}><{kind}>")
                stack.append((indent, kind))
        out.append(f"<li>{join_lines(parts)}")
    while stack:
        out.append(f"</li></{stack.pop()[1]}>")
    return "".join(out)


def render_markdown(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    i, n = 0, len(lines)
    list_stack: list[str] = []          # "ul" / "ol" currently open

    def close_lists() -> None:
        while list_stack:
            out.append(f"</{list_stack.pop()}>")

    while i < n:
        line = lines[i]

        fence = FENCE_RE.match(line.strip())
        if fence:
            close_lists()
            lang = fence.group(1)
            block: list[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1  # closing fence
            out.append(render_code(lang, block))
            continue

        if not line.strip():
            close_lists()
            i += 1
            continue

        heading = HEADING_RE.match(line)
        if heading:
            close_lists()
            level = len(heading.group(1))
            out.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            i += 1
            continue

        if HR_RE.match(line):
            close_lists()
            out.append("<hr />")
            i += 1
            continue

        if line.lstrip().startswith("|") and i + 1 < n and TABLE_SEP_RE.match(lines[i + 1].strip()):
            close_lists()
            header = split_row(line)
            aligns = []
            for cell in split_row(lines[i + 1]):
                left = cell.startswith(":")
                right = cell.endswith(":")
                aligns.append("center" if left and right else "right" if right
                              else "left")
            i += 2
            rows = []
            while i < n and lines[i].lstrip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            head_html = "".join(
                f'<th style="text-align:{aligns[k] if k < len(aligns) else "left"}">'
                f"{inline(cell)}</th>"
                for k, cell in enumerate(header)
            )
            body_html = []
            for row in rows:
                cells = "".join(
                    f'<td style="text-align:{aligns[k] if k < len(aligns) else "left"}">'
                    f"{inline(cell)}</td>"
                    for k, cell in enumerate(row)
                )
                body_html.append(f"<tr>{cells}</tr>")
            out.append(f"<table><thead><tr>{head_html}</tr></thead>"
                       f'<tbody>{"".join(body_html)}</tbody></table>')
            continue

        quote = QUOTE_RE.match(line)
        if quote:
            close_lists()
            block = []
            while i < n and QUOTE_RE.match(lines[i]):
                block.append(QUOTE_RE.match(lines[i]).group(1))
                i += 1
            out.append(f"<blockquote><p>{join_lines(block)}</p></blockquote>")
            continue

        ul = UL_RE.match(line)
        ol = OL_RE.match(line)
        if ul or ol:
            close_lists()
            # collect the whole list block, keeping each item's indent
            items: list[tuple[int, str, list[str]]] = []
            while i < n:
                m_ul = UL_RE.match(lines[i])
                m_ol = OL_RE.match(lines[i])
                m = m_ul or m_ol
                if m:
                    indent = len(m.group(1).replace("\t", "    "))
                    kind = "ul" if m_ul else "ol"
                    items.append((indent, kind, [m.group(3)]))
                    i += 1
                    continue
                if lines[i].strip() and items and (
                        lines[i].startswith("  ") or lines[i].startswith("\t")):
                    items[-1][2].append(lines[i].strip())
                    i += 1
                    continue
                break
            out.append(render_list(items))
            continue

        # paragraph: consume until a blank line or a block opener
        block = [line.strip()]
        i += 1
        while i < n and lines[i].strip() and not (
                HEADING_RE.match(lines[i]) or FENCE_RE.match(lines[i].strip())
                or UL_RE.match(lines[i]) or OL_RE.match(lines[i])
                or HR_RE.match(lines[i]) or QUOTE_RE.match(lines[i])
                or lines[i].lstrip().startswith("|")):
            block.append(lines[i].strip())
            i += 1
        close_lists()
        out.append(f"<p>{join_lines(block)}</p>")

    close_lists()
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    source = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else sys.stdin.read()
    print(render_markdown(source))
