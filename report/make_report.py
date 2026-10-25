#!/usr/bin/env python3
"""Render the JSL Markdown report into a self-contained HTML document.

No third-party dependencies: a deliberately small Markdown subset (headings, paragraphs,
fenced code, tables, lists, blockquotes, images, bold, inline code) is sufficient for the
report and keeps the toolchain dependency-free. Images are embedded as base64 so the HTML
file can be moved or opened offline without the figures directory.
"""

from __future__ import annotations

import base64
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "JSL_ARCHITECTURE_AND_BENCHMARK.md"
TARGET = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else SOURCE.with_suffix(".html")

CSS = """
:root {
  --ink: #111827; --muted: #6b7280; --line: #e5e7eb; --panel: #f9fafb;
  --accent: #1d4ed8; --accent-soft: #eff6ff; --code: #0f172a;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: #ffffff; color: var(--ink);
  font-family: "Segoe UI", "Microsoft YaHei", system-ui, -apple-system, sans-serif;
  font-size: 15px; line-height: 1.75;
}
main { max-width: 1040px; margin: 0 auto; padding: 48px 28px 96px; }
header.hero {
  border-bottom: 1px solid var(--line); padding-bottom: 28px; margin-bottom: 36px;
}
header.hero .eyebrow {
  color: var(--accent); font-weight: 700; letter-spacing: .12em;
  font-size: 12px; text-transform: uppercase;
}
header.hero h1 { font-size: 34px; line-height: 1.25; margin: 10px 0 12px; }
header.hero p { color: var(--muted); margin: 0; }
nav.toc {
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 18px 22px; margin: 0 0 40px;
}
nav.toc strong { display: block; margin-bottom: 8px; font-size: 13px; letter-spacing: .08em;
  text-transform: uppercase; color: var(--muted); }
nav.toc a { color: var(--ink); text-decoration: none; }
nav.toc a:hover { color: var(--accent); }
nav.toc ol { margin: 0; padding-left: 20px; columns: 2; column-gap: 36px; }
nav.toc li { margin: 3px 0; font-size: 13.5px; break-inside: avoid; }
h1, h2, h3, h4 { line-height: 1.35; margin: 2.2em 0 .7em; }
h2 { font-size: 25px; border-bottom: 1px solid var(--line); padding-bottom: 8px; }
h3 { font-size: 19px; }
h4 { font-size: 16px; }
p { margin: .8em 0; }
a { color: var(--accent); }
code {
  font-family: "Cascadia Code", "Consolas", monospace; font-size: 13px;
  background: #f3f4f6; padding: 1px 5px; border-radius: 4px;
}
pre {
  background: var(--code); color: #e5e7eb; padding: 16px 18px; border-radius: 10px;
  overflow-x: auto; line-height: 1.55;
}
pre code { background: transparent; color: inherit; padding: 0; font-size: 12.5px; }
table {
  border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 13.5px;
}
th, td { border: 1px solid var(--line); padding: 8px 10px; text-align: left; }
th { background: var(--panel); font-weight: 700; }
tr:nth-child(even) td { background: #fcfcfd; }
blockquote {
  margin: 18px 0; padding: 12px 18px; border-left: 3px solid var(--accent);
  background: var(--accent-soft); color: #1f2937;
}
figure { margin: 28px 0; text-align: center; }
figure img { max-width: 100%; border: 1px solid var(--line); border-radius: 8px; }
figcaption { color: var(--muted); font-size: 12.5px; margin-top: 8px; }
ul, ol { padding-left: 24px; }
li { margin: 4px 0; }
hr { border: none; border-top: 1px solid var(--line); margin: 36px 0; }
.small { color: var(--muted); font-size: 12.5px; }
@media (max-width: 720px) {
  main { padding: 28px 16px 64px; }
  nav.toc ol { columns: 1; }
  header.hero h1 { font-size: 26px; }
}
@media print {
  nav.toc { break-inside: avoid; }
  pre { white-space: pre-wrap; }
}
"""


def inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def image_block(path: str, alt: str) -> str:
    file = ROOT / path
    if not file.exists():
        return f'<p class="small">missing figure: {html.escape(path)}</p>'
    encoded = base64.b64encode(file.read_bytes()).decode("ascii")
    return (
        "<figure>"
        f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{encoded}">'
        f"<figcaption>{html.escape(alt)}</figcaption>"
        "</figure>"
    )


def parse_table(lines: list[str]) -> str:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    header = rows[0]
    body = rows[2:]
    parts = ["<table><thead><tr>"]
    parts.extend(f"<th>{inline(cell)}</th>" for cell in header)
    parts.append("</tr></thead><tbody>")
    for row in body:
        parts.append("<tr>")
        parts.extend(f"<td>{inline(cell)}</td>" for cell in row)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def convert(markdown: str) -> tuple[str, list[tuple[int, str, str]]]:
    lines = markdown.splitlines()
    out: list[str] = []
    toc: list[tuple[int, str, str]] = []
    index = 0
    heading_count = 0
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if not paragraph:
            return
        text = " ".join(part.strip() for part in paragraph).strip()
        paragraph = []
        if not text:
            return
        match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", text)
        if match:
            out.append(image_block(match.group(2), match.group(1)))
        else:
            out.append(f"<p>{inline(text)}</p>")

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            language = stripped[3:].strip()
            index += 1
            code: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            index += 1
            escaped = html.escape("\n".join(code))
            klass = f' class="language-{html.escape(language)}"' if language else ""
            out.append(f"<pre><code{klass}>{escaped}</code></pre>")
            continue

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            heading_count += 1
            anchor = f"h-{heading_count}"
            if level >= 2:
                toc.append((level, title, anchor))
            out.append(f'<h{level} id="{anchor}">{inline(title)}</h{level}>')
            index += 1
            continue

        if stripped == "---":
            flush_paragraph()
            out.append("<hr>")
            index += 1
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            quote: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote.append(lines[index].strip()[1:].strip())
                index += 1
            out.append(f"<blockquote>{inline(' '.join(quote))}</blockquote>")
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and re.match(
            r"^\|[\s:|-]+\|$", lines[index + 1].strip()
        ):
            flush_paragraph()
            table: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table.append(lines[index])
                index += 1
            out.append(parse_table(table))
            continue

        if re.match(r"^[-*]\s+", stripped):
            flush_paragraph()
            items: list[str] = []
            while index < len(lines) and re.match(r"^[-*]\s+", lines[index].strip()):
                items.append(re.sub(r"^[-*]\s+", "", lines[index].strip()))
                index += 1
            out.append("<ul>" + "".join(f"<li>{inline(item)}</li>" for item in items) + "</ul>")
            continue

        if re.match(r"^\d+\.\s+", stripped):
            flush_paragraph()
            items = []
            while index < len(lines) and re.match(r"^\d+\.\s+", lines[index].strip()):
                items.append(re.sub(r"^\d+\.\s+", "", lines[index].strip()))
                index += 1
            out.append("<ol>" + "".join(f"<li>{inline(item)}</li>" for item in items) + "</ol>")
            continue

        paragraph.append(line)
        index += 1

    flush_paragraph()
    return "\n".join(out), toc


def main() -> None:
    markdown = SOURCE.read_text(encoding="utf-8")
    body, toc = convert(markdown)
    title_match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else SOURCE.stem
    toc_html = "".join(
        f'<li style="margin-left:{(level - 2) * 14}px"><a href="#{anchor}">'
        f"{html.escape(title)}</a></li>"
        for level, title, anchor in toc
    )
    document = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<main>
<header class="hero">
  <div class="eyebrow">JSL · Pure-JDK 21 Machine Learning Runtime</div>
  <h1>{title}</h1>
  <p>JSL · Pure-JDK 21 Machine Learning Runtime · 可复现基准与对比</p>
</header>
<nav class="toc"><strong>目录</strong><ol>{toc_html}</ol></nav>
{body}
<p class="small">本页由 <code>report/make_report.py</code> 从 Markdown 生成，图表以内嵌 base64 保存，可离线打开。</p>
</main>
</body>
</html>
"""
    TARGET.write_text(document, encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT.parent)} ({TARGET.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
