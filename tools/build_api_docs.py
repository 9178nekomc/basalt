#!/usr/bin/env python3
"""Build the Basalt API documentation as a Sphinx/docs.python.org style HTML site.

Output layout (docs/):

    index.html             首页（文档章节网格）
    api/index.html         模块索引
    api/<module>.html      每个模块一页
    genindex.html          总索引
Each page is self-contained: CSS, the theme/search script and the search index
are inlined, so the site works from file://, from GitHub Pages (Jekyll) and from
static hosts that drop underscore-prefixed directories.

Signatures come from build/api/api.json (machine extracted from src/main/java);
prose comes from tools/docs_data/*.json (curated).  Neither is invented here.

Usage:
    python tools/build_api_docs.py --out docs
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import html
import json
import os
import re
import shutil
import sys
import urllib.parse

from md_render import render_markdown, set_java_highlighter

VERSION = "3.9.0"


def _mark_svg(stroke: str = "currentColor", weight: float = 24.0,
              px: int | None = None) -> str:
    """The Basalt mark: four cells of the hexagonal lattice basalt cools into.

    One stroke weight, no fill, no gradient -- the same geometry as
    assets/logo.svg, kept here so every page can inline it without a second file.
    """
    import math
    r, c = 84.0, 256.0
    dx = r * math.sqrt(3) / 2
    cells = [(c, c - 1.5 * r), (c - dx, c), (c + dx, c), (c, c + 1.5 * r)]
    paths = []
    for px_, py_ in cells:
        pts = [(px_ + r * math.cos(math.radians(90 + 60 * k)),
                py_ + r * math.sin(math.radians(90 + 60 * k))) for k in range(6)]
        paths.append("M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts) + " Z")
    size = f' width="{px}" height="{px}"' if px else ""
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"'
            f'{size} class="brand-mark" aria-hidden="true" focusable="false">'
            f'<g fill="none" stroke="{stroke}" stroke-width="{weight}" '
            'stroke-linecap="round" stroke-linejoin="round">'
            + "".join(f'<path d="{d}"/>' for d in paths) + "</g></svg>")


FAVICON = ("data:image/svg+xml," + urllib.parse.quote(_mark_svg("#14181d", 34.0, 32), safe=""))
BRAND_MARK = _mark_svg("currentColor", 38.0)
TITLE = f"Basalt {VERSION} 文档"
REPO = "https://github.com/9178nekomc/basalt"

MODULE_ORDER = [
    "basalt", "basalt.core", "basalt.data", "basalt.io", "basalt.ml", "basalt.nn",
    "basalt.nativebridge", "basalt.torch", "basalt.concurrent",
]

MODULE_SUMMARY = {
    "basalt": "统一门面：一次创建上下文，取张量、模块、优化器，并转发经典模型入口。",
    "basalt.core": "基础数据层：行主序矩阵、布局标记、原生经典核、模型容器读写原语。",
    "basalt.data": "BasaltD 数据管线：流式与内存映射两条读取路径、标准化参数。",
    "basalt.io": "模型持久化：.bsl v2 容器（magic + kind + 长度 + CRC32C）。",
    "basalt.ml": "经典机器学习：线性模型、近邻、SVM、树、梯度提升、聚类、降维、朴素贝叶斯、指标与切分。",
    "basalt.nn": "手写神经网络层：Dense / Conv2D / RNN / LSTM / GRU / 注意力、序列训练与序列化。",
    "basalt.nativebridge": "原生桥接专家层：上下文、张量句柄、128 字节指令队列、自动求导入口。",
    "basalt.torch": "eager 执行 API：动态张量、Module 组合、原生 autograd、优化器与损失。",
    "basalt.concurrent": "批量推理编排：虚拟线程 + 信号量限流。",
}


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #
def esc(text: str) -> str:
    return html.escape(text, quote=False)


CODE_SPAN_RE = re.compile(r"`([^`]+)`")
JAVADOC_CODE_RE = re.compile(r"\{@code\s+([^}]*)\}")
JAVADOC_LINK_RE = re.compile(r"\{@link\s+([^}\s]+)(?:\s+([^}]*))?\}")
JAVADOC_LITERAL_RE = re.compile(r"\{@literal\s+([^}]*)\}")


def inline(text: str, href_for=None) -> str:
    """Escape prose, expand `code`, {@code}, {@link} and [label](href)."""
    if not text:
        return ""
    text = JAVADOC_LITERAL_RE.sub(lambda m: m.group(1), text)
    text = JAVADOC_CODE_RE.sub(lambda m: "`" + m.group(1) + "`", text)

    def link(m: re.Match) -> str:
        target = m.group(1).strip()
        label = (m.group(2) or "").strip() or target
        href = href_for(target) if href_for else None
        return f"[{label}]({href})" if href else f"`{label}`"

    text = JAVADOC_LINK_RE.sub(link, text)
    out = esc(text)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                 lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', out)
    out = CODE_SPAN_RE.sub(lambda m: f"<code>{m.group(1)}</code>", out)
    return out.replace("\n", "<br />")


JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class",
    "const", "continue", "default", "do", "double", "else", "enum", "extends", "final",
    "finally", "float", "for", "goto", "if", "implements", "import", "instanceof", "int",
    "interface", "long", "native", "new", "package", "private", "protected", "public",
    "return", "short", "static", "strictfp", "super", "switch", "synchronized", "this",
    "throw", "throws", "transient", "try", "void", "volatile", "while", "var", "record",
    "sealed", "permits", "yield", "true", "false", "null",
}


def highlight_java(code: str) -> str:
    tokens: list[tuple[str, str]] = []
    buf: list[str] = []
    i, n = 0, len(code)

    def flush() -> None:
        if buf:
            tokens.append(("plain", "".join(buf)))
            buf.clear()

    while i < n:
        ch = code[i]
        if ch == "/" and i + 1 < n and code[i + 1] == "/":
            flush()
            j = code.find("\n", i)
            j = n if j == -1 else j
            tokens.append(("comment", code[i:j]))
            i = j
        elif ch == "/" and i + 1 < n and code[i + 1] == "*":
            flush()
            j = code.find("*/", i + 2)
            j = n if j == -1 else j + 2
            tokens.append(("comment", code[i:j]))
            i = j
        elif ch == '"':
            flush()
            j = i + 1
            while j < n:
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == '"':
                    j += 1
                    break
                j += 1
            tokens.append(("string", code[i:j]))
            i = j
        elif ch == "@" and i + 1 < n and (code[i + 1].isalpha() or code[i + 1] == "_"):
            flush()
            j = i + 1
            while j < n and (code[j].isalnum() or code[j] in "_."):
                j += 1
            tokens.append(("annotation", code[i:j]))
            i = j
        else:
            buf.append(ch)
            i += 1
    flush()

    out = []
    for kind, text in tokens:
        if kind != "plain":
            out.append(f'<span class="tok-{kind}">{esc(text)}</span>')
            continue
        for piece in re.split(r"(\b[A-Za-z_$][\w$]*\b|\b\d+(?:\.\d+)?[fFdDlL]?\b)", text):
            if not piece:
                continue
            if piece in JAVA_KEYWORDS:
                out.append(f'<span class="tok-kw">{piece}</span>')
            elif re.fullmatch(r"\d+(?:\.\d+)?[fFdDlL]?", piece):
                out.append(f'<span class="tok-num">{piece}</span>')
            else:
                out.append(esc(piece))
    return "".join(out)


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.]", "_", text)


def module_anchor(module: str) -> str:
    """Stable in-page anchor for a module heading."""
    return "module-" + re.sub(r"[^A-Za-z0-9_]", "_", module)


def page_name(module: str) -> str:
    """Module -> output file.  Dots become underscores so the name is portable."""
    return "api/" + re.sub(r"[^A-Za-z0-9]", "_", module) + ".html"


def anchor_type(qualified: str) -> str:
    return "type-" + slug(qualified)


def anchor_member(qualified: str, name: str, index: int) -> str:
    base = "m-" + slug(f"{qualified}.{name}")
    return base if index == 0 else f"{base}-{index}"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_api(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_docs(pattern: str) -> dict:
    types: dict[str, dict] = {}
    members: dict[tuple[str, str], dict] = {}
    intros: dict[str, list[str]] = {}
    for path in sorted(glob.glob(pattern)):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"WARN  {path}: {exc}", file=sys.stderr)
            continue
        for module in data.get("modules", []):
            if module.get("intro"):
                intros.setdefault(module["name"], module["intro"])
            for t in module.get("types", []):
                if t.get("doc"):
                    types[t["qualified"]] = t["doc"]
                for m in t.get("members", []):
                    if m.get("doc"):
                        members[(t["qualified"], m["key"])] = m["doc"]
    return {"types": types, "members": members, "intros": intros}


def visible_members(t: dict) -> list[dict]:
    out = []
    for m in t["members"]:
        mods = set(m.get("modifiers") or [])
        if "private" in mods:
            continue
        if m["kind"] == "constructor" and not (mods & {"public", "protected"}):
            continue
        if m["kind"] in ("method", "field") and not (mods & {"public", "protected"}):
            continue
        out.append(m)
    return out


# --------------------------------------------------------------------------- #
# CSS / JS
# --------------------------------------------------------------------------- #
CSS = r"""/* Basalt documentation — Sphinx / docs.python.org inspired theme */
:root{
  --bg:#ffffff; --fg:#1a1a1a; --muted:#5c6673; --border:#dfe3e8; --border-soft:#eef1f4;
  --surface:#f8fafc; --surface-2:#edf1f6;
  --sidebar-bg:#f8f8f8; --topbar-bg:#2b2b2b; --topbar-fg:#e6e6e6; --topbar-muted:#b4b4b4;
  --link:#0055bb; --link-hover:#003f8a; --link-visited:#6b2e8a;
  --sig-bg:#eef2f6; --sig-border:#d6dee7; --code-bg:#edf1f6; --code-fg:#1a1a1a;
  --shadow-soft:0 1px 2px rgba(16,24,40,.04), 0 6px 20px -14px rgba(16,24,40,.20);
  --shadow-pop:0 16px 40px -16px rgba(16,24,40,.30), 0 2px 8px rgba(16,24,40,.06);
  --admon-bg:#f3f7fb; --admon-border:#4a7fb5; --admon-ink:#2f5d8a; --table-head:#f2f2f2;
  --tok-kw:#8a2be2; --tok-str:#0a7a3d; --tok-com:#6b6b6b; --tok-num:#92400e; --tok-ann:#0f766e;
}
html[data-theme="dark"]{
  color-scheme:dark;
  --bg:#1a1c1f; --fg:#d8dde3; --muted:#939dab; --border:#2e343b; --border-soft:#252b32;
  --surface:#20242a; --surface-2:#272d35;
  --sidebar-bg:#262626; --topbar-bg:#1a1a1a; --topbar-fg:#e0e0e0; --topbar-muted:#9a9a9a;
  --link:#7ab7ff; --link-hover:#a6cdff; --link-visited:#c39bf0;
  --sig-bg:#22272e; --sig-border:#333b44; --code-bg:#272d35; --code-fg:#e0e0e0;
  --shadow-soft:0 1px 2px rgba(0,0,0,.30), 0 6px 20px -14px rgba(0,0,0,.60);
  --shadow-pop:0 16px 40px -16px rgba(0,0,0,.65), 0 2px 8px rgba(0,0,0,.40);
  --admon-bg:#212a33; --admon-border:#4f86bd; --admon-ink:#8ec3ea; --table-head:#2a2a2a;
  --tok-kw:#c792ea; --tok-str:#8fd3a3; --tok-com:#a0a8b3; --tok-num:#e2b93d; --tok-ann:#66c2b5;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans SC","Microsoft YaHei",sans-serif;
  font-size:16px;line-height:1.55;-webkit-font-smoothing:antialiased}
a{color:var(--link);text-decoration:none}
a:hover{color:var(--link-hover);text-decoration:underline}
a:visited{color:var(--link-visited)}
code,pre,kbd{font-family:"SFMono-Regular",Menlo,Consolas,"DejaVu Sans Mono",monospace;font-size:.875em}
code{background:var(--code-bg);color:var(--code-fg);padding:.1em .3em;border-radius:3px;
  border:1px solid var(--border-soft);word-break:break-word}
h1,h2,h3,h4{font-weight:700;line-height:1.25}
h1{font-size:2.25rem;margin:.4rem 0 1rem}
h2{font-size:1.5rem;margin:2rem 0 .8rem}
h3{font-size:1.2rem;margin:1.6rem 0 .6rem}
h4{font-size:.95rem;margin:1.4rem 0 .5rem;color:var(--muted);text-transform:uppercase;
  letter-spacing:.06em;border-bottom:1px solid var(--border-soft);padding-bottom:.3rem}
/* member group labels ("构造器"/"方法"/"字段") are rubrics, not headings: docs.python.org
   does the same, and it keeps the outline free of h2 -> h4 jumps inside a type block. */
.rubric{font-weight:700;font-size:.95rem;margin:1.4rem 0 .5rem;color:var(--muted);
  text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border-soft);
  padding-bottom:.3rem}

.topbar{position:sticky;top:0;z-index:40;display:flex;align-items:center;justify-content:space-between;
  gap:1rem;height:44px;padding:0 1rem;background:var(--topbar-bg);color:var(--topbar-fg);
  font-size:.8125rem}
.topbar a{color:var(--topbar-fg)}
.brand-mark{width:21px;height:21px;flex:0 0 auto;margin-right:.55rem;opacity:.92}
@media (max-width:900px){.brand-mark{display:none}}
.topbar a:hover{color:#fff;text-decoration:none}
.crumbs{display:flex;align-items:center;gap:.4rem;flex-wrap:wrap;min-width:0}
.crumbs .sep{color:var(--topbar-muted)}
.crumbs .cur{color:var(--topbar-muted)}
.topbar-tools{display:flex;align-items:center;gap:.9rem;white-space:nowrap}
.theme-select{background:transparent;color:var(--topbar-fg);border:1px solid #5a5a5a;
  border-radius:3px;padding:2px 4px;font-size:.8125rem}
.theme-select option{background:var(--topbar-bg);color:var(--topbar-fg)}
.tool-btn{background:transparent;border:1px solid transparent;color:var(--topbar-fg);
  font-size:.8125rem;cursor:pointer;padding:2px 6px;border-radius:3px}
.tool-btn:hover{border-color:#5a5a5a}
kbd{background:#3a3a3a;border:1px solid #555;border-radius:3px;padding:0 4px;color:#ddd;font-size:.75rem}

.shell{display:flex;align-items:flex-start;min-height:calc(100vh - 44px)}
.sidebar{position:sticky;top:44px;flex:0 0 300px;width:300px;max-height:calc(100vh - 44px);
  overflow-y:auto;background:var(--sidebar-bg);border-right:1px solid var(--border);
  padding:1rem .9rem 3rem;font-size:.8125rem}
.sidebar-head{font-weight:700;font-size:.875rem;margin:1.2rem 0 .4rem;color:var(--fg)}
.sidebar-head:first-child{margin-top:0}
.sidebar ul{list-style:none;margin:0 0 .6rem;padding:0}
.sidebar li{margin:.12rem 0;line-height:1.4}
.sidebar li a{display:block;padding:.12rem .2rem;color:var(--fg)}
.sidebar li a:hover{color:var(--link);text-decoration:none;background:var(--border-soft)}
.sidebar li.active>a{color:var(--link);font-weight:700}
main{flex:1 1 auto;min-width:0;padding:1.4rem 2.6rem 4rem;max-width:1100px}
article{max-width:78ch}
article.wide{max-width:none}

p.biglink{font-size:1.5rem;font-weight:700;margin:0}
span.linkdescr{display:block;font-size:.875rem;color:var(--muted);font-style:italic;margin-top:.15rem}
ul.chapters{list-style:none;padding:0;margin:1rem 0 1.5rem;
  display:grid;grid-template-columns:1fr 1fr;gap:1.4rem 2.5rem}
ul.chapters li{margin:0}
ul.chapters a{font-size:1.5rem;font-weight:700}
.welcome{margin:.2rem 0 1.6rem;color:var(--muted)}
.abstract{border-left:3px solid var(--admon-border);background:var(--admon-bg);
  padding:.8rem 1rem;border-radius:0 4px 4px 0;margin:1rem 0}
.abstract p{margin:.4rem 0}

dl.py{margin:0 0 1.1rem}
dl.py>dt{font-family:"SFMono-Regular",Menlo,Consolas,monospace;font-size:.875rem;
  background:var(--sig-bg);border-left:3px solid var(--sig-border);padding:.45rem .7rem;
  border-radius:2px;position:relative;overflow-x:auto;white-space:nowrap}
dl.py.method>dt,dl.py.constructor>dt{background:transparent;border-left:0;
  border-bottom:1px solid var(--border-soft);border-radius:0;padding-left:0}
dl.py>dt .sig-name{font-weight:700;color:var(--fg)}
dl.py>dt .sig-paren{color:var(--muted)}
dl.py>dt .sig-param{font-style:italic;color:var(--fg)}
dl.py>dt .sig-return{color:var(--muted)}
dl.py>dt .sig-return-typehint{color:var(--muted)}
dl.py>dt .property{color:var(--muted);font-size:.9em}
dl.py>dt .headerlink{float:right;opacity:0;color:var(--muted);margin-left:1rem;font-size:.85em}
dl.py>dt:hover .headerlink{opacity:1}
dl.py>dd{margin:.5rem 0 .9rem 0;padding-left:0}
dl.py.method>dd,dl.py.constructor>dd{padding-left:.9rem;border-left:1px solid var(--border-soft)}
dl.py.class>dd{margin-top:.7rem}
.py-flag{font-size:.7rem;color:var(--muted);border:1px solid var(--border);border-radius:8px;
  padding:0 .35rem;margin-left:.4rem;font-family:sans-serif;vertical-align:middle}

dl.field-list{display:grid;grid-template-columns:5.5rem 1fr;gap:.25rem .8rem;margin:.6rem 0 .3rem;
  font-size:.9375rem}
dl.field-list>dt{color:var(--muted);font-weight:600;font-size:.8125rem;padding-top:.15rem}
dl.field-list>dd{margin:0}
dl.field-list ul{list-style:none;margin:0;padding:0}
dl.field-list li{margin:.15rem 0}
dl.field-list .param-name{font-weight:600}
dl.field-list .param-type{color:var(--muted);font-style:italic}
dl.field-list .exc{font-family:monospace;font-size:.85em}
.summary{font-size:1rem}
p{margin:.6rem 0}
ul.simple{padding-left:1.4rem}
ul.simple li{margin:.25rem 0}
.example{margin:.9rem 0}
.example-label{font-size:.75rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);
  margin-bottom:.3rem}
pre.code{background:var(--code-bg);border:1px solid var(--border);border-radius:4px;
  padding:.7rem .9rem;overflow-x:auto;margin:0;line-height:1.5}
pre.code code{background:none;border:0;padding:0;font-size:.8125rem}
.tok-kw{color:var(--tok-kw)} .tok-str{color:var(--tok-str)}
.tok-com{color:var(--tok-com);font-style:italic}
.tok-num{color:var(--tok-num)} .tok-ann{color:var(--tok-ann)}
blockquote{margin:.9rem 0;padding:.15rem 0 .15rem .85rem;
  border-left:3px solid var(--border);color:var(--muted)}
blockquote p{margin:.35rem 0}
.admonition{background:var(--admon-bg);border-left:3px solid var(--admon-border);
  padding:.6rem .9rem;margin:.9rem 0;border-radius:0 4px 4px 0}
.admonition-title{font-weight:700;margin:0 0 .2rem;font-size:.8125rem;color:var(--admon-ink);
  text-transform:uppercase;letter-spacing:.05em}
.admonition p:last-child{margin-bottom:0}
.seealso{color:var(--muted);font-size:.875rem;margin:.5rem 0}
.undocumented{color:var(--muted);font-style:italic}

table{border-collapse:collapse;width:100%;margin:.9rem 0;font-size:.9rem}
table th,table td{border:1px solid var(--border);padding:.4rem .6rem;text-align:left;vertical-align:top}
table thead th{background:var(--table-head);font-weight:700}
table td.num{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
table.autosummary td:first-child{white-space:nowrap}

.genindex-jumpbox{margin:.6rem 0 1.2rem;font-size:.875rem}
.genindex-jumpbox a{margin-right:.5rem}
ul.genindex-section{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));
  gap:.2rem 1.4rem;list-style:none;padding:0;margin:0 0 1.2rem}
ul.genindex-section li{font-size:.9rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

.relbar{display:flex;justify-content:space-between;gap:1rem;margin:2.5rem 0 0;padding-top:1rem;
  border-top:1px solid var(--border);font-size:.875rem}
.relbar .prev::before{content:"« ";color:var(--muted)}
.relbar .next::after{content:" »";color:var(--muted)}
.relbar .spacer{flex:1}
footer.doc-footer{margin-top:2rem;padding-top:1rem;border-top:1px solid var(--border);
  color:var(--muted);font-size:.8125rem}
footer.doc-footer a{color:var(--muted);text-decoration:underline}

.nav-toggle{display:none;background:transparent;border:1px solid #5a5a5a;color:var(--topbar-fg);
  font-size:1rem;line-height:1;padding:.25rem .45rem;border-radius:4px;cursor:pointer;margin-right:.4rem}
.nav-toggle:hover{border-color:#8a8a8a}
.nav-mask{position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:45;display:none}
body.nav-open .nav-mask{display:block}
.table-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:.9rem 0}
.table-wrap table{margin:0}
.search-mask{position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:60;display:none}
.search-mask.open{display:block}
.search-dialog{position:fixed;top:12vh;left:50%;transform:translateX(-50%);width:min(720px,92vw);
  background:var(--bg);border:1px solid var(--border);border-radius:8px;z-index:61;
  box-shadow:0 18px 48px rgba(0,0,0,.35);display:none;overflow:hidden}
.search-dialog.open{display:block}
.search-dialog input{width:100%;padding:.85rem 1rem;border:0;border-bottom:1px solid var(--border);
  background:var(--bg);color:var(--fg);font-size:1rem;outline:none}
.search-results{max-height:56vh;overflow-y:auto}
.search-results a{display:block;padding:.6rem 1rem;border-bottom:1px solid var(--border-soft);
  color:var(--fg)}
.search-results a:hover{background:var(--sidebar-bg);text-decoration:none}
.search-results .sr-name{font-family:monospace;font-size:.9rem;color:var(--link);font-weight:600}
.search-results .sr-meta{display:block;color:var(--muted);font-size:.78rem;margin-top:.15rem}
.search-empty{padding:1rem;color:var(--muted);font-size:.9rem}


/* ---------- motion, depth, tree ---------- */
:root{ --dur:180ms; --dur-slow:260ms; --ease:cubic-bezier(.2,.7,.3,1); }

/* Only while switching: painting a 200 ms cross-fade beats an instant flip. */
html.theme-anim, html.theme-anim *{
  transition:background-color var(--dur) var(--ease), color var(--dur) var(--ease),
    border-color var(--dur) var(--ease), box-shadow var(--dur) var(--ease) !important;
}
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation-duration:.001ms !important;animation-iteration-count:1 !important;
    transition-duration:.001ms !important;scroll-behavior:auto !important}
}

/* cross-document fade (Chrome 126+): the chrome and background stay put and only
   the article cross-fades, so navigating between pages no longer hard-cuts. */
@view-transition { navigation: auto; }
.topbar{view-transition-name:jsl-topbar}
main{view-transition-name:jsl-main}
::view-transition-old(root),::view-transition-new(root){animation:none;mix-blend-mode:normal}
::view-transition-old(jsl-main){animation:fadeOut 90ms ease both}
::view-transition-new(jsl-main){animation:fadeIn 220ms var(--ease) both}
::view-transition-old(jsl-topbar),::view-transition-new(jsl-topbar){animation:none}
@keyframes fadeOut{to{opacity:0}}
@keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}

/* content rises in once instead of appearing in a single frame */
main{animation:pageIn var(--dur-slow) var(--ease) both}
@keyframes pageIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}

/* top progress bar: real feedback for same-document + cross-page navigation */
#pbar{position:fixed;top:0;left:0;height:2px;width:0;z-index:80;background:var(--accent);
  opacity:0;transition:width 260ms var(--ease),opacity 200ms linear}
#pbar.on{opacity:1}

/* three-dot loader */
.dots{display:inline-flex;gap:4px;align-items:center;vertical-align:middle}
.dots i{display:block;width:5px;height:5px;border-radius:50%;background:currentColor;opacity:.3;
  animation:dot 1.05s var(--ease) infinite}
.dots i:nth-child(2){animation-delay:.14s}
.dots i:nth-child(3){animation-delay:.28s}
@keyframes dot{0%,100%{opacity:.25;transform:translateY(0)}45%{opacity:.95;transform:translateY(-2px)}}

/* sidebar tree: rotating caret, guide rail, active marker */
.nav-modules{margin:0 0 .7rem;padding:0;list-style:none}
.nav-modules>li{margin:.08rem 0}
.nav-modules summary a{font-weight:600;color:var(--fg);display:inline}
.nav-modules summary:hover a{color:var(--link)}
.nav-modules ul{margin:.1rem 0 .3rem .55rem;padding-left:.45rem;
  border-left:1px solid var(--border);list-style:none}
.nav-modules ul a{display:block;font-size:.78rem;color:var(--muted);
  padding:.1rem .35rem;border-radius:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.nav-modules ul a:hover{color:var(--link)}
.nav-modules details>summary{position:relative;padding:.1rem 0 .1rem 1.2rem;cursor:pointer;
  list-style:none}
.nav-modules details>summary::-webkit-details-marker{display:none}
.nav-modules details>summary::marker{content:""}
.nav-modules details>summary::before{content:"";position:absolute;left:.3rem;top:50%;
  margin-top:-3px;width:0;height:0;border-left:5px solid currentColor;
  border-top:3.5px solid transparent;border-bottom:3.5px solid transparent;
  opacity:.45;transition:transform var(--dur) var(--ease)}
.nav-modules details[open]>summary::before{transform:rotate(90deg)}
.nav-modules details[open]>ul{animation:treeIn var(--dur) var(--ease) both}
@keyframes treeIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}
.nav-modules ul li a{position:relative;transition:background-color var(--dur) var(--ease),
  color var(--dur) var(--ease)}
.nav-modules ul li a:hover{background:var(--surface-2);border-radius:4px;text-decoration:none}
.nav-modules li.active>a{color:var(--link);font-weight:650}
.nav-modules li.active>a::before{content:"";position:absolute;left:0;top:.32em;bottom:.32em;
  width:2px;border-radius:1px;background:var(--link)}
.sidebar-head{letter-spacing:.02em}

/* interaction polish */
.sidebar a,.topbar a,.tool-btn,.genlist a{transition:color var(--dur) var(--ease),
  background-color var(--dur) var(--ease),border-color var(--dur) var(--ease)}
ul.chapters a{transition:color var(--dur) var(--ease),transform var(--dur) var(--ease)}
ul.chapters li:hover a{transform:translateX(2px)}

/* depth by background, not by borders; shadows only for floating layers */
dl.py.class{background:var(--surface);border:1px solid var(--border-soft);border-radius:10px;
  padding:.25rem 1rem .8rem;margin:0 0 1.35rem}
dl.py.class>dt{background:transparent;border-left:0;padding-left:0;padding-bottom:.5rem;
  border-bottom:1px solid var(--border-soft);border-radius:0}
dl.py.class>dd{margin-top:.2rem}
dl.py.method>dt,dl.py.constructor>dt{border-bottom:1px solid var(--border-soft)}
table{background:var(--surface);border-radius:8px;overflow:hidden}
table thead th{background:var(--surface-2)}
.abstract{background:var(--surface);box-shadow:none}
.search-dialog{box-shadow:var(--shadow-pop);opacity:0;transform:translateX(-50%) translateY(-6px);
  transition:opacity var(--dur) var(--ease),transform var(--dur) var(--ease)}
.search-dialog.open{opacity:1;transform:translateX(-50%) translateY(0)}
.search-mask{opacity:0;transition:opacity var(--dur) linear}
.search-mask.open{opacity:1}
.genindex-section li{transition:background-color var(--dur) var(--ease)}
ul.genindex-section li:hover{background:var(--surface-2);border-radius:4px}

/* ---------- tablet ---------- */
@media (max-width:1024px){
  .sidebar{flex:0 0 260px;width:260px}
  main{padding:1.2rem 1.6rem 3rem}
}
/* ---------- phone: drawer navigation ---------- */
@media (max-width:900px){
  html,body{overflow-x:hidden}
  .topbar{height:auto;min-height:48px;padding:.4rem .7rem;flex-wrap:nowrap;gap:.4rem;
    overflow:hidden}
  .nav-toggle{display:inline-block;flex:0 0 auto}
  .crumbs{font-size:.78rem;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;
    flex:1 1 auto;min-width:0;flex-wrap:nowrap;padding-right:.5rem;
    mask-image:linear-gradient(to right,#000 calc(100% - 12px),transparent)}
  .crumbs>*{flex:0 0 auto}
  .crumbs .cur:last-child{overflow:hidden;text-overflow:ellipsis}
  .crumbs .sep{margin:0 .25rem}
  .crumb-version,.crumb-version+.sep,
  .crumb-mid,.crumb-mid+.sep{display:none}
  .topbar-tools{gap:.3rem;flex:0 0 auto}
  .topbar-tools .tool-btn{padding:.2rem .3rem;font-size:.78rem}
  .topbar-tools kbd{display:none}
  .topbar-tools a[href*="api/index"]{display:none}
  .topbar-tools a[href*="genindex"]{display:none}
  .theme-select{max-width:4.6rem;font-size:.78rem}
  .shell{display:block}
  .sidebar{position:fixed;top:0;left:0;bottom:0;width:min(84vw,320px);max-height:none;
    transform:translateX(-102%);transition:transform .22s ease;z-index:50;
    padding-top:1rem;box-shadow:0 0 0 1px var(--border), 0 18px 44px -18px rgba(0,0,0,.45)}
  body.nav-open .sidebar{transform:none}
  /* the drawer is used with a thumb, so give every row a real touch target */
  .nav-toggle{min-width:38px;min-height:38px;display:inline-flex;align-items:center;
    justify-content:center;margin-right:.2rem}
  .topbar-tools .tool-btn{min-height:34px;display:inline-flex;align-items:center}
  .theme-select{min-height:34px}
  .sidebar li a{padding:.42rem .35rem}
  .nav-modules details>summary{padding-top:.42rem;padding-bottom:.42rem}
  .nav-modules ul a{padding:.4rem .35rem}
  .topbar a{padding-top:.4rem;padding-bottom:.4rem}
  .relbar a{padding:.5rem 0}
  main{padding:1rem .95rem 3rem;max-width:none}
  article{max-width:none}
  h1{font-size:1.6rem}
  h2{font-size:1.22rem;margin:1.5rem 0 .6rem}
  h3{font-size:1.05rem}
  body{font-size:15.5px}
  ul.chapters{grid-template-columns:1fr;gap:1rem}
  p.biglink,ul.chapters a{font-size:1.22rem}
  /* signatures wrap instead of scrolling sideways */
  dl.py>dt{white-space:normal;overflow-wrap:anywhere;line-height:1.5}
  dl.py.method>dd,dl.py.constructor>dd{padding-left:.6rem}
  /* field lists stack */
  dl.field-list{grid-template-columns:1fr;gap:.1rem .5rem}
  dl.field-list>dt{padding-top:.35rem}
  pre.code{font-size:.75rem;padding:.6rem .7rem}
  .table-wrap table{min-width:0}
  table{font-size:.78rem}
  table th,table td{padding:.3rem .4rem}
  table.autosummary td:first-child{white-space:nowrap}
  table td code{font-size:.76rem;padding:0 .2em}
  .relbar{flex-direction:column;align-items:stretch;gap:.5rem}
  .relbar .spacer{display:none}
  .relbar .next{text-align:right}
  ul.genindex-section{grid-template-columns:1fr}
  footer.doc-footer{padding:1rem .95rem 2.5rem}
}
/* ---------- small phone ---------- */
@media (max-width:420px){
  .crumbs{font-size:.74rem}
  .table-wrap table{min-width:0}
  .admonition{padding:.5rem .7rem}
  h1{font-size:1.45rem}
  .topbar{padding:.35rem .5rem}
}
"""

JS = r"""(function(){
  var KEY='basalt-docs-theme';
  var root=document.documentElement;
  var media=window.matchMedia?window.matchMedia('(prefers-color-scheme: dark)'):null;
  function resolve(pref){return pref==='auto'?(media&&media.matches?'dark':'light'):pref;}
  var animTimer=null;
  function apply(pref,animate){
    if(animate){
      root.classList.add('theme-anim');
      clearTimeout(animTimer);
      animTimer=setTimeout(function(){root.classList.remove('theme-anim');},320);
    }
    root.setAttribute('data-theme',resolve(pref));
    root.setAttribute('data-theme-pref',pref);
    var sel=document.getElementById('theme-select');
    if(sel&&sel.value!==pref){sel.value=pref;}
    try{localStorage.setItem(KEY,pref);}catch(e){}
  }
  var pref='auto';
  try{pref=localStorage.getItem(KEY)||'auto';}catch(e){}
  var qm=/[?&]theme=(light|dark|auto)/.exec(location.search);
  if(qm){pref=qm[1];}
  apply(pref);
  var sel=document.getElementById('theme-select');
  if(sel){sel.addEventListener('change',function(){apply(sel.value,true);});}
  if(media&&media.addEventListener){media.addEventListener('change',function(){
    if((root.getAttribute('data-theme-pref')||'auto')==='auto'){apply('auto',true);}
  });}

  // ---- navigation progress bar ----
  var pbar=document.getElementById('pbar');
  function barStart(){
    if(!pbar){return;}
    pbar.classList.add('on');
    pbar.style.width='16%';
    setTimeout(function(){if(pbar.style.width==='16%'){pbar.style.width='68%';}},160);
  }
  function barDone(){
    if(!pbar){return;}
    pbar.style.width='100%';
    setTimeout(function(){pbar.classList.remove('on');pbar.style.width='0';},240);
  }
  window.addEventListener('pageshow',barDone);
  document.addEventListener('click',function(e){
    var a=e.target.closest?e.target.closest('a'):null;
    if(!a){return;}
    var href=a.getAttribute('href')||'';
    if(!href||href.charAt(0)==='#'||a.target==='_blank'
       ||href.indexOf('http')===0||href.indexOf('mailto:')===0){return;}
    if(e.metaKey||e.ctrlKey||e.shiftKey||e.button!==0){return;}
    barStart();
  });

  // ---- mobile drawer ----
  var navToggle=document.getElementById('nav-toggle');
  var navMask=document.getElementById('nav-mask');
  function setNav(open){
    document.body.classList.toggle('nav-open',open);
    if(navToggle){navToggle.setAttribute('aria-expanded',open?'true':'false');}
  }
  if(navToggle){navToggle.addEventListener('click',function(){
    setNav(!document.body.classList.contains('nav-open'));
  });}
  if(navMask){navMask.addEventListener('click',function(){setNav(false);});}
  document.addEventListener('click',function(e){
    var a=e.target.closest?e.target.closest('.sidebar a'):null;
    if(a&&window.innerWidth<=900){setNav(false);}
  });
  document.addEventListener('keydown',function(e){
    if(e.key==='Escape'){setNav(false);}
  });
  window.addEventListener('resize',function(){
    if(window.innerWidth>900){setNav(false);}
  });

  var mask=document.getElementById('search-mask');
  var dlg=document.getElementById('search-dialog');
  var input=document.getElementById('search-input');
  var box=document.getElementById('search-results');
  var data=window.__SEARCH_INDEX__||[];
  var ROOT=window.__DOCS_ROOT__||'';
  function closeSearch(){if(dlg){dlg.classList.remove('open');}if(mask){mask.classList.remove('open');}}
  function openSearch(){
    if(!dlg){return;}
    dlg.classList.add('open');if(mask){mask.classList.add('open');}
    input.value='';input.focus();box.innerHTML='';
  }
  var dotsHtml='<div class="search-empty"><span class="dots"><i></i><i></i><i></i></span></div>';
  function render(q){
    q=q.trim().toLowerCase();
    if(!q){box.innerHTML='';return;}
    var t0=window.performance&&performance.now?performance.now():0;
    var out=[];
    for(var i=0;i<data.length&&out.length<50;i++){
      var it=data[i];
      if(it[0].toLowerCase().indexOf(q)>=0||it[1].toLowerCase().indexOf(q)>=0||
         (it[3]&&it[3].toLowerCase().indexOf(q)>=0)){out.push(it);}
    }
    out.sort(function(a,b){return (a[5]==='type'?0:1)-(b[5]==='type'?0:1);});
    if(!out.length){box.innerHTML='<div class="search-empty">没有匹配项</div>';return;}
    var h='';
    for(var j=0;j<out.length;j++){
      var o=out[j];
      h+='<a href="'+ROOT+o[4]+'"><span class="sr-name">'+o[1]+'</span>'+
         '<span class="sr-meta">'+o[2]+(o[3]?' · '+o[3]:'')+'</span></a>';
    }
    var elapsed=(window.performance&&performance.now?performance.now():0)-t0;
    if(elapsed>50){
      // filter was slow: paint the spinner first, then the rows on the next frame
      box.innerHTML=dotsHtml;
      requestAnimationFrame(function(){box.innerHTML=h;});
    }else{
      box.innerHTML=h;
    }
  }
  if(input){input.addEventListener('input',function(){render(input.value);});}
  document.addEventListener('keydown',function(e){
    if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){e.preventDefault();openSearch();}
    else if(e.key==='Escape'){closeSearch();}
  });
  var trigger=document.getElementById('search-trigger');
  if(trigger){trigger.addEventListener('click',openSearch);}
  if(mask){mask.addEventListener('click',closeSearch);}
  if(input){input.addEventListener('keydown',function(e){
    if(e.key==='Enter'){var a=box.querySelector('a');if(a){location.href=a.getAttribute('href');}}
  });}
})();
"""


# --------------------------------------------------------------------------- #
# page shell
# --------------------------------------------------------------------------- #
def sidebar_html(active: str, page_toc: list[tuple[str, str]] | None = None,
                 root: str = "") -> str:
    def item(label: str, href: str, key: str = "") -> str:
        cls = ' class="active"' if key and key == active else ""
        full = href if href.startswith(("http://", "https://", "#")) else root + href
        return f'<li{cls}><a href="{full}">{label}</a></li>'

    parts = ['<aside class="sidebar">']
    if NAV_MODULES:
        parts.append('<div class="sidebar-head">模块</div><ul class="nav-modules">')
        for name, page_file, types in NAV_MODULES:
            current = name == active
            parts.append(
                f'<li class="{"active" if current else ""}">'
                f'<details{" open" if current else ""}>'
                f'<summary><a href="{root}{page_file}#{module_anchor(name)}">{esc(name)}</a></summary><ul>')
            for simple, anchor in types:
                href = f"#{anchor}" if current else f"{root}{page_file}#{anchor}"
                parts.append(f'<li><a href="{href}">{esc(simple)}</a></li>')
            parts.append("</ul></details></li>")
        parts.append("</ul>")

    parts.append('<div class="sidebar-head">下载</div><ul>')
    # Absolute URLs: the published site only ships docs/, so ../target would 404.
    # Release assets are the canonical download location.
    rel = f"{REPO}/releases/download/v{VERSION}"
    parts.append(item(f"Basalt {VERSION} 发布页（全部资产）", f"{REPO}/releases/tag/v{VERSION}"))
    parts.append(item(f"basalt-{VERSION}-all.jar", f"{rel}/basalt-{VERSION}-all.jar"))
    parts.append(item(f"basalt-{VERSION}.jar", f"{rel}/basalt-{VERSION}.jar"))
    parts.append(item(f"basalt-{VERSION}-sources.jar",
                      f"{rel}/basalt-{VERSION}-sources.jar"))
    parts.append(item("basalt_native.dll（Windows x86_64）", f"{rel}/basalt_native.dll"))
    parts.append(item(f"basalt-docs-{VERSION}.zip（离线文档）", f"{rel}/basalt-docs-{VERSION}.zip"))
    parts.append("</ul>")

    parts.append('<div class="sidebar-head">各版文档</div><ul>')
    parts.append(item(f"Basalt {VERSION} (stable)", "index.html", "index"))
    parts.append(item("模块索引", "api/index.html", "modindex"))
    parts.append(item("总索引", "genindex.html", "genindex"))
    parts.append("</ul>")

    parts.append('<div class="sidebar-head">Other resources</div><ul>')
    parts.append(item("README", "readme.html", "readme"))
    parts.append(item("架构与基准报告", "report/Basalt_ARCHITECTURE_AND_BENCHMARK.html"))
    parts.append(item("与参考实现的对比", "report/Basalt_VS_REFERENCE.html"))
    parts.append(item("GitHub 仓库", REPO))
    parts.append("</ul>")
    parts.append("</aside>")
    return "".join(parts)


def topbar_html(crumbs: list[tuple[str, str | None]], root: str) -> str:
    """Crumbs carry final page-relative hrefs; only absolute URLs are left untouched."""
    bits = []
    for label, href in crumbs:
        if href:
            bits.append(f'<a href="{href}">{esc(label)}</a>')
        elif label == VERSION:
            bits.append(f'<span class="cur crumb-version">{esc(label)}</span>')
        elif label == "API 参考":
            bits.append(f'<a class="crumb-mid" href="{href}">{esc(label)}</a>')
        else:
            bits.append(f'<span class="cur">{esc(label)}</span>')
    crumb = '<span class="sep">»</span>'.join(bits)
    return f'''<header class="topbar">
<button id="nav-toggle" class="nav-toggle" type="button" aria-label="打开目录"
        aria-controls="sidebar" aria-expanded="false">☰</button>
{BRAND_MARK}
<div class="crumbs">{crumb}</div>
<div class="topbar-tools">
  <label>主题 <select id="theme-select" class="theme-select" aria-label="主题">
    <option value="auto">自动</option><option value="light">浅色</option><option value="dark">深色</option>
  </select></label>
  <button id="search-trigger" class="tool-btn" type="button">快速搜索 <kbd>Ctrl</kbd><kbd>K</kbd></button>
  <a class="tool-btn" href="{root}api/index.html">模块</a>
  <a class="tool-btn" href="{root}genindex.html">索引</a>
  <a class="tool-btn" href="{REPO}" rel="noopener">GitHub</a>
</div>
</header>'''


# Inlined into every page by page(); assigned in main() before rendering.
SEARCH_JSON = "[]"

# (module, page file, [(simple type name, anchor)]) built once in main().
NAV_MODULES: list[tuple[str, str, list[tuple[str, str]]]] = []

SEARCH_OVERLAY = '''<div id="search-mask" class="search-mask"></div>
<div id="search-dialog" class="search-dialog" role="dialog" aria-label="搜索">
  <input id="search-input" type="search" placeholder="搜索类型、方法或说明…" autocomplete="off" />
  <div id="search-results" class="search-results"></div>
</div>'''


def page(title: str, crumbs: list[tuple[str, str | None]], body: str, *,
         active: str = "", toc: list[tuple[str, str]] | None = None,
         root: str = "") -> str:
    """Render one fully self-contained page.

    CSS, the theme/search script and the search index are inlined on purpose:
    static hosts (including the WorkBuddy publish preview and Jekyll on GitHub
    Pages) skip directories whose name starts with an underscore, so an external
    ``_static/`` folder silently 404s and the page renders unstyled.
    """
    # tables are wide by nature; wrap them so phones scroll them instead of
    # stretching the whole page
    body = body.replace("<table", '<div class="table-wrap"><table')
    body = body.replace("</table>", "</table></div>")
    # Runs before the first paint: without it the document paints light and then
    # flips to dark as soon as the footer script executes (a visible white flash).
    theme_boot = (
        "(function(){var r=document.documentElement;try{"
        "var p=localStorage.getItem('basalt-docs-theme')||'auto';"
        "var q=/[?&]theme=(light|dark|auto)/.exec(location.search);if(q){p=q[1];}"
        "var d=p==='dark'||(p==='auto'&&window.matchMedia&&"
        "window.matchMedia('(prefers-color-scheme: dark)').matches);"
        "r.setAttribute('data-theme',d?'dark':'light');"
        "r.setAttribute('data-theme-pref',p);}catch(e){r.setAttribute('data-theme','light');}})();"
    )
    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<script>{theme_boot}</script>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{esc(title)}</title>
<link rel="icon" href="{FAVICON}" type="image/svg+xml" />
<meta name="theme-color" content="#1a1c1f" media="(prefers-color-scheme: dark)" />
<meta name="theme-color" content="#ffffff" media="(prefers-color-scheme: light)" />
<meta name="generator" content="tools/build_api_docs.py" />
<style>{CSS}</style>
</head>
<body>
<div id="pbar"></div>
{topbar_html(crumbs, root)}
<div class="shell">
{sidebar_html(active, toc, root)}
<main><article>
{body}
</article></main>
</div>
<div id="nav-mask" class="nav-mask"></div>
{SEARCH_OVERLAY}
<script>window.__DOCS_ROOT__="{root}";window.__SEARCH_INDEX__={SEARCH_JSON};</script>
<script>{JS}</script>
</body>
</html>
'''


# --------------------------------------------------------------------------- #
# content renderers
# --------------------------------------------------------------------------- #
def field_list(doc: dict, params: list[dict], href_for) -> str:
    rows = []
    pdoc = doc.get("params") or {}
    if params:
        items = []
        for p in params:
            desc = pdoc.get(p["name"], "")
            tail = f' – {inline(desc, href_for)}' if desc else ""
            items.append(
                f'<li><p><span class="param-name">{esc(p["name"])}</span> '
                f'(<span class="param-type">{esc(p["type"])}</span>){tail}</p></li>'
            )
        rows.append(("参数", f'<ul>{"".join(items)}</ul>'))
    if doc.get("returns"):
        rows.append(("返回", f'<p>{inline(doc["returns"], href_for)}</p>'))
    throws = doc.get("throws") or {}
    if throws:
        items = [f'<li><p><span class="exc">{esc(k)}</span> – {inline(v, href_for)}</p></li>'
                 for k, v in throws.items()]
        rows.append(("抛出", f'<ul>{"".join(items)}</ul>'))
    if not rows:
        return ""
    out = ['<dl class="field-list simple">']
    for label, body in rows:
        out.append(f'<dt>{label}</dt><dd>{body}</dd>')
    out.append("</dl>")
    return "".join(out)


def prose(doc: dict, href_for) -> str:
    parts = []
    if doc.get("summary"):
        parts.append(f'<p class="summary">{inline(doc["summary"], href_for)}</p>')
    for para in doc.get("body") or []:
        if para.strip():
            parts.append(f"<p>{inline(para, href_for)}</p>")
    if doc.get("example"):
        code = doc["example"].strip("\n")
        parts.append('<div class="example"><div class="example-label">示例</div>'
                     f'<pre class="code"><code>{highlight_java(code)}</code></pre></div>')
    for note in doc.get("notes") or []:
        if note.strip():
            parts.append('<div class="admonition note"><p class="admonition-title">注意</p>'
                         f"<p>{inline(note, href_for)}</p></div>")
    see = [s for s in (doc.get("see") or []) if s.strip()]
    if see:
        links = []
        for target in see:
            href = href_for(target)
            links.append(f'<a href="{href}">{esc(target)}</a>' if href
                         else f"<code>{esc(target)}</code>")
        parts.append(f'<p class="seealso">参见：{", ".join(links)}</p>')
    return "".join(parts)


def signature_html(member: dict) -> str:
    name = member["name"]
    ret = member["return_type"]
    generic = ""
    if ret.startswith("<"):
        depth = 0
        for i, ch in enumerate(ret):
            if ch == "<":
                depth += 1
            elif ch == ">":
                depth -= 1
                if depth == 0:
                    generic = ret[:i + 1]
                    ret = ret[i + 1:].strip()
                    break
    head = f'<span class="pre">{esc(generic)} </span>' if generic else ""
    head += f'<span class="sig-name descname">{esc(name)}</span>'
    args = [f'<em class="sig-param">{esc(p["type"])} {esc(p["name"])}</em>'
            for p in member["params"]]
    sig = (f'{head}<span class="sig-paren">(</span>{", ".join(args)}'
           '<span class="sig-paren">)</span>')
    if member["kind"] == "method" and ret and ret != "void":
        sig += (f' <span class="sig-return">→</span> '
                f'<span class="sig-return-typehint">{esc(ret)}</span>')
    return sig


def render_type_block(t: dict, docs: dict, href_for) -> str:
    qualified = t["qualified"]
    doc = docs["types"].get(qualified, {})
    anchor = anchor_type(qualified)

    mods = " ".join(m for m in t["modifiers"] if m != "public")
    head = f'class {esc(t["simple_qualified"])}'
    if t["kind"] != "class":
        head = f'{t["kind"]} {esc(t["simple_qualified"])}'
    if t.get("generic"):
        head += esc(t["generic"])
    if t.get("extends"):
        head += f' <span class="property">extends</span> {esc(t["extends"])}'
    if t.get("implements"):
        head += (' <span class="property">implements</span> '
                 + esc(", ".join(t["implements"])))
    if mods:
        head = f'<span class="property">{esc(mods)}</span> ' + head

    members = visible_members(t)
    constants = [m for m in members if m["kind"] == "enum_constant"]
    constructors = [m for m in members if m["kind"] == "constructor"]
    methods = [m for m in members if m["kind"] == "method"]
    fields = [m for m in members if m["kind"] == "field"]

    out = [f'<dl class="py class" id="{anchor}">',
           f'<dt class="sig sig-object py">{head}'
           f'<a class="headerlink" href="#{anchor}" title="永久链接">¶</a></dt>',
           "<dd>", prose(doc, href_for)]

    counters: dict[str, int] = {}
    for group, title in ((constants, "枚举成员"), (constructors, "构造器"),
                         (methods, "方法"), (fields, "字段")):
        if not group:
            continue
        out.append(f'<p class="rubric">{title}</p>')
        for m in group:
            name = m["name"]
            idx = counters.get(name, 0)
            counters[name] = idx + 1
            m_anchor = anchor_member(qualified, name, idx)
            mdoc = docs["members"].get((qualified, m["key"]), {})
            flags = []
            if m.get("static"):
                flags.append("static")
            if m.get("abstract"):
                flags.append("abstract")
            if (m.get("javadoc") or {}).get("deprecated"):
                flags.append("deprecated")
            flag_html = "".join(f'<span class="py-flag">{f}</span>' for f in flags)

            body = prose(mdoc, href_for) if mdoc else ""
            if not body:
                jd = (m.get("javadoc") or {}).get("text", "")
                body = (f"<p>{inline(jd, href_for)}</p>" if jd
                        else '<p class="undocumented">（未提供说明）</p>')
            body += field_list(mdoc, m["params"], href_for)

            out.append(f'<dl class="py {m["kind"]}" id="{m_anchor}">')
            out.append(f'<dt class="sig sig-object py">{signature_html(m)}{flag_html}'
                       f'<a class="headerlink" href="#{m_anchor}" title="永久链接">¶</a></dt>')
            out.append(f"<dd>{body}</dd></dl>")
    out.append("</dd></dl>")
    return "".join(out)


def relbar(prev_link, next_link) -> str:
    left = (f'<a class="prev" href="{prev_link[1]}">{esc(prev_link[0])}</a>'
            if prev_link else "<span></span>")
    right = (f'<a class="next" href="{next_link[1]}">{esc(next_link[0])}</a>'
             if next_link else "<span></span>")
    return f'<div class="relbar">{left}<span class="spacer"></span>{right}</div>'


def render_module_page(module: dict, docs: dict, href_for, crumbs, prev_next) -> str:
    name = module["name"]
    intro = docs["intros"].get(name) or [MODULE_SUMMARY.get(name, "")]
    types = module["types"]
    public = [t for t in types if "public" in t["modifiers"]]
    internal = [t for t in types if "public" not in t["modifiers"]]

    toc = [(t["simple_qualified"], anchor_type(t["qualified"])) for t in public]
    body = [f'<h1 id="{module_anchor(name)}">模块 <code>{esc(name)}</code>'
            f'<a class="headerlink" href="#{module_anchor(name)}" '
            'title="永久链接">¶</a></h1>',
            f'<p class="welcome">{len(public)} 个公开类型'
            + (f" · {len(internal)} 个包内可见类型" if internal else "") + "</p>"]
    for para in intro:
        if para.strip():
            body.append(f"<p>{inline(para, href_for)}</p>")

    body.append('<h2 id="overview">类型概览</h2>')
    body.append('<table class="autosummary"><thead><tr><th>类型</th><th>种类</th>'
                "<th>说明</th></tr></thead><tbody>")
    for t in public:
        doc = docs["types"].get(t["qualified"], {})
        summary = doc.get("summary") or (t.get("javadoc") or {}).get("text", "").split("\n")[0]
        body.append(
            f'<tr><td><a href="#{anchor_type(t["qualified"])}">'
            f'<code>{esc(t["simple_qualified"])}</code></a></td>'
            f'<td>{esc(t["kind"])}</td><td>{inline(summary, href_for)}</td></tr>'
        )
    body.append("</tbody></table>")

    for t in public:
        body.append(render_type_block(t, docs, href_for))

    if internal:
        body.append('<h2 id="internal">包内可见类型</h2>')
        body.append('<p class="welcome">以下类型没有 <code>public</code> 修饰符，'
                    "但出现在公开签名中，因此一并列出。它们不是稳定 API，跨版本可能变化。</p>")
        for t in internal:
            body.append(render_type_block(t, docs, href_for))

    body.append(prev_next)
    return page(f"{name} — {TITLE}", crumbs, "".join(body),
                active=name, toc=toc, root="../")


def render_index(modules: list[dict], stats: dict) -> str:
    chapters = [
        ("API 参考", "按模块组织的完整 API：类型、方法、参数、返回值与抛出条件", "api/index.html"),
        ("快速开始", "十行代码完成一次前向与反向传播", "#quickstart"),
        ("安装", "Maven 坐标、运行参数与原生库加载顺序", "#install"),
        ("经典机器学习", "估计器、指标、数据切分与交叉验证", "api/basalt_ml.html"),
        ("神经网络", "手写层、序列模型、注意力与序列训练", "api/basalt_nn.html"),
        ("eager 张量", "动态张量、Module 组合与自动求导", "api/basalt_torch.html"),
        ("原生内核", "指令队列、计算图、内存所有权与失败回退", "api/basalt_nativebridge.html"),
        ("持久化", ".bsl v2 容器格式、CRC32C 校验与往返一致性", "api/basalt_io.html"),
    ]
    resources = [
        ("Basalt 开发者指南", "构建、验证与基准脚本", "api/index.html"),
        ("架构与基准报告", "原生 GEMM、自动求导与实测数据",
         "report/Basalt_ARCHITECTURE_AND_BENCHMARK.html"),
        ("与参考实现的对比", "同一份 BasaltD 数据下的十项对比", "report/Basalt_VS_REFERENCE.html"),
        ("GitHub 仓库", "源码、发布产物与问题追踪", REPO),
    ]
    install = f"""<dependency>
    <groupId>io.github.9178nekomc</groupId>
    <artifactId>basalt</artifactId>
    <version>{VERSION}</version>
</dependency>"""
    quick = """try (Basalt jsl = Basalt.create()) {
    Tensor x = basalt.randn(32, 8);
    Tensor y = basalt.randn(32, 1);
    Linear hidden = basalt.linear(8, 16);
    Linear output = basalt.linear(16, 1);
    Tensor loss = output.forward(hidden.forward(x).relu()).mse(y);
    loss.backward();
}"""
    build = """mvn clean install        # 编译 + JUnit 5 + 标准 jar + Fat Jar + 源码包
./verify.sh              # 编译期拒绝用例 + legacy 套件 + Rust 单元测试
./benchmark.sh           # 基准，输出 build/bench/*.csv
./native-build.sh        # 只构建 Rust 内核到 build/native/"""

    chapter_items = "".join(
        f'<li><p><a class="biglink" href="{href}">{esc(label)}</a>'
        f'<span class="linkdescr">{esc(desc)}</span></p></li>'
        for label, desc, href in chapters
    )
    resource_items = "".join(
        f'<li><p><a class="biglink" href="{href}">{esc(label)}</a>'
        f'<span class="linkdescr">{esc(desc)}</span></p></li>'
        for label, desc, href in resources
    )
    module_rows = "".join(
        f'<tr><td><a href="{page_name(m["name"])}"><code>{esc(m["name"])}</code></a></td>'
        f'<td class="num">{sum(1 for t in m["types"] if "public" in t["modifiers"])}</td>'
        f'<td>{inline(MODULE_SUMMARY.get(m["name"], ""), None)}</td></tr>'
        for m in modules
    )

    body = f'''<h1>{esc(TITLE)}</h1>
<p class="welcome">欢迎！这里是 Basalt {VERSION} 的官方文档。Basalt 是一个 Java 调度、Rust 托管底层的
机器学习运行时，覆盖 eager 张量、自动求导、原生 CNN/Transformer 算子与经典统计学习估计器。</p>

<h2 id="chapters">文档章节：</h2>
<ul class="chapters">{chapter_items}</ul>

<h2 id="resources">其他资源：</h2>
<ul class="chapters">{resource_items}</ul>

<h2 id="install">安装</h2>
<pre class="code"><code>{highlight_java(install)}</code></pre>
<p>运行期需要开启 Vector API 孵化模块：<code>--add-modules jdk.incubator.vector</code>。
原生库已随 <code>basalt-{VERSION}-all.jar</code> 打包；加载失败时经典模型自动回退纯 Java 路径。</p>

<h2 id="quickstart">快速开始</h2>
<pre class="code"><code>{highlight_java(quick)}</code></pre>

<h2 id="layers">三层 API 边界</h2>
<table><thead><tr><th>层级</th><th>入口</th><th>适用场景</th></tr></thead><tbody>
<tr><td>整体</td><td><code>Basalt</code> 门面、<code>Estimator.fit/predict</code></td>
<td>一行完成训练、预测、交叉验证</td></tr>
<tr><td>模块</td><td><code>RandomForest.builder()</code>、<code>HistogramGradientBoosting.builder()</code></td>
<td>按模块调参、替换求解器、控制线程</td></tr>
<tr><td>原子</td><td><code>NativeContext</code> + <code>CommandQueueBuilder</code></td>
<td>手写算子序列、自定义损失与优化器</td></tr>
</tbody></table>

<h2 id="modules">模块索引</h2>
<table><thead><tr><th>模块</th><th class="num">公开类型</th><th>职责</th></tr></thead>
<tbody>{module_rows}</tbody></table>

<h2 id="conventions">文档约定</h2>
<ul class="simple">
<li><p><strong>签名</strong>：由 <code>tools/api_dump.py</code> 扫描源码生成，与编译产物一一对应；
<code>→</code> 之后是返回类型。</p></li>
<li><p><strong>参数</strong>：逐个列出名称、类型与语义。标注"默认值"的，取自字段初始值而非推测。</p></li>
<li><p><strong>抛出</strong>：只列出源码中真实存在的 <code>throw</code> 或显式校验分支。</p></li>
<li><p><strong>包内可见类型</strong>：没有 <code>public</code> 修饰符但出现在公开签名中的类型，
归入各模块页末尾，不作为稳定 API 承诺。</p></li>
<li><p><strong>精度分层</strong>：原生张量引擎为 <code>f32</code>，经典估计器保持 <code>f64</code>。</p></li>
<li><p><strong>规模</strong>：{stats['public_types']} 个公开类型、{stats['members']} 个公开成员，
其中 {stats['documented_members']} 个成员附有逐参数说明。</p></li>
</ul>

<h2 id="build">构建与验证</h2>
<pre class="code"><code>{highlight_java(build)}</code></pre>

<div class="relbar"><span></span><span class="spacer"></span>
<a class="next" href="api/index.html">API 参考</a></div>
<footer class="doc-footer">
<p>Basalt {VERSION} · 文档由 <code>tools/build_api_docs.py</code> 生成于 {stats['generated']}。
签名来源 <code>build/api/api.json</code>，说明来源 <code>tools/docs_data/</code>。</p>
<p><a href="{REPO}" rel="noopener">{REPO}</a></p>
</footer>'''
    return page(TITLE, [("Basalt", "index.html"), (f"{VERSION} 文档", None)],
                body, active="index", root="")


def render_readme_page(stats: dict) -> str:
    """Render README.md as a styled page so static hosts never serve raw text."""
    source = open("README.md", encoding="utf-8").read()
    body_html = render_markdown(source)
    # rewrite repository-relative targets to the copies that live inside docs/
    body_html = body_html.replace('href="docs/index.html"', 'href="index.html"')
    body_html = re.sub(r'href="report/([\w.-]+)\.md"', r'href="report/\1.html"', body_html)
    body_html = re.sub(r'href="(target/[^"]+)"',
                       r'href="https://github.com/9178nekomc/basalt/raw/main/\1"',
                       body_html)
    body = (
        '<p class="welcome">本页由 <code>README.md</code> 自动渲染，内容与仓库根目录的 '
        f'<a href="{REPO}/blob/main/README.md" rel="noopener">README</a> 一致。'
        f'生成于 {stats["generated"]}。</p>'
        + body_html +
        '<div class="relbar"><a class="prev" href="index.html">Basalt 3.9.0 文档</a>'
        '<span class="spacer"></span><a class="next" href="api/index.html">模块索引</a></div>'
    )
    return page(f"README — {TITLE}",
                [("Basalt", "index.html"), (VERSION, None), ("README", None)],
                body, active="readme", root="")


REPORT_LABELS = {
    "Basalt_ARCHITECTURE_AND_BENCHMARK.html": "架构与基准测试报告",
    "Basalt_VS_REFERENCE.html": "与参考实现的对比",
}

# The two report files are standalone documents generated by report/make_report.py.
# They ship their own palette and have no navigation, so a copy opened from the site
# used to be a one-way trip into a light-only page.  This shell adds a way back, the
# same theme as the docs, and the cross-document fade.
REPORT_HEAD = """<script>(function(){var r=document.documentElement,K='basalt-docs-theme';
function res(p){return p==='auto'?(window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'):p;}
function set(p){r.setAttribute('data-theme',res(p));r.setAttribute('data-theme-pref',p);try{localStorage.setItem(K,p);}catch(e){}}
var p='auto',q=/[?&]theme=(light|dark|auto)/.exec(location.search);
try{p=(q&&q[1])||localStorage.getItem(K)||'auto';}catch(e){}
set(p);
document.addEventListener('DOMContentLoaded',function(){
  var b=document.getElementById('jslThemeBtn');
  if(!b){return;}
  b.addEventListener('click',function(){
    var next=r.getAttribute('data-theme')==='dark'?'light':'dark';
    if(document.startViewTransition&&!window.matchMedia('(prefers-reduced-motion: reduce)').matches){
      document.startViewTransition(function(){set(next);});
    }else{set(next);}
  });
});})();</script>
<style>
@view-transition{navigation:auto}
:root{--jsl-bar-bg:rgba(255,255,255,.88);--jsl-bar-ink:#233040;
  --jsl-bar-line:rgba(15,23,42,.10);--jsl-bar-hover:rgba(15,23,42,.06);
  --jsl-bar-strong:#0b1420;--jsl-bar-edge:#cfd8e3}
html[data-theme="dark"]{--jsl-bar-bg:rgba(26,29,33,.88);--jsl-bar-ink:#dde3ea;
  --jsl-bar-line:rgba(255,255,255,.09);--jsl-bar-hover:rgba(255,255,255,.08);
  --jsl-bar-strong:#fff;--jsl-bar-edge:#3d454f}
.bsl-bar{position:fixed;top:0;left:0;right:auto;width:100%;max-width:100vw;box-sizing:border-box;
  height:44px;z-index:99;display:flex;
  align-items:center;gap:12px;padding:0 16px;background:var(--jsl-bar-bg);color:var(--jsl-bar-ink);
  font:13px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans SC","Microsoft YaHei",sans-serif;
  box-shadow:0 1px 0 var(--jsl-bar-line);view-transition-name:jsl-bar;
  -webkit-backdrop-filter:saturate(180%) blur(14px);backdrop-filter:saturate(180%) blur(14px);
  transition:background-color .25s ease,color .25s ease,box-shadow .25s ease}
.bsl-bar a{color:inherit;text-decoration:none;padding:5px 10px;border-radius:6px;
  transition:background-color .18s ease,color .18s ease}
.bsl-bar a:hover{background:var(--jsl-bar-hover);color:var(--jsl-bar-strong)}
.bsl-bar .sep{opacity:.32;user-select:none}
.bsl-bar .t{color:inherit;opacity:.68;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bsl-bar .sp{flex:1}
.bsl-bar button{background:transparent;border:1px solid var(--jsl-bar-edge);color:inherit;cursor:pointer;
  border-radius:6px;padding:4px 10px;font-size:12px;font-family:inherit;
  transition:border-color .18s ease,color .18s ease,background-color .18s ease}
.bsl-bar button:hover{border-color:var(--jsl-bar-strong);color:var(--jsl-bar-strong);
  background:var(--jsl-bar-hover)}
.bsl-bar :focus-visible{outline:2px solid var(--jsl-bar-strong);outline-offset:2px}
body>main{padding-top:96px}
/* wide tables have to scroll on their own, otherwise the document outgrows the
   viewport and every fixed element stretches with it */
.tbl-scroll{max-width:100%;overflow-x:auto;-webkit-overflow-scrolling:touch;margin:18px 0}
.tbl-scroll>table{margin:0}
nav.toc ol{list-style:none;padding-left:0;margin:0}
nav.toc li{list-style:none}
nav.toc li::marker{content:""}
body>main{animation:jsl-rise .34s cubic-bezier(.22,.61,.36,1) both}
@keyframes jsl-rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
@media (prefers-reduced-motion: reduce){body>main{animation:none}}
@media (max-width:720px){body>main{padding-top:84px}.bsl-bar .t,.bsl-bar .sep{display:none}
  code{overflow-wrap:anywhere}}
html[data-theme="dark"]{color-scheme:dark;--ink:#d8dde3;--muted:#93a0ad;--line:#2e343b;
  --panel:#20242a;--accent:#7ab7ff;--accent-soft:#1b2733;--code:#14181d}
::view-transition-old(jsl-bar){animation:none}
::view-transition-new(jsl-bar){animation:none}
html[data-theme="dark"] body{background:#1a1c1f;color:#d8dde3}
/* the report leaves <code> to inherit the surrounding colour, which turns muted
   grey on a grey pill whenever a snippet sits inside a caption.  Pin it. */
code{color:#374151}
html[data-theme="dark"] code{background:#272d35;color:#dfe4ea}
html[data-theme="dark"] pre{background:#14181d;color:#d8dde3}
html[data-theme="dark"] pre code{background:transparent}
html[data-theme="dark"] blockquote{color:#cfd6de;background:#1b2733}
html[data-theme="dark"] tr:nth-child(even) td{background:#1e2126}
html[data-theme="dark"] th{background:#20242a}
html[data-theme="dark"] figure img{background:#fff;border-color:#39414b}
html[data-theme="dark"] nav.toc{background:#20242a}
</style>
"""


def wrap_report(html: str, label: str) -> str:
    """Add the docs bar, theme inheritance and cross-document fade to a report."""
    bar = (
        '<div class="jsl-bar">'
        '<a href="../index.html">&#8592; 返回文档</a>'
        '<span class="sep">/</span>'
        '<a href="../index.html#resources">其他资源</a>'
        f'<span class="t">{esc(label)}</span>'
        '<span class="sp"></span>'
        '<button id="jslThemeBtn" type="button" title="切换主题">&#9681; 主题</button>'
        "</div>"
    )
    html = html.replace("</head>", REPORT_HEAD + "</head>", 1)
    html = html.replace("<body>", "<body>\n" + bar, 1)
    html = re.sub(r"<table>(.*?)</table>",
                  r'<div class="tbl-scroll"><table>\1</table></div>',
                  html, flags=re.S)
    return html


def render_module_index(modules: list[dict]) -> str:
    rows = "".join(
        f'<tr><td><a href="{os.path.basename(page_name(m["name"]))}">'
        f'<code>{esc(m["name"])}</code></a></td>'
        f'<td class="num">{sum(1 for t in m["types"] if "public" in t["modifiers"])}</td>'
        f'<td>{inline(MODULE_SUMMARY.get(m["name"], ""), None)}</td></tr>'
        for m in modules
    )
    body = f'''<h1>模块索引</h1>
<p class="welcome">Basalt 按包组织 API。每个模块一页，页内按类型分组，类型下依次列出构造器、方法与字段。</p>
<table><thead><tr><th>模块</th><th class="num">公开类型</th><th>职责</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2 id="order">阅读顺序建议</h2>
<ul class="simple">
<li><p>先读 <a href="basalt.html"><code>jsl</code></a> 门面，确认创建上下文与取模块的方式。</p></li>
<li><p>张量训练走 <a href="basalt_torch.html"><code>basalt.torch</code></a>；经典模型走
<a href="basalt_ml.html"><code>basalt.ml</code></a>。</p></li>
<li><p>需要手写算子或排查生命周期时读
<a href="basalt_nativebridge.html"><code>basalt.nativebridge</code></a>。</p></li>
</ul>
<div class="relbar"><a class="prev" href="../index.html">Basalt {VERSION} 文档</a>
<span class="spacer"></span><a class="next" href="basalt.html">jsl</a></div>'''
    return page(f"模块索引 — {TITLE}",
                [("Basalt", "../index.html"), (VERSION, None), ("模块索引", None)],
                body, active="modindex", root="../")


def render_genindex(modules: list[dict]) -> str:
    entries: dict[str, list[tuple[str, str]]] = {}
    for module in modules:
        for t in module["types"]:
            if "public" not in t["modifiers"]:
                continue
            letter = t["name"][0].upper()
            entries.setdefault(letter, []).append(
                (t["simple_qualified"],
                 f'{page_name(module["name"])}#{anchor_type(t["qualified"])}')
            )
    letters = sorted(entries)
    jump = " ".join(f'<a href="#{l}">{l}</a>' for l in letters)
    sections = []
    for letter in letters:
        sections.append(f'<h2 id="{letter}">{letter}</h2><ul class="genindex-section">')
        for simple, href in sorted(entries[letter]):
            sections.append(f'<li><a href="{href}">{esc(simple)}</a></li>')
        sections.append("</ul>")
    body = (f'<h1>总索引</h1>'
            f'<p class="welcome">按类型名首字母排列，共 {sum(len(v) for v in entries.values())} '
            "个公开类型。方法请在各模块页内查找，或用顶部快速搜索（Ctrl+K）。</p>"
            f'<div class="genindex-jumpbox">跳转：{jump}</div>' + "".join(sections) +
            '<div class="relbar"><a class="prev" href="index.html">Basalt 3.9.0 文档</a>'
            '<span class="spacer"></span><a class="next" href="api/index.html">模块索引</a></div>')
    return page(f"总索引 — {TITLE}",
                [("Basalt", "index.html"), (VERSION, None), ("总索引", None)],
                body, active="genindex", root="")


def build_search_index(modules: list[dict], docs: dict) -> list[list[str]]:
    index: list[list[str]] = []
    for module in modules:
        page_file = page_name(module["name"])
        for t in module["types"]:
            if "public" not in t["modifiers"]:
                continue
            doc = docs["types"].get(t["qualified"], {})
            index.append([t["qualified"], t["simple_qualified"], module["name"],
                          (doc.get("summary") or "")[:100],
                          f'{page_file}#{anchor_type(t["qualified"])}', "type"])
            counters: dict[str, int] = {}
            for m in visible_members(t):
                if m["kind"] == "field":
                    continue
                idx = counters.get(m["name"], 0)
                counters[m["name"]] = idx + 1
                mdoc = docs["members"].get((t["qualified"], m["key"]), {})
                index.append([
                    f'{t["qualified"]}.{m["name"]}',
                    f'{t["simple_qualified"]}.{m["name"]}()',
                    module["name"], (mdoc.get("summary") or "")[:100],
                    f'{page_file}#{anchor_member(t["qualified"], m["name"], idx)}', "method",
                ])
    return index


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="build/api/api.json")
    ap.add_argument("--docs", default="tools/docs_data/g*.json")
    ap.add_argument("--out", default="docs")
    args = ap.parse_args()

    set_java_highlighter(highlight_java)
    api = load_api(args.api)
    docs = load_docs(args.docs)
    modules = sorted(api["modules"],
                     key=lambda m: (MODULE_ORDER.index(m["name"])
                                    if m["name"] in MODULE_ORDER else 99, m["name"]))

    anchor_map: dict[str, str] = {}
    qualified_of: dict[str, str] = {}
    for module in modules:
        # cross references are rendered inside api/ pages, so the basename is enough
        page_file = os.path.basename(page_name(module["name"]))
        for t in module["types"]:
            href = f"{page_file}#{anchor_type(t['qualified'])}"
            anchor_map[t["qualified"]] = href
            anchor_map.setdefault(t["simple_qualified"], href)
            qualified_of[t["qualified"]] = t["qualified"]
            qualified_of.setdefault(t["simple_qualified"], t["qualified"])

    def href_for(target: str) -> str | None:
        """Resolve `Type` or `Type#member` to a page (and member) link."""
        base, _, member = target.partition("#")
        href = anchor_map.get(base)
        if not href:
            return None
        if not member:
            return href
        page_file = href.split("#", 1)[0]
        qualified = qualified_of.get(base, base)
        member_name = member.split("(", 1)[0].strip()
        return f"{page_file}#m-{slug(qualified + '.' + member_name)}"

    public_types = [t for m in modules for t in m["types"] if "public" in t["modifiers"]]
    documented = sum(1 for t in public_types for m in visible_members(t)
                     if (t["qualified"], m["key"]) in docs["members"])
    stats = {
        "public_types": len(public_types),
        "members": sum(len(visible_members(t)) for t in public_types),
        "documented_members": documented,
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    global NAV_MODULES
    NAV_MODULES = [
        (m["name"],
         page_name(m["name"]),
         [(t["simple_qualified"], anchor_type(t["qualified"]))
          for t in m["types"] if "public" in t["modifiers"]])
        for m in modules
    ]

    out = args.out
    api_dir = os.path.join(out, "api")
    os.makedirs(api_dir, exist_ok=True)

    # Assets are inlined into every page: a sibling `_static/` directory is
    # dropped by hosts that ignore underscore-prefixed paths (Jekyll, the
    # WorkBuddy publish preview), which left the pages unstyled.
    global SEARCH_JSON
    search = build_search_index(modules, docs)
    SEARCH_JSON = json.dumps(search, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(render_index(modules, stats))

    for i, module in enumerate(modules):
        prev_mod = modules[i - 1] if i > 0 else None
        next_mod = modules[i + 1] if i + 1 < len(modules) else None
        prev_link = ((prev_mod["name"], os.path.basename(page_name(prev_mod["name"])))
                     if prev_mod else ("Basalt 3.9.0 文档", "../index.html"))
        next_link = ((next_mod["name"], os.path.basename(page_name(next_mod["name"])))
                     if next_mod else ("模块索引", "index.html"))
        crumbs = [("Basalt", "../index.html"), (VERSION, None),
                  ("API 参考", "index.html"), (module["name"], None)]
        doc_html = render_module_page(module, docs, href_for, crumbs,
                                      relbar(prev_link, next_link))
        with open(os.path.join(api_dir, os.path.basename(page_name(module["name"]))),
                  "w", encoding="utf-8") as fh:
            fh.write(doc_html)

    with open(os.path.join(out, "readme.html"), "w", encoding="utf-8") as fh:
        fh.write(render_readme_page(stats))

    # The report HTMLs are self-contained (base64 images) but live outside docs/,
    # so static hosts that publish only docs/ would 404 on them.  Copy the built
    # snapshots next to the documentation.
    report_out = os.path.join(out, "report")
    os.makedirs(report_out, exist_ok=True)
    for path in sorted(glob.glob("report/*.html")):
        name = os.path.basename(path)
        label = REPORT_LABELS.get(name, name)
        with open(path, encoding="utf-8") as fh:
            payload = wrap_report(fh.read(), label)
        with open(os.path.join(report_out, name), "w", encoding="utf-8") as fh:
            fh.write(payload)

    with open(os.path.join(api_dir, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(render_module_index(modules))

    with open(os.path.join(out, "genindex.html"), "w", encoding="utf-8") as fh:
        fh.write(render_genindex(modules))

    total = 0
    for base, _dirs, files in os.walk(out):
        for name in files:
            total += os.path.getsize(os.path.join(base, name))

    print(f"output      : {out}/")
    print(f"pages       : {len(modules) + 4} html (+ readme)")
    print(f"total size  : {total / 1024:.1f} KiB")
    print(f"types       : {stats['public_types']} public")
    print(f"members     : {stats['members']} public, {stats['documented_members']} documented")
    print(f"search index: {len(search)} entries")
    missing = [(t["qualified"], m["key"]) for t in public_types
               for m in visible_members(t)
               if (t["qualified"], m["key"]) not in docs["members"]]
    if missing:
        print(f"UNDOCUMENTED ({len(missing)}): {missing[:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
