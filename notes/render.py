#!/usr/bin/env python3
"""Render the notes in this directory to self-contained HTML under docs/notes/.

    python3 notes/render.py

# Why this exists

raw.githack.com serves the repository's raw files with a corrected Content-Type.
That makes `docs/index.html` a working page — and makes a `.md` file a wall of
plain text, because Markdown is a source format and nothing renders it. So the
notes are converted to HTML here and the result is committed, the same way
`minesweeper.wasm` and `standalone.html` are.

# Why the converter is hand-written

Not because a CDN would be wrong — the pages are served from one, and referencing
another is fine. (What `standalone.html` guarantees is that the *game* needs no
server, which is a different claim.) Three ordinary reasons:

* `nbconvert` renders notebooks and not Markdown, so the two `.md` notes would
  need a second converter anyway, and the two halves of the same set of notes
  would not look alike.
* Regenerating the site then needs a pip install. `wasm/build.sh` asks for a Rust
  toolchain and nothing else; this keeps that true.
* 350 KB a page against 15-35 KB.

The cost is real and worth stating: this is a *partial* Markdown implementation.
It covers exactly what these notes use — headings, paragraphs, fenced code,
inline code, bold, italic, links, tables, lists, block quotes and rules — and
nothing else. If a note starts needing something it does not support, teach it
that one thing rather than assuming it works.
"""

import hashlib
import html
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
OUT = HERE.parent / "docs" / "notes"

STYLE = """
:root {
  --ink: #23272e; --dim: #6b7280; --rule: #e3e6ea; --bg: #ffffff;
  --code-bg: #f6f7f9; --accent: #2b5fa8; --out: #f9fafb; --guess: #6a4fb6;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  -webkit-text-size-adjust: 100%;
}
main { max-width: 46rem; margin: 0 auto; padding: 2.5rem 1.25rem 5rem; }
h1, h2, h3, h4 { line-height: 1.25; margin: 2.2em 0 0.6em; font-weight: 650; }
h1 { font-size: 1.9rem; margin-top: 0; letter-spacing: -0.01em; }
h2 { font-size: 1.35rem; padding-top: 0.6em; border-top: 1px solid var(--rule); }
/* A rule immediately before a heading would draw the same line twice. */
hr + h2 { border-top: 0; padding-top: 0; margin-top: 1.4em; }
h3 { font-size: 1.1rem; }
h4 { font-size: 1rem; color: var(--dim); }
p, ul, ol, blockquote, table, pre { margin: 0 0 1.1em; }
a { color: var(--accent); }
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.87em; background: var(--code-bg); padding: 0.12em 0.35em; border-radius: 3px;
}
pre {
  background: var(--code-bg); border: 1px solid var(--rule); border-radius: 6px;
  padding: 0.85em 1em; overflow-x: auto; line-height: 1.45;
}
pre code { background: none; padding: 0; font-size: 0.82rem; }
blockquote {
  margin-left: 0; padding: 0.1em 0 0.1em 1em;
  border-left: 3px solid var(--accent); color: var(--ink);
}
table { border-collapse: collapse; width: 100%; font-size: 0.93rem; }
th, td { text-align: left; padding: 0.45em 0.7em; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 650; border-bottom: 2px solid var(--rule); }
hr { border: 0; border-top: 1px solid var(--rule); margin: 2.2em 0; }
li { margin: 0.25em 0; }
em { color: inherit; }

/* Notebook rendering */
.cell { margin: 0 0 1.4em; }
.cell.code { border-left: 3px solid var(--rule); padding-left: 0.9em; }
.cell.code pre.src { background: #fbfcfd; }
.cell .label {
  font: 600 0.68rem/1 ui-monospace, monospace; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--dim); margin: 0.9em 0 0.4em;
}
pre.out {
  background: var(--out); border-style: dashed; color: #333; font-size: 0.78rem;
}
.nav { font-size: 0.9rem; color: var(--dim); margin: 0 0 2.2em; }
.nav a { margin-right: 1.1em; }
footer {
  margin-top: 3.5em; padding-top: 1.2em; border-top: 1px solid var(--rule);
  font-size: 0.8rem; color: var(--dim);
}
@media (prefers-color-scheme: dark) {
  :root {
    --ink: #d9dde3; --dim: #8b93a1; --rule: #2c313a; --bg: #14171c;
    --code-bg: #1b1f26; --accent: #7aa7e8; --out: #171b21; --guess: #a88fe0;
  }
  pre.out { color: #c3c9d2; }
  .cell.code pre.src { background: #191d24; }
}
"""

# ------------------------------------------------------------------ inline


def inline(text: str) -> str:
    """Inline Markdown on one already-unescaped string."""
    spans: list[str] = []

    def stash(match):
        spans.append(html.escape(match.group(1)))
        return f"\x00{len(spans) - 1}\x00"

    # Code spans first, so nothing below rewrites their contents.
    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: f'<a href="{link(m.group(2))}">{m.group(1)}</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{spans[int(m.group(1))]}</code>", text)


def link(target: str) -> str:
    """Point a link at the rendered page rather than the source file."""
    if target.startswith(("http://", "https://", "#")):
        return html.escape(target)
    target = re.sub(r"\.ipynb$", "-notebook.html", target)
    target = re.sub(r"(?<!README)\.md$", ".html", target)
    return html.escape(target)


def slug(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", re.sub(r"[`*]", "", heading).lower()).strip("-")


# ------------------------------------------------------------------- blocks


def convert(source: str) -> str:
    """Markdown to HTML, for the subset these notes use."""
    lines = source.split("\n")
    out: list[str] = []
    i = 0

    def paragraph(buffer):
        if buffer:
            out.append(f"<p>{inline(' '.join(buffer))}</p>")
        buffer.clear()

    buffer: list[str] = []
    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):
            paragraph(buffer)
            i += 1
            block = []
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(block))}</code></pre>")
            continue

        if match := re.match(r"^(#{1,6})\s+(.*)$", line):
            paragraph(buffer)
            level, title = len(match.group(1)), match.group(2)
            out.append(f'<h{level} id="{slug(title)}">{inline(title)}</h{level}>')
            i += 1
            continue

        if re.match(r"^(-{3,}|\*{3,})\s*$", line):
            paragraph(buffer)
            out.append("<hr>")
            i += 1
            continue

        # A table: a header row, a separator of dashes and pipes, then rows.
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|?\s*$", lines[i + 1]):
            paragraph(buffer)
            def split(row):
                return [c.strip() for c in row.strip().strip("|").split("|")]
            header = split(line)
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(split(lines[i]))
                i += 1
            head = "".join(f"<th>{inline(c)}</th>" for c in header)
            body = "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>" for row in rows
            )
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
            continue

        if re.match(r"^\s*([-*]|\d+\.)\s+", line):
            paragraph(buffer)
            ordered = bool(re.match(r"^\s*\d+\.\s", line))
            items: list[list[str]] = []
            while i < len(lines) and (
                re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]) or (lines[i].startswith("  ") and lines[i].strip() and items)
            ):
                if re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
                    items.append([re.sub(r"^\s*([-*]|\d+\.)\s+", "", lines[i])])
                else:
                    items[-1].append(lines[i].strip())      # continuation line
                i += 1
                # A blank line inside a list is allowed if the list resumes after it.
                if i < len(lines) and not lines[i].strip():
                    ahead = i + 1
                    if ahead < len(lines) and re.match(r"^\s*([-*]|\d+\.)\s+", lines[ahead]):
                        i = ahead
                    elif ahead < len(lines) and lines[ahead].startswith("  ") and lines[ahead].strip():
                        items[-1].append("")
                        i = ahead
            tag = "ol" if ordered else "ul"
            rendered = "".join(f"<li>{paragraphs(item)}</li>" for item in items)
            out.append(f"<{tag}>{rendered}</{tag}>")
            continue

        if line.startswith(">"):
            paragraph(buffer)
            quoted = []
            while i < len(lines) and lines[i].startswith(">"):
                quoted.append(lines[i].lstrip(">").strip())
                i += 1
            out.append(f"<blockquote><p>{inline(' '.join(quoted))}</p></blockquote>")
            continue

        if not line.strip():
            paragraph(buffer)
        else:
            buffer.append(line.strip())
        i += 1

    paragraph(buffer)
    return "\n".join(out)


def paragraphs(parts: list[str]) -> str:
    """A list item's lines, with a blank line starting a new paragraph."""
    chunks, current = [], []
    for part in parts:
        if part:
            current.append(part)
        elif current:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    if len(chunks) <= 1:
        return inline(chunks[0]) if chunks else ""
    return "".join(f"<p>{inline(c)}</p>" for c in chunks)


# ---------------------------------------------------------------- notebooks


def text_of(value) -> str:
    return "".join(value) if isinstance(value, list) else (value or "")


def convert_notebook(raw: str) -> tuple[str, str]:
    """A committed .ipynb to HTML: markdown cells rendered, code and output verbatim."""
    notebook = json.loads(raw)
    title = "Notebook"
    out = []

    for cell in notebook["cells"]:
        source = text_of(cell["source"])
        if cell["cell_type"] == "markdown":
            if title == "Notebook":
                if heading := re.search(r"^#\s+(.*)$", source, re.M):
                    title = heading.group(1)
            out.append(f'<div class="cell md">{convert(source)}</div>')
            continue

        pieces = [
            '<p class="label">in</p>',
            f'<pre class="src"><code>{html.escape(source)}</code></pre>',
        ]
        printed = []
        for output in cell.get("outputs", []):
            kind = output.get("output_type")
            if kind == "stream":
                printed.append(text_of(output.get("text")))
            elif kind in ("execute_result", "display_data"):
                printed.append(text_of(output.get("data", {}).get("text/plain")))
            elif kind == "error":
                printed.append("\n".join(output.get("traceback", [])))
        body = "".join(printed).rstrip()
        if body:
            # Escape codes from a coloured traceback would print as mojibake.
            body = re.sub(r"\x1b\[[0-9;]*m", "", body)
            pieces.append('<p class="label">out</p>')
            pieces.append(f'<pre class="out"><code>{html.escape(body)}</code></pre>')
        out.append(f'<div class="cell code">{"".join(pieces)}</div>')

    return title, "\n".join(out)


# ------------------------------------------------------------------- output


def page(title: str, body: str, nav: str, stamp: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{STYLE}</style>
</head>
<body>
<main>
<nav class="nav">{nav}</nav>
{body}
<footer>Generated from the repository by <code>notes/render.py</code> · source {stamp}</footer>
</main>
</body>
</html>
"""


NAV = (
    '<a href="index.html">Notes</a>'
    '<a href="constraint-estimator.html">Exact solver</a>'
    '<a href="constraint-estimator-notebook.html">· notebook</a>'
    '<a href="neural-estimator.html">Neural</a>'
    '<a href="neural-estimator-notebook.html">· notebook</a>'
    '<a href="../index.html">Play</a>'
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []

    for source in sorted(HERE.glob("*.md")):
        raw = source.read_text(encoding="utf-8")
        heading = re.search(r"^#\s+(.*)$", raw, re.M)
        title = heading.group(1) if heading else source.stem
        stamp = hashlib.sha256(raw.encode()).hexdigest()[:8]
        name = "index.html" if source.stem == "README" else f"{source.stem}.html"
        (OUT / name).write_text(page(title, convert(raw), NAV, stamp), encoding="utf-8")
        written.append(name)

    for source in sorted(HERE.glob("*.ipynb")):
        raw = source.read_text(encoding="utf-8")
        title, body = convert_notebook(raw)
        stamp = hashlib.sha256(raw.encode()).hexdigest()[:8]
        name = f"{source.stem}-notebook.html"
        (OUT / name).write_text(page(title, body, NAV, stamp), encoding="utf-8")
        written.append(name)

    for name in written:
        print(f"docs/notes/{name}  ({(OUT / name).stat().st_size:,} bytes)")


if __name__ == "__main__":
    sys.exit(main())
