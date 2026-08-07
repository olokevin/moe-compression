#!/usr/bin/env python3
"""
Build a reveal.js Jupyter-notebook slideshow from ``midpoint_slides.md``.

The output ``midpoint_slideshow.ipynb`` is a markdown-only notebook where every
cell carries ``metadata.slideshow.slide_type = "slide"`` so the VSCode *Jupyter
Slide Show* extension (or ``nbconvert --to slides``) renders one reveal.js slide
per cell. Export the PDF from VSCode.

Slide boundaries are markdown headings in the source:
  * a single ``#`` heading      -> a centered *section divider* slide
  * a ``## Slide N: ...`` heading -> a *content* slide (the "Slide N:" prefix is
    dropped for a clean title)

Referenced figures (``figs/*.png`` and ``figs/*.svg``) are embedded as base64
data URIs so the notebook is fully self-contained and the exported PDF always
shows them.

Run from ``docs/presentation``:  ``python build_slides.py``
"""
from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
SRC_MD = HERE / "midpoint_slides.md"
NB_PATH = HERE / "midpoint_slideshow.ipynb"

TITLE_RE = re.compile(r"^Title$", re.IGNORECASE)
IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def slide_meta(slide_type: str = "slide") -> dict:
    return {"slideshow": {"slide_type": slide_type}}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_slides(md_text: str):
    """Split markdown into (kind, title, body) blocks.

    kind is "section" (single-# divider) or "content" (## Slide heading).
    Fenced code blocks are respected; horizontal rules are dropped.
    """
    lines = md_text.splitlines()
    slides = []
    cur = None

    def flush():
        nonlocal cur
        if cur is not None:
            blines = cur["body"]
            while blines and blines[-1].strip() in ("", "---"):
                blines.pop()
            slides.append((cur["kind"], cur["title"], "\n".join(blines).strip()))
            cur = None

    in_code = False
    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code = not in_code
            if cur is not None:
                cur["body"].append(line)
            continue

        if not in_code:
            m_slide = re.match(r"^##\s+Slide\s+\d+\s*:\s*(.*)$", line)
            m_section = re.match(r"^#\s+(?!#)(.*)$", line)
            if m_slide:
                flush()
                cur = {"kind": "content", "title": m_slide.group(1).strip(), "body": []}
                continue
            if m_section:
                flush()
                cur = {"kind": "section", "title": m_section.group(1).strip(), "body": []}
                continue
            if stripped == "---" and cur is None:
                continue

        if cur is not None:
            cur["body"].append(line)

    flush()
    return slides


# ---------------------------------------------------------------------------
# Image embedding
# ---------------------------------------------------------------------------
def data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/svg+xml" if path.suffix == ".svg" else "application/octet-stream"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def embed_images(body: str) -> str:
    """Replace ``![alt](figs/foo.png)`` with an embedded data URI."""
    def repl(m: re.Match) -> str:
        alt, src = m.group(1), m.group(2).strip()
        if src.startswith("data:") or src.startswith("http"):
            return m.group(0)
        img = (HERE / src).resolve()
        if not img.exists():
            print(f"  WARNING: missing image {src}")
            return m.group(0)
        return f"![{alt}]({data_uri(img)})"

    return IMG_RE.sub(repl, body)


# ---------------------------------------------------------------------------
# Notebook build
# ---------------------------------------------------------------------------
def build_notebook(slides):
    nb = nbf.v4.new_notebook()
    nb.metadata.update(
        {
            "celltoolbar": "Slideshow",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "rise": {"theme": "simple", "transition": "slide", "scroll": True},
        }
    )

    cells = []
    for idx, (kind, title, body) in enumerate(slides):
        if idx == 0 and kind == "section" and "Slide Contents" in title:
            continue  # drop the document's own meta header

        body = embed_images(body)

        if kind == "content" and TITLE_RE.match(title.strip()):
            # Real title slide: first body line -> big h1, rest -> subtitle.
            blines = [l for l in body.splitlines() if l.strip()]
            deck_title = blines[0].strip().strip("*") if blines else title
            sub = "\n\n".join(l.strip() for l in blines[1:])
            src = f"# {deck_title}\n\n{sub}".rstrip() + "\n"
        elif kind == "section":
            src = f"# {title}\n"
        else:
            src = f"## {title}\n\n{body}".rstrip() + "\n"

        cells.append(nbf.v4.new_markdown_cell(src, metadata=slide_meta("slide")))

    nb.cells = cells
    return nb


def main():
    slides = parse_slides(SRC_MD.read_text())
    n_section = sum(1 for k, *_ in slides if k == "section")
    n_content = sum(1 for k, *_ in slides if k == "content")
    print(f"Parsed {len(slides)} slides ({n_content} content, {n_section} dividers)")

    nb = build_notebook(slides)
    for i, c in enumerate(nb.cells):
        first = c.source.splitlines()[0] if c.source else ""
        print(f"  cell {i:2d} [{c.metadata['slideshow']['slide_type']}] {first[:66]}")

    nbf.write(nb, str(NB_PATH))
    kb = NB_PATH.stat().st_size / 1024
    print(f"\nWrote {NB_PATH} ({kb:.0f} KB, {len(nb.cells)} slides)")


if __name__ == "__main__":
    main()
