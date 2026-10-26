"""
Render PHASE_E_RUNBOOK.md to a print-quality PDF, with figures.

Approach: Markdown -> styled HTML -> headless Chrome --print-to-pdf. Chrome is
used rather than a Python PDF library because it is the only option here with
full CSS paged-media support, and that is what gives real page breaks, repeated
table headers and vector SVG rather than bitmaps.

**Nothing is rewritten.** The Markdown is converted, not edited: the same
parser output carries every heading, table row, list item and code block
through to the page. Figures are *added* alongside the prose they illustrate,
never in place of it -- the ASCII architecture block stays, because someone
reading the Markdown on a terminal still needs it.

`verify_pdf.py` then extracts the PDF's text and checks every heading, number
and table cell from the source survived. A conversion that silently drops a row
is the obvious failure mode and it is not left to inspection.
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runbook_figures as figures  # noqa: E402

import markdown  # noqa: E402

# Resolved from this file, never hardcoded: the repo has to build on any
# machine and any OS, which an absolute path quietly prevents.
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "PHASE_E_RUNBOOK.md"
HTML_OUT = ROOT / "docs" / "_runbook.build.html"
PDF_OUT = ROOT / "docs" / "PHASE_E_RUNBOOK.pdf"

# Chrome is used for its CSS paged-media support; Edge is the same engine.
# Listed per platform, and PATH is consulted first so a non-standard
# install still works.
CHROME_ON_PATH = [
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
    "microsoft-edge", "chrome",
]
CHROME_CANDIDATES = [
    # Windows
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    # macOS
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    # Linux
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    "/snap/bin/chromium",
]

# Where each figure goes: after the first heading whose text matches. Anchored
# on headings rather than line numbers so editing the prose cannot silently
# move a figure into the wrong section.
FIGURE_PLACEMENT = [
    ("1. What this project actually is", "architecture",
     "Figure 1 — The three layers. Memory management is C++; orchestration is "
     "Python; pybind11 is the only seam."),
    ("2. Where the project stands", "phase_status",
     "Figure 2 — Phase status. Amber means built and working but with a "
     "definition of done written for hardware this machine does not have."),
    ("3. Models — what to download, and why", "vram_budget",
     "Figure 3 — VRAM budget on a 6 GB card. 4B/1B does not fit; the weights "
     "alone exceed the card."),
    ("5.1 Context sweep — the Phase A claim", "context_sweep",
     "Figure 4 — Phase A on GPU. Paging loses at short contexts and catches up "
     "by 2048, where the target is met. Output byte-identical throughout."),
    ("5.3 Chunked prefill and fusion — the Phase C claim", "chunked_prefill",
     "Figure 5 — Phase C on CPU. Each step toward the target costs more "
     "scheduler steps, visible in the pass counts."),
    ("5.4 Speculation inside batching — the Phase D claim", "speculation",
     "Figure 6 — Phase D, both halves. The structural win is real (left); the "
     "wall clock is still a loss (right)."),
    ("6.2 The real optimisation target nobody has touched", "engine_vs_stock",
     "Figure 7 — The unmeasured win, and the overhead that is the actual "
     "bottleneck."),
]

CSS = """
@page {
  size: A4;
  margin: 16mm 14mm 18mm 14mm;
  @bottom-center { content: counter(page); }
}
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 10.1pt;
  line-height: 1.52;
  color: #1a1d24;
  margin: 0;
}
/* ---------------------------------------------------------- title page */
.cover { page-break-after: always; padding-top: 42mm; }
.cover h1 {
  font-size: 30pt; line-height: 1.12; margin: 0 0 6mm;
  border: none; padding: 0; letter-spacing: -0.4pt;
}
.cover .sub { font-size: 12.5pt; color: #4b5563; margin-bottom: 14mm; }
.cover .meta {
  font-size: 9.5pt; color: #6b7280; border-top: 2px solid #1a1d24;
  padding-top: 4mm; line-height: 1.85;
}
.cover .meta b { color: #1a1d24; }
.cover .warn {
  margin-top: 12mm; padding: 4mm 5mm; border-left: 3px solid #d97706;
  background: #fffbeb; font-size: 9.8pt; color: #78350f;
}
/* ------------------------------------------------------------ headings */
h1 {
  font-size: 18pt; margin: 0 0 4mm; padding-bottom: 2.5mm;
  border-bottom: 2px solid #1a1d24; page-break-after: avoid;
  page-break-before: always;
}
h1:first-of-type { page-break-before: avoid; }
h2 {
  font-size: 13.5pt; margin: 9mm 0 3mm; padding-bottom: 1.5mm;
  border-bottom: 1px solid #d1d5db; page-break-after: avoid;
}
h3 { font-size: 11.4pt; margin: 6mm 0 2mm; page-break-after: avoid; }
h4 { font-size: 10.4pt; margin: 4.5mm 0 1.5mm; page-break-after: avoid; }
p { margin: 0 0 2.6mm; orphans: 3; widows: 3; }
ul, ol { margin: 0 0 3mm; padding-left: 6mm; }
li { margin-bottom: 1.2mm; }
strong { font-weight: 650; }
a { color: #2563eb; text-decoration: none; }
/* ---------------------------------------------------------------- code */
code {
  font-family: "Cascadia Mono", Consolas, "Courier New", monospace;
  font-size: 8.9pt; background: #f3f4f6; padding: 0.4mm 1.1mm;
  border-radius: 2px; color: #111827;
}
pre {
  background: #f8f9fb; border: 1px solid #e5e7eb; border-left: 3px solid #6b7280;
  border-radius: 3px; padding: 3mm 3.5mm; margin: 0 0 3.5mm;
  overflow: hidden; page-break-inside: avoid;
}
pre code {
  background: none; padding: 0; line-height: 1.4;
  /* 7.4pt fits the widest line in this document (107 chars of the ASCII
     architecture diagram) inside the 175mm of usable column, so nothing
     wraps at all. Critically there is NO word-break: Chrome's break-word
     split "--fused" across lines as "-" + "-fused", which a reader would
     copy and run as a broken flag -- a wrong command rather than a missing
     one. Long lines may now wrap at spaces; tokens stay intact. */
  font-size: 7.4pt;
  white-space: pre-wrap;
  word-break: normal;
  overflow-wrap: normal;
}
/* -------------------------------------------------------------- tables */
table {
  width: 100%; border-collapse: collapse; margin: 0 0 4mm;
  font-size: 9.1pt; page-break-inside: avoid;
}
thead { display: table-header-group; }
th {
  background: #f3f4f6; text-align: left; font-weight: 650;
  padding: 1.7mm 2.2mm; border-bottom: 1.5px solid #9ca3af;
}
td { padding: 1.5mm 2.2mm; border-bottom: 1px solid #e5e7eb; vertical-align: top; }
tr:nth-child(even) td { background: #fafafa; }
/* ---------------------------------------------------------- blockquote */
blockquote {
  margin: 0 0 3.5mm; padding: 2.5mm 4mm; border-left: 3px solid #2563eb;
  background: #eff6ff; font-size: 9.6pt; page-break-inside: avoid;
}
blockquote p:last-child { margin-bottom: 0; }
/* ------------------------------------------------------------- figures */
figure {
  margin: 4mm 0 5mm; page-break-inside: avoid; border: 1px solid #e5e7eb;
  border-radius: 4px; padding: 3mm 3mm 2mm; background: #fff;
}
figure svg { display: block; }
figcaption {
  font-size: 8.8pt; color: #4b5563; margin-top: 2mm; padding-top: 1.5mm;
  border-top: 1px solid #f3f4f6; line-height: 1.45;
}
hr { border: none; border-top: 1px solid #d1d5db; margin: 6mm 0; }
"""


def build_html() -> str:
    text = SOURCE.read_text(encoding="utf-8")

    # Strip the H1 and the audience line; they become the cover page. Captured
    # verbatim so no wording is invented for the cover.
    lines = text.split("\n")
    title = lines[0].lstrip("# ").strip()
    body_start = 1
    audience = ""
    for i in range(1, min(12, len(lines))):
        if lines[i].startswith("**Written for"):
            audience = lines[i]
            # The audience note continues to the next blank line.
            j = i + 1
            while j < len(lines) and lines[j].strip():
                audience += " " + lines[j].strip()
                j += 1
            body_start = j
            break
    audience_html = markdown.markdown(audience) if audience else ""

    rest = "\n".join(lines[body_start:])
    rendered = markdown.markdown(
        rest,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )

    figs = figures.build()

    # Insert each figure after its heading. Matching on the rendered heading
    # text rather than the raw Markdown, because that is what is actually in
    # the document being assembled.
    inserted = []
    for heading_text, fig_name, caption in FIGURE_PLACEMENT:
        escaped = re.escape(html.escape(heading_text))
        pattern = re.compile(rf"(<h[23][^>]*>{escaped}</h[23]>)")
        block = (
            f"\\1\n<figure>{figs[fig_name]}"
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )
        new_rendered, n = pattern.subn(block, rendered, count=1)
        if n == 1:
            rendered = new_rendered
            inserted.append(fig_name)
        else:
            print(f"  WARNING: no anchor for figure {fig_name!r} "
                  f"(heading {heading_text!r})")
    print(f"  figures placed: {len(inserted)}/{len(FIGURE_PLACEMENT)}")

    cover = f"""
<div class="cover">
  <h1>{html.escape(title)}</h1>
  <div class="sub">An inference engine built from scratch: paged KV cache,
     continuous batching, prefix sharing and speculative decoding — and the
     runbook to finish measuring it.</div>
  <div class="meta">
    <b>Repository</b> &nbsp; memforge &nbsp;·&nbsp; branch <code>main</code><br>
    <b>Measured on</b> &nbsp; 2026-10-26<br>
    <b>Hardware</b> &nbsp; Intel 10-core / 23.7 GB RAM &nbsp;·&nbsp;
       NVIDIA RTX 4050 Laptop, 6141 MiB, compute 8.9<br>
    <b>Stack</b> &nbsp; torch 2.11.0+cu128 (CUDA 12.8) &nbsp;·&nbsp;
       Python 3.12.7 &nbsp;·&nbsp; Windows 11<br>
    <b>Test state</b> &nbsp; 398 Python tests &nbsp;·&nbsp; 18 C++ cases
       &nbsp;·&nbsp; lint clean &nbsp;·&nbsp; 0 blocks leaked, 0 memory errors
  </div>
  {audience_html}
  <div class="warn"><b>On the numbers in this document.</b> Every figure comes
    from a committed result in <code>benchmarks/results/</code> on the hardware
    named above, and every chart is generated from those files rather than
    transcribed. Where a result is unfavourable it is stated as such — two of
    the features measured here are losses on this hardware, and both say so.
  </div>
</div>
"""

    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
        f"<body>{cover}{rendered}</body></html>"
    )


def find_chrome() -> str:
    """Locate a Chromium-family browser on any of the three platforms."""
    import shutil

    for name in CHROME_ON_PATH:
        found = shutil.which(name)
        if found:
            return found
    for path in CHROME_CANDIDATES:
        if Path(path).exists():
            return path
    raise SystemExit(
        "No Chrome, Chromium or Edge found. One is required: it is the only "
        "renderer available here with full CSS paged-media support, which is "
        "what gives real page breaks and vector figures rather than bitmaps. "
        "Install one, or put it on PATH."
    )


def main() -> int:
    html_text = build_html()
    HTML_OUT.write_text(html_text, encoding="utf-8")
    print(f"  html: {len(html_text):,} bytes -> {HTML_OUT}")

    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    if PDF_OUT.exists():
        PDF_OUT.unlink()

    browser = find_chrome()
    print(f"  renderer: {Path(browser).name}")
    done = subprocess.run(
        [
            browser,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=10000",
            f"--print-to-pdf={PDF_OUT}",
            HTML_OUT.resolve().as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if not PDF_OUT.exists():
        print(done.stdout[-2000:])
        print(done.stderr[-2000:])
        raise SystemExit("Chrome did not produce a PDF")

    size = PDF_OUT.stat().st_size
    print(f"  pdf : {size:,} bytes -> {PDF_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
