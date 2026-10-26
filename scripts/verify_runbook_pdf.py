"""
Prove the PDF lost nothing from the Markdown.

"It looks fine" is not a check. A Markdown-to-PDF conversion drops content in
specific, quiet ways -- a table row that did not match the parser's expected
shape, a fenced block swallowed by an unbalanced fence, a heading absorbed into
a paragraph -- and none of those announce themselves. So this extracts the
PDF's text and asserts that every structural element of the source survived.

What is checked, and why each matters:

  * **headings** -- a missing one means a whole section vanished.
  * **table cells** -- the most common silent loss, and this document's numbers
    live in tables.
  * **code lines** -- the commands are the point of a runbook; a dropped flag
    makes one wrong rather than absent, which is worse.
  * **numbers** -- every measurement in the prose, independent of structure.
  * **list items** -- the "things that waste your time" section is all lists.

Comparison is on text normalised for the ways a PDF renderer legitimately
alters characters: it re-wraps lines, converts `--` to en dashes, and uses
typographic quotes. Those are not data loss. Dropped words are.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

from pypdf import PdfReader

# The Windows console is cp1252; a missing item may well contain the very
# character that broke the conversion, so reporting must not itself fail.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Resolved from this file so the check runs anywhere the repo is cloned.
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "PHASE_E_RUNBOOK.md"
PDF = ROOT / "docs" / "PHASE_E_RUNBOOK.pdf"


def normalise(text: str) -> str:
    """
    Collapse the differences a PDF renderer is entitled to introduce.

    Unicode dashes and quotes become ASCII, inline-code backticks and emphasis
    markers are dropped, and all whitespace collapses to single spaces. What
    remains is the words and numbers, which is what must not go missing.
    """
    text = unicodedata.normalize("NFKD", text)
    for a, b in [
        ("\u2014", "-"), ("\u2013", "-"), ("\u2212", "-"), ("\u00ad", ""),
        ("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"'),
        ("\u00a0", " "), ("\u2192", "->"), ("\u00b7", " "),
        ("\u2713", ""), ("\u26a0", ""),
    ]:
        text = text.replace(a, b)
    text = text.replace("`", "").replace("**", "").replace("*", "")
    # Status emoji and their variation selectors. A PDF renderer keeps the text
    # beside such a glyph and drops the colour glyph, which is a rendering
    # difference rather than data loss.
    #
    # Listed explicitly instead of stripping the "So" Unicode category, which
    # was the first attempt: box-drawing characters are also "So", so that
    # erased the ASCII architecture diagram from both sides of the comparison
    # and silently dropped three code lines from the check. A verifier that
    # quietly stops checking something is worse than one that reports a false
    # positive.
    for emoji in ("️", "✅", "⚠", "\U0001f7e1", "\U0001f6a7"):
        text = text.replace(emoji, "")
    text = re.sub(r"-+", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def pdf_text() -> str:
    reader = PdfReader(str(PDF))
    pages = [p.extract_text() or "" for p in reader.pages]
    print(f"  pdf pages: {len(pages)}")
    return normalise("\n".join(pages))


def source_items():
    """Pull every structural element worth checking out of the Markdown."""
    raw = SOURCE.read_text(encoding="utf-8")
    lines = raw.split("\n")

    headings, cells, code, bullets = [], [], [], []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            if stripped and not stripped.startswith("#"):
                code.append(stripped)
            continue
        if stripped.startswith("#"):
            headings.append(stripped.lstrip("# ").strip())
        elif stripped.startswith("|") and stripped.count("|") >= 2:
            if set(stripped) <= set("|-: "):
                continue  # separator row
            for cell in stripped.strip("|").split("|"):
                cell = cell.strip()
                if cell and not set(cell) <= set("-: "):
                    cells.append(cell)
        elif re.match(r"^[-*] |^\d+\. ", stripped):
            bullets.append(re.sub(r"^([-*] |\d+\. )", "", stripped))

    # Every number that appears anywhere, so a measurement cannot vanish even
    # if its surrounding structure changed.
    numbers = re.findall(r"\d+(?:[.,]\d+)*(?:x|%|\s?(?:MiB|MB|GB|ms|us|µs))?", raw)
    return headings, cells, code, bullets, numbers


def check(name: str, items, haystack: str, min_len: int = 3):
    """Report how many items survived, and show the first few that did not."""
    missing = []
    checked = 0
    for item in items:
        needle = normalise(item)
        if len(needle) < min_len:
            continue
        checked += 1
        if needle not in haystack:
            missing.append(item)
    status = "OK  " if not missing else "FAIL"
    print(f"  [{status}] {name:14} {checked - len(missing):4}/{checked:4} present")
    for m in missing[:8]:
        print(f"           missing: {m[:100]!r}")
    if len(missing) > 8:
        print(f"           ... and {len(missing) - 8} more")
    return len(missing)


def main() -> int:
    if not PDF.exists():
        raise SystemExit(f"no PDF at {PDF}")
    print(f"  source: {SOURCE.stat().st_size:,} bytes")
    print(f"  pdf   : {PDF.stat().st_size:,} bytes")

    haystack = pdf_text()
    print(f"  extracted text: {len(haystack):,} chars")
    print()

    headings, cells, code, bullets, numbers = source_items()
    failures = 0
    failures += check("headings", headings, haystack)
    failures += check("table cells", cells, haystack)
    failures += check("code lines", code, haystack, min_len=6)
    failures += check("list items", bullets, haystack, min_len=12)
    failures += check("numbers", numbers, haystack, min_len=2)

    print()
    if failures:
        print(f"  {failures} item(s) missing from the PDF -- NOT safe to ship")
        return 1
    print("  every heading, table cell, code line, list item and number from")
    print("  the Markdown is present in the PDF. Nothing was lost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
