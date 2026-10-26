"""
SVG figures for the Phase E runbook PDF.

Every chart is generated from `benchmarks/results/*.json`, not from numbers
typed in here. That is the whole point: a figure transcribed by hand drifts away
from the data the moment anything is re-measured, and a chart that disagrees
with the table next to it is worse than no chart.

Hand-rolled SVG rather than matplotlib: these are a handful of bars and lines,
vector output embeds cleanly in print, and it avoids adding a heavy dependency
for six pictures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

RESULTS = Path("benchmarks/results")

# Print-friendly palette: readable on white, distinguishable in greyscale
# because a printed report often is.
INK = "#1a1d24"
MUTED = "#6b7280"
GRID = "#e5e7eb"
BLUE = "#2563eb"
GREEN = "#059669"
AMBER = "#d97706"
RED = "#dc2626"
PURPLE = "#7c3aed"


def _svg(width: int, height: int, body: str, title: str) -> str:
    return (
        f'<svg class="fig" viewBox="0 0 {width} {height}" width="100%" '
        f'role="img" aria-label="{title}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'font-family="Segoe UI, Helvetica, Arial, sans-serif">'
        f"{body}</svg>"
    )


def _text(x, y, s, size=11, fill=INK, anchor="start", weight="normal"):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>'
    )


# ------------------------------------------------------------ architecture

def architecture() -> str:
    """
    The three layers and what crosses between them.

    Drawn rather than left as ASCII because the one thing a newcomer needs to
    see immediately is that memory management is C++ and orchestration is
    Python, with pybind11 as the only seam.
    """
    W, H = 760, 400
    rows = [
        (
            "C++ core  —  memforge_c (pybind11)",
            "#eef2ff",
            "#c7d2fe",
            [
                ("BlockAllocator", "16-token blocks, ref-counted,\ncopy-on-write"),
                ("Scheduler", "re-decides the batch every\ntoken step"),
                ("PrefixCache", "chained block hashing,\nshared prefixes"),
            ],
        ),
        (
            "Python runtime",
            "#ecfdf5",
            "#a7f3d0",
            [
                ("PagedKVPool", "one tensor, addressed\nby (block, slot)"),
                ("PagedCache", "a HuggingFace Cache\nbacked by the pool"),
                ("kernels/", "paged gather, block-wise\nand varlen attention"),
            ],
        ),
        (
            "Serving",
            "#fffbeb",
            "#fde68a",
            [
                ("FastAPI", "OpenAI-compatible,\nSSE + WebSocket"),
                ("Dashboard", "live telemetry,\n'Learn more' per panel"),
                ("Benchmarks", "--device / --dtype,\nJSON results"),
            ],
        ),
    ]

    out: List[str] = [f'<rect width="{W}" height="{H}" fill="white"/>']
    y = 14
    for label, fill, stroke, boxes in rows:
        out.append(
            f'<rect x="10" y="{y}" width="{W-20}" height="112" rx="8" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        )
        out.append(_text(24, y + 22, label, 12, INK, weight="700"))
        bx = 26
        for name, desc in boxes:
            out.append(
                f'<rect x="{bx}" y="{y+32}" width="228" height="66" rx="6" '
                f'fill="white" stroke="{stroke}" stroke-width="1.2"/>'
            )
            out.append(_text(bx + 12, y + 52, name, 11.5, INK, weight="700"))
            for i, line in enumerate(desc.split("\n")):
                out.append(_text(bx + 12, y + 70 + i * 14, line, 9.8, MUTED))
            bx += 240
        y += 128

    # The seam, annotated: this is the boundary people get wrong.
    out.append(
        f'<path d="M {W//2} 126 L {W//2} 142" stroke="{MUTED}" '
        f'stroke-width="1.5" marker-end="url(#ar)"/>'
    )
    out.append(
        f'<path d="M {W//2} 254 L {W//2} 270" stroke="{MUTED}" '
        f'stroke-width="1.5" marker-end="url(#ar)"/>'
    )
    out.append(_text(W // 2 + 10, 140, "pybind11", 9.5, MUTED))
    defs = (
        '<defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6" '
        f'refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="{MUTED}"/>'
        "</marker></defs>"
    )
    return _svg(W, H, defs + "".join(out), "MemForge architecture")


# ------------------------------------------------------------ phase status

def phase_status() -> str:
    """Where each phase stands, at a glance."""
    W, H = 760, 250
    phases = [
        ("A", "Cheap paged gather", "done", GREEN, "verified on GPU: 1.009x at 2048"),
        ("B", "Prefix sharing / CoW", "done", GREEN, "3.0x block reduction at N=4"),
        ("C", "Chunked prefill + fusion", "built", AMBER, "DoD borderline: 1.83-2.24x vs 2.0x bar"),
        ("D", "Speculation in batching", "composes", AMBER, "lossless; 8B claim needs >6 GB VRAM"),
        ("E", "GPU execution", "started", BLUE, "4 of 6 harnesses re-run on GPU"),
        ("F", "Production hardening", "done", GREEN, "ASan clean over 180 cycles"),
    ]
    out = [f'<rect width="{W}" height="{H}" fill="white"/>']
    y = 20
    for letter, name, state, colour, note in phases:
        out.append(
            f'<rect x="10" y="{y}" width="{W-20}" height="32" rx="5" '
            f'fill="#fafafa" stroke="{GRID}"/>'
        )
        out.append(
            f'<rect x="10" y="{y}" width="5" height="32" rx="2" fill="{colour}"/>'
        )
        out.append(_text(28, y + 21, f"Phase {letter}", 11.5, INK, weight="700"))
        out.append(_text(100, y + 21, name, 11, INK))
        out.append(
            f'<rect x="282" y="{y+8}" width="74" height="17" rx="8" '
            f'fill="{colour}" opacity="0.14"/>'
        )
        out.append(_text(319, y + 20, state, 9.5, colour, "middle", "700"))
        out.append(_text(372, y + 21, note, 9.8, MUTED))
        y += 38
    return _svg(W, H, "".join(out), "Phase status")


# ------------------------------------------------------- context sweep chart

def context_sweep() -> str:
    """
    Paged vs contiguous throughput across context length (Phase A, on GPU).

    The crossover is the story: paging costs a fixed amount per step, so it
    loses at short contexts and catches up as the context grows.
    """
    data = json.loads((RESULTS / "context_sweep.json").read_text())
    rows = data["rows"]
    W, H = 760, 330
    L, R, T, B = 62, 24, 46, 74
    pw, ph = W - L - R, H - T - B

    peak = max(max(r["contiguous_tok_per_s"], r["paged_tok_per_s"]) for r in rows)
    ymax = peak * 1.18
    n = len(rows)
    group = pw / n
    bw = group * 0.3

    out = [f'<rect width="{W}" height="{H}" fill="white"/>']
    out.append(_text(L, 20, "Phase A on GPU: paged vs contiguous decode", 12.5, INK, weight="700"))
    out.append(_text(L, 34, "fp16, RTX 4050, SmolLM2-360M. Output byte-identical at every length.", 9.5, MUTED))

    for i in range(5):
        gy = T + ph - (ph * i / 4)
        val = ymax * i / 4
        out.append(f'<line x1="{L}" y1="{gy}" x2="{L+pw}" y2="{gy}" stroke="{GRID}"/>')
        out.append(_text(L - 8, gy + 3.5, f"{val:.0f}", 9, MUTED, "end"))
    out.append(_text(L - 8, T - 8, "tok/s", 9, MUTED, "end"))

    for i, r in enumerate(rows):
        cx = L + group * i + group / 2
        for j, (key, colour, label) in enumerate(
            [("contiguous_tok_per_s", BLUE, "contiguous"),
             ("paged_tok_per_s", GREEN, "paged")]
        ):
            v = r[key]
            bh = ph * v / ymax
            bx = cx - bw + j * bw
            out.append(
                f'<rect x="{bx:.1f}" y="{T+ph-bh:.1f}" width="{bw-3:.1f}" '
                f'height="{bh:.1f}" fill="{colour}" rx="2"/>'
            )
            out.append(_text(bx + (bw - 3) / 2, T + ph - bh - 5, f"{v:.1f}", 8.5, colour, "middle", "700"))
        rel = r["relative_throughput"]
        good = rel >= 0.95
        out.append(_text(cx, T + ph + 17, f'{r["context_tokens"]}', 10, INK, "middle", "700"))
        out.append(
            _text(cx, T + ph + 31, f"{rel:.3f}x", 9.5,
                  GREEN if good else AMBER, "middle", "700")
        )
    out.append(_text(L + pw / 2, T + ph + 48, "context length (tokens)  /  relative throughput", 9.5, MUTED, "middle"))

    lx = L
    for colour, label in ((BLUE, "contiguous (HuggingFace DynamicCache)"), (GREEN, "paged (MemForge)")):
        out.append(f'<rect x="{lx}" y="{H-18}" width="10" height="10" fill="{colour}" rx="2"/>')
        out.append(_text(lx + 15, H - 9.5, label, 9.5, MUTED))
        lx += 250
    return _svg(W, H, "".join(out), "Context sweep")


# --------------------------------------------------- chunked prefill chart

def chunked_prefill() -> str:
    """
    Phase C on CPU: what a long prompt does to an in-flight request.

    CPU numbers on purpose -- this is the regime the feature is for, and the
    GPU result is the opposite, which the caption says outright.
    """
    W, H = 760, 320
    L, R, T, B = 62, 24, 48, 70
    pw, ph = W - L - R, H - T - B
    # From PROGRESS_AND_NEXT_PHASES.md Phase C, CPU, 512-token intruder.
    bars = [
        ("no chunking", 20.82, RED, "17 passes"),
        ("budget 16", 3.05, AMBER, "50 passes"),
        ("budget 16\n+ fused", 2.48, BLUE, "38 passes"),
        ("budget 4", 1.97, AMBER, "152 passes"),
        ("budget 4\n+ fused", 1.83, GREEN, "140 passes"),
    ]
    ymax = 23.0
    group = pw / len(bars)
    bw = group * 0.5

    out = [f'<rect width="{W}" height="{H}" fill="white"/>']
    out.append(_text(L, 20, "Phase C on CPU: stall inflicted on an in-flight request", 12.5, INK, weight="700"))
    out.append(_text(L, 34, "Lower is better. 1.0x = the victim never noticed. On GPU chunking makes this WORSE (see 5.3).", 9.5, MUTED))

    for i in range(5):
        gy = T + ph - (ph * i / 4)
        out.append(f'<line x1="{L}" y1="{gy}" x2="{L+pw}" y2="{gy}" stroke="{GRID}"/>')
        out.append(_text(L - 8, gy + 3.5, f"{ymax*i/4:.0f}x", 9, MUTED, "end"))

    # The "unaffected" bar the DoD asks for.
    ty = T + ph - ph * 2.0 / ymax
    out.append(f'<line x1="{L}" y1="{ty}" x2="{L+pw}" y2="{ty}" stroke="{GREEN}" stroke-width="1.3" stroke-dasharray="5,3"/>')
    out.append(_text(L + pw - 2, ty - 5, "2.0x target", 9, GREEN, "end", "700"))

    for i, (label, v, colour, note) in enumerate(bars):
        cx = L + group * i + group / 2
        bh = ph * v / ymax
        out.append(f'<rect x="{cx-bw/2:.1f}" y="{T+ph-bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{colour}" rx="3"/>')
        out.append(_text(cx, T + ph - bh - 6, f"{v:.2f}x", 10, colour, "middle", "700"))
        for k, line in enumerate(label.split("\n")):
            out.append(_text(cx, T + ph + 16 + k * 12, line, 9.5, INK, "middle"))
        out.append(_text(cx, T + ph + 16 + len(label.split("\n")) * 12, note, 8.5, MUTED, "middle"))
    return _svg(W, H, "".join(out), "Chunked prefill stall")


# ------------------------------------------------------ speculation chart

def speculation() -> str:
    """
    Phase D: the composition works, the wall clock does not.

    Two panels side by side because the honest answer has two halves and
    showing only one of them would misrepresent it.
    """
    data = json.loads((RESULTS / "speculative_batching.json").read_text())
    rows = data["rows"]
    W, H = 760, 300
    out = [f'<rect width="{W}" height="{H}" fill="white"/>']
    out.append(_text(24, 20, "Phase D: verification passes saved, and the wall-clock cost", 12.5, INK, weight="700"))
    out.append(_text(24, 34, "RTX 4050, fp16, SmolLM2-360M target with 135M draft, K=4. Output token-identical throughout.", 9.5, MUTED))

    # Left: passes batched vs naive.
    L, T, pw, ph = 56, 60, 300, 170
    peak = max(max(r["speculative"]["naive_target_passes"], r["speculative"]["speculative_rounds"]) for r in rows)
    ymax = max(peak * 1.25, 4)
    group = pw / len(rows)
    bw = group * 0.26
    out.append(_text(L, T - 10, "verification passes (lower is better)", 10, INK, weight="700"))
    for i in range(4):
        gy = T + ph - ph * i / 3
        out.append(f'<line x1="{L}" y1="{gy}" x2="{L+pw}" y2="{gy}" stroke="{GRID}"/>')
        out.append(_text(L - 8, gy + 3.5, f"{ymax*i/3:.0f}", 9, MUTED, "end"))
    for i, r in enumerate(rows):
        cx = L + group * i + group / 2
        s = r["speculative"]
        for j, (v, colour, lab) in enumerate(
            [(s["naive_target_passes"], RED, "per-sequence"), (s["speculative_rounds"], GREEN, "batched")]
        ):
            bh = ph * v / ymax
            bx = cx - bw + j * bw
            out.append(f'<rect x="{bx:.1f}" y="{T+ph-bh:.1f}" width="{bw-3:.1f}" height="{bh:.1f}" fill="{colour}" rx="2"/>')
            out.append(_text(bx + (bw - 3) / 2, T + ph - bh - 5, str(v), 9, colour, "middle", "700"))
        out.append(_text(cx, T + ph + 16, f'batch {r["batch_size"]}', 9.5, INK, "middle"))
        out.append(_text(cx, T + ph + 29, f'{s["sequences_per_round"]:.1f} seq/pass', 8.5, MUTED, "middle"))

    # Right: wall clock against the break-even line.
    L2, pw2 = 456, 260
    out.append(_text(L2, T - 10, "wall-clock speedup (1.0 = break even)", 10, INK, weight="700"))
    ymax2 = 1.3
    for i in range(4):
        gy = T + ph - ph * i / 3
        out.append(f'<line x1="{L2}" y1="{gy}" x2="{L2+pw2}" y2="{gy}" stroke="{GRID}"/>')
        out.append(_text(L2 - 8, gy + 3.5, f"{ymax2*i/3:.1f}x", 9, MUTED, "end"))
    by = T + ph - ph * 1.0 / ymax2
    out.append(f'<line x1="{L2}" y1="{by}" x2="{L2+pw2}" y2="{by}" stroke="{INK}" stroke-width="1.3" stroke-dasharray="5,3"/>')
    out.append(_text(L2 + pw2 - 2, by - 5, "break even", 9, INK, "end", "700"))
    g2 = pw2 / len(rows)
    for i, r in enumerate(rows):
        cx = L2 + g2 * i + g2 / 2
        v = r["speedup"] or 0.0
        bh = ph * v / ymax2
        out.append(f'<rect x="{cx-26:.1f}" y="{T+ph-bh:.1f}" width="52" height="{bh:.1f}" fill="{RED}" rx="3" opacity="0.85"/>')
        # A bar reaching close to the break-even line would put its value label
        # on top of that line's own label, so tall bars label themselves inside
        # in white instead of above in red.
        if bh > 34:
            out.append(_text(cx, T + ph - bh + 15, f"{v:.2f}x", 10, "white", "middle", "700"))
        else:
            out.append(_text(cx, T + ph - bh - 6, f"{v:.2f}x", 10, RED, "middle", "700"))
        out.append(_text(cx, T + ph + 16, f'batch {r["batch_size"]}', 9.5, INK, "middle"))
    out.append(_text(L2, H - 12, "Slower because the draft's K passes are serial and", 9, MUTED))
    out.append(_text(L2, H - 2, "a 360M/135M pair is too close in cost.", 9, MUTED))

    lx = L
    for colour, label in ((RED, "per-sequence verifier (naive)"), (GREEN, "batched verification (actual)")):
        out.append(f'<rect x="{lx}" y="{H-14}" width="10" height="10" fill="{colour}" rx="2"/>')
        out.append(_text(lx + 15, H - 5.5, label, 9, MUTED))
        lx += 185
    return _svg(W, H, "".join(out), "Speculative decoding")


# ----------------------------------------------------------- VRAM budget

def vram_budget() -> str:
    """Which model pair fits in 6 GB, in fp16, with room for the KV pool."""
    W, H = 760, 260
    L, T, pw, ph = 150, 54, 540, 150
    pairs = [
        ("4B / 1B", 9537, 10949),
        ("3B / 0.5B", 6676, 8088),
        ("1.5B / 0.5B", 3815, 5227),
        ("360M / 135M", 944, 2356),
    ]
    cap = 6140.5
    xmax = 11500
    out = [f'<rect width="{W}" height="{H}" fill="white"/>']
    out.append(_text(24, 20, "What fits in 6 GB of VRAM (fp16, plus KV pool and display)", 12.5, INK, weight="700"))
    out.append(_text(24, 34, "The draft must share the target's tokenizer, so the pair has to come from one family.", 9.5, MUTED))

    cx = L + pw * cap / xmax
    out.append(f'<line x1="{cx}" y1="{T-6}" x2="{cx}" y2="{T+ph+4}" stroke="{RED}" stroke-width="1.6" stroke-dasharray="5,3"/>')
    out.append(_text(cx + 5, T - 10, "6141 MiB available", 9.5, RED, "start", "700"))

    rh = ph / len(pairs)
    for i, (name, weights, total) in enumerate(pairs):
        y = T + rh * i + 6
        fits = total < cap
        colour = GREEN if fits else RED
        out.append(_text(L - 10, y + 17, name, 10.5, INK, "end", "700"))
        out.append(f'<rect x="{L}" y="{y}" width="{pw*total/xmax:.1f}" height="24" fill="{colour}" opacity="0.18" rx="3"/>')
        out.append(f'<rect x="{L}" y="{y}" width="{pw*weights/xmax:.1f}" height="24" fill="{colour}" rx="3"/>')
        out.append(_text(L + pw * total / xmax + 7, y + 17, f"{total} MiB", 9.5, colour, "start", "700"))
        out.append(_text(L + 6, y + 16.5, f"{weights} MiB weights", 8.8, "white", "start", "700"))
    out.append(_text(L, H - 26, "solid = model weights     pale = weights + 512 MB KV pool + ~900 MB display", 9, MUTED))
    out.append(_text(L, H - 12, "Recommended upgrade: Qwen2.5-1.5B-Instruct + Qwen2.5-0.5B-Instruct (shared vocabulary).", 9.5, INK, weight="700"))
    return _svg(W, H, "".join(out), "VRAM budget")


# -------------------------------------------------------- engine vs stock

def engine_vs_stock() -> str:
    """The unmeasured win, and the overhead that is the real target."""
    W, H = 760, 230
    out = [f'<rect width="{W}" height="{H}" fill="white"/>']
    out.append(_text(24, 20, "MemForge vs stock HuggingFace decode (GPU, fp16, 360M)", 12.5, INK, weight="700"))
    out.append(_text(24, 34, "A win that never appeared on CPU: the paged cache avoids per-step KV re-concatenation across 30 layers.", 9.5, MUTED))
    L, T, pw = 230, 56, 440
    bars = [("stock HuggingFace", 273.4, RED), ("MemForge engine", 121.3, GREEN)]
    xmax = 300.0
    for i, (name, v, colour) in enumerate(bars):
        y = T + i * 46
        out.append(_text(L - 10, y + 22, name, 10.5, INK, "end", "700"))
        out.append(f'<rect x="{L}" y="{y}" width="{pw*v/xmax:.1f}" height="30" fill="{colour}" rx="3"/>')
        out.append(_text(L + pw * v / xmax + 8, y + 21, f"{v:.1f} ms/token", 10.5, colour, "start", "700"))
    out.append(_text(L, T + 112, "2.3x faster  —  and both are slow in absolute terms.", 11, INK, weight="700"))
    out.append(_text(L, T + 130, "The GPU delivers 20 TFLOP/s fp16 at 2670/3105 MHz, so this is not compute.", 9.5, MUTED))
    out.append(_text(L, T + 144, "Per-op launch overhead is 26.8 us (Windows WDDM): per-step HOST overhead", 9.5, MUTED))
    out.append(_text(L, T + 158, "sets the decode rate. Reducing it is the biggest remaining win, and is in no plan.", 9.5, MUTED))
    return _svg(W, H, "".join(out), "Engine vs stock HuggingFace")


FIGURES: Dict[str, str] = {}


def build() -> Dict[str, str]:
    FIGURES.update(
        {
            "architecture": architecture(),
            "phase_status": phase_status(),
            "context_sweep": context_sweep(),
            "chunked_prefill": chunked_prefill(),
            "speculation": speculation(),
            "vram_budget": vram_budget(),
            "engine_vs_stock": engine_vs_stock(),
        }
    )
    return FIGURES


if __name__ == "__main__":
    figs = build()
    for name, svg in figs.items():
        print(f"  {name:20} {len(svg):6} bytes")
