"""
Default model ids, in one place.

These used to be hardcoded separately in five files, and they had drifted: the
Phase-1 benchmarks defaulted to `TinyLlama-1.1B-Chat-v1.0` and `SmolLM-135M`,
neither of which this project downloads, so running one triggered a surprise
~2 GB fetch instead of using the weights already in the cache.

Both are read from the same environment variables the server reads
(`server/config.py`), so a benchmark, the CLI and the running engine cannot
silently disagree about which model is under test.

This module deliberately holds *only* these two values. The previous
`memforge_ml/config.py` tried to be a whole parallel configuration system,
was imported by nothing, and carried stale model names for a year; it was
deleted rather than fixed. Configuration that the server actually uses lives in
`server/config.py`.
"""

from __future__ import annotations

import os

#: Target (full-size) model. Override with MEMFORGE_TARGET_MODEL.
DEFAULT_TARGET_MODEL = os.environ.get(
    "MEMFORGE_TARGET_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct"
)

#: Draft model for speculative decoding. Override with MEMFORGE_DRAFT_MODEL.
#: Must share a tokenizer with the target, which speculative decoding requires.
DEFAULT_DRAFT_MODEL = os.environ.get(
    "MEMFORGE_DRAFT_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"
)
