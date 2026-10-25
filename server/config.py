"""Server + engine configuration. Every field can be overridden by an env var."""

import os

from pydantic import BaseModel, ConfigDict


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class ServerConfig(BaseModel):
    # `model_` is a protected prefix in pydantic v2; we use target_/draft_ instead.
    model_config = ConfigDict(protected_namespaces=())

    host: str = _env_str("MEMFORGE_HOST", "127.0.0.1")
    port: int = _env_int("MEMFORGE_PORT", 8000)
    reload: bool = _env_bool("MEMFORGE_RELOAD", False)

    # Models. Defaults are small enough to run on CPU; both share one tokenizer,
    # which speculative decoding requires.
    target_model: str = _env_str("MEMFORGE_TARGET_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
    draft_model: str = _env_str("MEMFORGE_DRAFT_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")

    # Engine
    block_size: int = _env_int("MEMFORGE_BLOCK_SIZE", 16)
    kv_budget_mb: float = _env_float("MEMFORGE_KV_BUDGET_MB", 384.0)
    max_num_seqs: int = _env_int("MEMFORGE_MAX_NUM_SEQS", 8)
    max_model_len: int = _env_int("MEMFORGE_MAX_MODEL_LEN", 1024)
    lookahead_k: int = _env_int("MEMFORGE_LOOKAHEAD_K", 4)
    batched_decode: bool = _env_bool("MEMFORGE_BATCHED_DECODE", True)

    # Phase A tier 2: the append-only decode mirror. Off by default because
    # it is a contiguous copy of each running sequence's KV -- the thing
    # paging exists to avoid -- in exchange for a measured 1.15x over stock
    # contiguous decode. The budget caps the total across all sequences; a
    # request that would exceed it simply reads the pool instead, so enabling
    # this can never turn into an out-of-memory failure.
    decode_view_cache: bool = _env_bool("MEMFORGE_DECODE_VIEW_CACHE", False)
    view_cache_budget_mb: float = _env_float("MEMFORGE_VIEW_CACHE_BUDGET_MB", 512.0)

    # Phase A tier 3: block-wise attention straight out of the pool, so the
    # context is never materialised and working memory is O(block_size).
    # The only tier that costs no memory, and slower than the others on CPU
    # because iterating blocks in Python pays interpreter overhead per
    # block. Off by default for that reason; it is the structure a GPU
    # kernel would take. Mutually exclusive with the decode mirror.
    block_attention: bool = _env_bool("MEMFORGE_BLOCK_ATTENTION", False)

    # Toggles. Loading the draft model costs extra memory, so it is opt-out.
    load_draft_model: bool = _env_bool("MEMFORGE_LOAD_DRAFT", True)
    eager_load: bool = _env_bool("MEMFORGE_EAGER_LOAD", True)

    max_concurrency: int = _env_int("MEMFORGE_MAX_CONCURRENCY", 32)
    default_max_tokens: int = _env_int("MEMFORGE_DEFAULT_MAX_TOKENS", 96)


SERVER_CONFIG = ServerConfig()
