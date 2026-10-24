"""
REST API for MemForge: OpenAI-compatible chat completions plus engine telemetry.

Non-streaming completions go through the shared continuous-batching engine, so
concurrent callers are interleaved token-by-token rather than served one at a
time. `use_speculative` routes to the draft+target speculative runtime instead.
"""

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from memforge_ml.runtime.sampling import SamplingParams
from server.api.disconnect_handler import ClientDisconnected, run_cancellable
from server.config import SERVER_CONFIG
from server.engine_manager import ENGINE_MANAGER, EngineSaturated

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)

    model: str = "memforge-target"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=None, alias="max_new_tokens")
    temperature: float = 0.7
    top_p: float = 0.9
    seed: Optional[int] = None
    stream: bool = False
    use_speculative: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo
    memforge_telemetry: Optional[Dict[str, Any]] = None


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=64, ge=1)
    temperature: float = 0.7
    top_p: float = 0.9
    seed: Optional[int] = None
    use_speculative: bool = False


def _build_prompt(messages: List[ChatMessage]) -> str:
    """Render chat messages with the model's own chat template when it has one."""
    tokenizer = ENGINE_MANAGER.tokenizer
    payload = [{"role": m.role, "content": m.content} for m in messages]
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                payload, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    rendered = "\n".join(f"{m.role}: {m.content}" for m in messages)
    return f"{rendered}\nassistant:"


def _sampling(
    max_tokens: Optional[int], temperature: float, top_p: float, seed: Optional[int]
) -> SamplingParams:
    return SamplingParams(
        max_new_tokens=max_tokens or SERVER_CONFIG.default_max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )


@router.get("/v1/health")
async def health_check():
    return {
        "status": "healthy" if ENGINE_MANAGER.load_error is None else "degraded",
        "service": "MemForge LLM Inference Engine",
        "engine_ready": ENGINE_MANAGER.ready,
        "speculative_available": ENGINE_MANAGER.spec_engine is not None,
        "load_error": ENGINE_MANAGER.load_error,
        "timestamp": time.time(),
    }


@router.get("/v1/models")
async def list_models():
    cfg = SERVER_CONFIG
    created = int(time.time())
    data = [
        {
            "id": "memforge-target",
            "object": "model",
            "created": created,
            "owned_by": "memforge",
            "backing_model": cfg.target_model,
        }
    ]
    if ENGINE_MANAGER.spec_engine is not None:
        data.append(
            {
                "id": "memforge-speculative",
                "object": "model",
                "created": created,
                "owned_by": "memforge",
                "backing_model": f"{cfg.target_model} + draft {cfg.draft_model}",
            }
        )
    return {"object": "list", "data": data}


@router.get("/v1/engine/status")
async def engine_status():
    """Full engine configuration, KV pool geometry and cumulative telemetry."""
    return ENGINE_MANAGER.status()


@router.get("/v1/engine/telemetry")
async def engine_telemetry():
    if ENGINE_MANAGER.engine is None:
        raise HTTPException(status_code=503, detail="engine not loaded yet")
    return ENGINE_MANAGER.engine.telemetry()


@router.get("/v1/engine/blocks")
async def engine_blocks():
    """
    Physical block ownership map for the memory heatmap: index -> owning
    sequence id, 0 meaning free.

    `occupancy` keeps its original one-owner-per-block shape so the existing
    heatmap is unaffected. `block_owners` is the share-aware view alongside it:
    a prefix-shared block appears there with several owners, which a flat list
    cannot express.
    """
    if ENGINE_MANAGER.engine is None:
        raise HTTPException(status_code=503, detail="engine not loaded yet")
    occupancy = ENGINE_MANAGER.block_occupancy()
    owners = ENGINE_MANAGER.block_owners()
    return {
        "block_size": ENGINE_MANAGER.engine.block_size,
        "total_blocks": len(occupancy),
        "allocated_blocks": sum(1 for owner in occupancy if owner),
        "occupancy": occupancy,
        "block_owners": owners,
        "shared_blocks": sum(1 for ids in owners.values() if len(ids) > 1),
    }


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus exposition of engine, memory and prefix-sharing state."""
    return PlainTextResponse(
        ENGINE_MANAGER.prometheus_metrics(), media_type="text/plain; version=0.0.4"
    )


#: Returned when the client hung up before its response was ready. Not a real
#: HTTP status -- nothing is sent, because there is no longer a socket to send
#: it on -- but FastAPI needs a response, and 499 is the established convention.
CLIENT_CLOSED_REQUEST = 499


async def _generate_or_fail(request: Request, prompt: str, params: SamplingParams,
                            speculative: bool) -> Dict[str, Any]:
    """
    Run a non-streaming generation, mapping engine failures onto HTTP status.

    Wrapped in `run_cancellable` because the bare await has no idea the client
    left: it would generate every token into a response nobody reads, holding
    physical blocks the whole time. On disconnect the sequence is aborted so its
    blocks come straight back to the pool.
    """
    work = (
        ENGINE_MANAGER.speculative_generate(prompt, params)
        if speculative
        else ENGINE_MANAGER.generate(prompt, params)
    )
    try:
        return await run_cancellable(request, work)
    except ClientDisconnected:
        raise HTTPException(status_code=CLIENT_CLOSED_REQUEST, detail="client disconnected")
    except EngineSaturated as exc:
        # Must precede RuntimeError: EngineSaturated is one, and saturation is a
        # "come back shortly" (429), not a "the engine is broken" (503).
        raise HTTPException(
            status_code=429, detail=str(exc), headers={"Retry-After": "1"}
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/v1/completions")
async def create_completion(req: CompletionRequest, request: Request):
    """Minimal raw-prompt completion endpoint (no chat template applied)."""
    params = _sampling(req.max_tokens, req.temperature, req.top_p, req.seed)
    return await _generate_or_fail(request, req.prompt, params, req.use_speculative)


@router.post("/v1/chat/completions")
async def create_chat_completion(req: ChatCompletionRequest, request: Request):
    prompt = _build_prompt(req.messages)
    params = _sampling(req.max_tokens, req.temperature, req.top_p, req.seed)

    if req.stream:
        return StreamingResponse(
            _sse_stream(prompt, params, req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = await _generate_or_fail(request, prompt, params, req.use_speculative)

    if req.use_speculative:
        telemetry = {
            "mode": "speculative",
            "tokens_per_second": result["tokens_per_second"],
            "time_to_first_token_ms": result["time_to_first_token_ms"],
            "acceptance_rate": result["acceptance_rate"],
            "drafted_tokens": result["drafted_tokens"],
            "accepted_tokens": result["accepted_tokens"],
            "target_forward_passes": result["target_forward_passes"],
            "tokens_per_target_pass": result["tokens_per_target_pass"],
            "rolled_back_tokens": result["rolled_back_tokens"],
            "token_sources": result["token_sources"],
        }
    else:
        telemetry = {
            "mode": "continuous_batching",
            "tokens_per_second": result["tokens_per_second"],
            "time_to_first_token_ms": result["time_to_first_token_ms"],
            "num_preemptions": result["num_preemptions"],
            "engine": ENGINE_MANAGER.live_metrics(),
        }

    completion_tokens = result["generated_tokens"]
    prompt_tokens = result.get("prompt_tokens", 0)
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time() * 1000)}",
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=result["text"]),
                finish_reason=result.get("finish_reason") or "stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        memforge_telemetry=telemetry,
    )


async def _sse_stream(prompt: str, params: SamplingParams, req: ChatCompletionRequest):
    """Server-Sent Events in the OpenAI streaming chunk format."""
    import json

    completion_id = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())

    def chunk(delta: Dict[str, Any], finish_reason: Optional[str] = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    try:
        source = (
            ENGINE_MANAGER.speculative_stream(prompt, params)
            if req.use_speculative
            else ENGINE_MANAGER.stream(prompt, params)
        )
        yield chunk({"role": "assistant", "content": ""})
        # A client hangup closes this generator, which runs the `finally` in
        # EngineManager.stream() and aborts the sequence -- so the blocks come
        # back instead of being held for a connection that is gone.
        async for event in source:
            if event["type"] == "token":
                yield chunk(
                    {"content": event["token"], "memforge_source": event.get("source")}
                )
            elif event["type"] == "error":
                yield f"data: {json.dumps({'error': event['message']})}\n\n"
                break
            elif event["type"] == "done":
                yield chunk({}, finish_reason=event.get("finish_reason") or "stop")
    except Exception as exc:
        yield f"data: {json.dumps({'error': f'{type(exc).__name__}: {exc}'})}\n\n"
    yield "data: [DONE]\n\n"
