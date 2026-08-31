"""
FastAPI REST routes for MemForge server (OpenAI-compatible chat completions + telemetry).
"""

import time
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field


router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "memforge-llama"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=128, alias="max_new_tokens")
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    use_speculative: bool = True


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo
    memforge_telemetry: Optional[Dict[str, Any]] = None


@router.get("/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "MemForge LLM Inference Engine",
        "timestamp": time.time(),
    }


@router.get("/v1/models")
async def list_models(request: Request):
    app_state = request.app.state
    return {
        "object": "list",
        "data": [
            {
                "id": "memforge-target-model",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "memforge",
            },
            {
                "id": "memforge-speculative",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "memforge",
            }
        ]
    }


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(req: ChatCompletionRequest, request: Request):
    app_state = request.app.state

    # Format user prompt
    prompt = "\n".join([f"{m.role}: {m.content}" for m in req.messages]) + "\nassistant:"

    # If speculative engine is available in app_state
    if hasattr(app_state, "spec_engine") and req.use_speculative:
        res = app_state.spec_engine.generate(
            prompt,
            max_new_tokens=req.max_tokens or 128,
            temperature=req.temperature,
        )
        generated_text = res["generated_text"]
        num_tokens = res["num_generated_tokens"]
        telemetry = {
            "tokens_per_second": res["tokens_per_second"],
            "acceptance_rate": res["acceptance_rate"],
            "total_drafted": res["total_drafted_tokens"],
            "total_accepted": res["total_accepted_tokens"],
            "token_origins": res.get("token_origins", []),
        }
    else:
        # Fallback to naive runner or mock
        if hasattr(app_state, "naive_runner"):
            res = app_state.naive_runner.generate(
                prompt,
                max_new_tokens=req.max_tokens or 128,
                temperature=req.temperature,
            )
            generated_text = res["generated_text"]
            num_tokens = res["num_generated_tokens"]
            telemetry = {
                "tokens_per_second": res["tokens_per_second"],
                "time_to_first_token_ms": res["time_to_first_token_seconds"] * 1000,
            }
        else:
            # Fallback mock for offline scaffolding
            generated_text = "MemForge engine processed your request successfully using PagedAttention."
            num_tokens = 12
            telemetry = {"tokens_per_second": 45.2, "mock": True}

    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time()*1000)}",
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=generated_text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=len(prompt.split()),
            completion_tokens=num_tokens,
            total_tokens=len(prompt.split()) + num_tokens,
        ),
        memforge_telemetry=telemetry,
    )
