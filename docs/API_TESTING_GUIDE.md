# MemForge — API Testing Guide

**For:** anyone calling the MemForge server, by hand or from code
**Last verified:** 2026-10-25, against a live server

Every example here was executed against a running server before being written
down. Where a response surprised us, the surprise is documented rather than
smoothed over — see [§3.1](#31-v1completions-does-not-return-an-openai-envelope)
and [§6](#6-error-responses).

Start a server first:

```bash
source .venv/Scripts/activate
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

The first start takes 10-30 seconds to load the model. Interactive docs, with a
"try it" button for every endpoint, are at **http://127.0.0.1:8000/docs**.

---

## 1. Every endpoint

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/v1/health` | liveness, `engine_ready`, `load_error` |
| `GET` | `/v1/models` | available model ids |
| `GET` | `/v1/engine/status` | config, telemetry, admission state |
| `GET` | `/v1/engine/telemetry` | full engine metrics |
| `GET` | `/v1/engine/blocks` | physical block map, with sharing |
| `GET` | `/metrics` | Prometheus text exposition |
| `POST` | `/v1/completions` | raw-prompt completion |
| `POST` | `/v1/chat/completions` | OpenAI-compatible, `stream: true` for SSE |
| `WS` | `/ws/stream` | one message per generated token |
| `WS` | `/ws/telemetry` | engine state, pushed every 0.4 s |

---

## 2. Health and metadata

### 2.1 Health

```bash
curl http://127.0.0.1:8000/v1/health
```

```json
{
  "status": "healthy",
  "service": "MemForge LLM Inference Engine",
  "engine_ready": true,
  "speculative_available": false,
  "load_error": null,
  "timestamp": 1792868023.459
}
```

`status` is `"degraded"` when `load_error` is set. **Poll this before sending
work to a freshly started server**, because `engine_ready` is `false` while the
weights load:

```bash
until curl -s http://127.0.0.1:8000/v1/health | grep -q '"engine_ready":true'; do
  echo "waiting for the engine…"; sleep 1
done
```

### 2.2 Models

```bash
curl http://127.0.0.1:8000/v1/models
```

`memforge-speculative` only appears when a draft model is loaded, so this is how
you check whether `use_speculative` will work.

### 2.3 Engine status

```bash
curl -s http://127.0.0.1:8000/v1/engine/status | python -m json.tool
```

Returns `ready`, `load_error`, `speculative_available`, `in_flight`,
`at_capacity`, the whole `config` block and current `telemetry`. **The `config`
block is the authoritative answer to "what settings is this server actually
using?"** — more reliable than remembering which environment variables you
exported.

---

## 3. Completions

### 3.1 `/v1/completions` does not return an OpenAI envelope

```bash
curl -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Paged attention is", "max_tokens": 24}'
```

```json
{
  "type": "done",
  "seq_id": 60,
  "prompt_tokens": 4,
  "generated_tokens": 8,
  "text": " a crucial aspect of online marketing, as",
  "token_ids": [253, 3202, 2325, "..."],
  "token_sources": ["target", "target", "..."],
  "finish_reason": "length",
  "time_to_first_token_ms": 60.37,
  "total_time_s": 0.4893,
  "tokens_per_second": 16.35,
  "num_prefills": 1,
  "num_preemptions": 0,
  "metrics": { "…live engine metrics…" }
}
```

**This is the engine's raw result, not `{"choices": [...]}`.** Only
`/v1/chat/completions` is OpenAI-compatible. If you are writing a client that
expects `choices[0].message.content`, use that endpoint instead. This one is
useful precisely because it exposes the engine's own fields —
`num_preemptions`, `num_prefills`, `token_sources`.

Note also that **no chat template is applied** here. With an instruction-tuned
model that means the raw text goes straight in, which is what you want for
testing the engine and not what you want for a conversation.

### 3.2 `/v1/chat/completions`

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "messages": [{"role": "user", "content": "What is a KV cache?"}],
        "max_tokens": 32
      }'
```

```json
{
  "id": "chatcmpl-1792868123456",
  "object": "chat.completion",
  "created": 1792868123,
  "model": "memforge-target",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "A KV cache stores the key…"},
      "finish_reason": "length"
    }
  ],
  "usage": {"prompt_tokens": 42, "completion_tokens": 32, "total_tokens": 74},
  "memforge_telemetry": {
    "mode": "continuous_batching",
    "tokens_per_second": 18.7,
    "time_to_first_token_ms": 371.2,
    "num_preemptions": 0,
    "engine": { }
  }
}
```

`memforge_telemetry` is the non-standard addition — an OpenAI client will ignore
it, and it is where the interesting numbers are.

This endpoint **does** apply the model's chat template, which is why it produces
coherent output from a bare question where `/v1/completions` would not.

Multi-turn:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "messages": [
          {"role": "system",    "content": "You are concise."},
          {"role": "user",      "content": "What is paging?"},
          {"role": "assistant", "content": "Splitting memory into fixed blocks."},
          {"role": "user",      "content": "Why does that help?"}
        ],
        "max_tokens": 48
      }'
```

### 3.3 Request fields

| Field | Default | Notes |
| :--- | :--- | :--- |
| `messages` | required | chat endpoint only |
| `prompt` | required | completions endpoint only |
| `max_tokens` | 96 | also accepted as `max_new_tokens` on the chat endpoint |
| `temperature` | 0.7 | `0.0` for deterministic greedy output |
| `top_p` | 0.9 | |
| `seed` | null | set it for reproducible sampling |
| `stream` | false | chat endpoint only; switches to SSE |
| `use_speculative` | false | needs a draft model loaded |

For a reproducible answer, set both:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],
       "max_tokens":16,"temperature":0.0,"seed":42}'
```

---

## 4. Streaming

### 4.1 Server-Sent Events

`-N` disables curl's buffering; without it you see everything at once at the
end, which defeats the purpose.

```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Count to five"}],
       "max_tokens":48,"stream":true}'
```

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"memforge-target","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"...","choices":[{"index":0,"delta":{"content":"One","memforge_source":"target"},"finish_reason":null}]}

data: {"id":"...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Each chunk is `data: ` plus JSON, blank-line separated, ending with
`data: [DONE]`. `delta.memforge_source` is the non-standard field marking where
the token came from: `target`, `draft`, `resample` or `bonus`.

### 4.2 In Python

```python
import json
import httpx

with httpx.Client(timeout=300) as client:
    with client.stream(
        "POST", "http://127.0.0.1:8000/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Explain paging"}],
              "max_tokens": 48, "stream": True},
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if "error" in chunk:
                print("\nerror:", chunk["error"])
                break
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                print(delta["content"], end="", flush=True)
```

### 4.3 Abandoning a stream is a test, not a mistake

Closing the connection part-way through is a supported case and worth
exercising: the server should notice, abort the sequence, and return its KV
blocks. You can verify it did:

```bash
# Note blocks before, hang up mid-stream, then compare.
curl -s http://127.0.0.1:8000/metrics | grep memforge_blocks_allocated

curl -N -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Write a long essay"}],
       "max_tokens":500,"stream":true}' | head -c 200

sleep 2
curl -s http://127.0.0.1:8000/metrics | grep memforge_blocks_allocated
```

The second reading should match the first. If blocks stay allocated, that is a
real leak — report it.

---

## 5. WebSockets

### 5.1 `/ws/stream`

Send one JSON object, receive a stream of events.

**Request**

```json
{
  "prompt": "What is continuous batching?",
  "max_new_tokens": 32,
  "temperature": 0.7,
  "top_p": 0.9,
  "use_speculative": false,
  "raw": false
}
```

`raw: false` (the default) applies the model's chat template. Set `raw: true`
only if you are sending a fully-formed prompt and want to control the formatting
yourself — with an instruction-tuned model, a bare prompt sent raw tends to
produce an immediate end-of-sequence token and no answer.

**Events**

```json
{"type":"start","mode":"continuous_batching","prompt":"...","max_new_tokens":32}
{"type":"token","seq_id":1,"index":0,"token":"Continuous","token_id":9906,
 "source":"target","is_draft":false,"metrics":{ }}
{"type":"done","generated_tokens":32,"finish_reason":"length",
 "tokens_per_second":18.7,"time_to_first_token_ms":104.0,"metrics":{ }}
{"type":"error","message":"prompt is required"}
```

**Python**

```python
import asyncio, json
import websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8000/ws/stream") as ws:
        await ws.send(json.dumps({"prompt": "Define paged attention",
                                  "max_new_tokens": 32}))
        while True:
            event = json.loads(await ws.recv())
            if event["type"] == "token":
                print(event["token"], end="", flush=True)
            elif event["type"] == "done":
                print(f"\n{event['generated_tokens']} tokens, "
                      f"{event['tokens_per_second']} tok/s")
                break
            elif event["type"] == "error":
                print("error:", event["message"])
                break

asyncio.run(main())
```

The socket stays open after an answer, so you can send another prompt on the
same connection. It also stays open after an error — a malformed frame, a
missing prompt or bad sampling values are each reported and survivable, so a
dashboard does not have to reconnect.

### 5.2 `/ws/telemetry`

No request needed. Pushes engine state every 0.4 seconds.

```python
import asyncio, json
import websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8000/ws/telemetry") as ws:
        for _ in range(5):
            event = json.loads(await ws.recv())
            t = event.get("telemetry")
            if t is None:
                print("engine still loading:", event.get("load_error"))
                continue
            print(f"tok/s {t['tokens_per_second']:6.1f}  "
                  f"blocks {t['allocated_blocks']}/{t['total_blocks']}  "
                  f"shared {t['shared_blocks']}  "
                  f"saved {t['memory_saved_mb']} MB")

asyncio.run(main())
```

**Handle `telemetry: null`.** On a cold server the first payloads carry `ready`
and `load_error` instead of metrics, and a client that assumes metrics are
always present crashes on its first render.

---

## 6. Error responses

| Code | Meaning | Example trigger |
| :--- | :--- | :--- |
| `400` | well-formed request, unusable content | empty prompt; prompt longer than `max_model_len` |
| `422` | body does not match the schema | missing `messages`; `max_tokens: 0` |
| `429` | engine at capacity | more in-flight requests than `MEMFORGE_MAX_CONCURRENCY` |
| `499` | client disconnected before the response was ready | you hung up on a non-streaming request |
| `503` | engine failed to load | weights missing |

**400 and 422 are different on purpose.** 422 means the JSON does not match the
schema, which Pydantic rejects before the handler runs. 400 means the body is
valid but the values cannot be used. Assert the right one.

> **Characters are not tokens, and this trips people up when writing the
> too-long test.** `"x" * 10000` is 10,000 characters but only **2,500 tokens**
> to SmolLM2's BPE tokenizer, because it compresses repeats aggressively.
> `"x" * 4000` is just **1,000 tokens** — under the default 1024 limit, so it
> returns **200**, not the 400 you were testing for. Check first:
>
> ```python
> from transformers import AutoTokenizer
> t = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
> print(len(t("x" * 10000)["input_ids"]))   # 2500
> ```
>
> Note that the unit tests do not have this problem: they use a stub tokenizer
> that emits one token per character, so a length assertion that passes in
> `pytest` can still surprise you against a real model.

```bash
# 400 — too long for the context window
curl -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"$(python -c 'print("x"*10000)')\", \"max_tokens\": 4}"
# {"detail":"prompt of 2500 tokens leaves no room under max_model_len=1024"}

# 400 — nothing to generate from
curl -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" -d '{"prompt": "   ", "max_tokens": 4}'
# {"detail":"prompt must not be empty or whitespace-only"}

# 422 — schema violation
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" -d '{"max_tokens": 4}'
# {"detail":[{"type":"missing","loc":["body","messages"],"msg":"Field required",...}]}

# 422 — max_tokens must be >= 1
curl -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" -d '{"prompt": "hi", "max_tokens": 0}'
```

### 6.1 429 and backpressure

When more requests are in flight than `MEMFORGE_MAX_CONCURRENCY`, further
requests get a 429 with a `Retry-After` header. **This is the server working
correctly, not failing** — it would rather tell you to come back than hold your
connection open until it times out.

```bash
MEMFORGE_MAX_CONCURRENCY=2 uvicorn server.app:app --port 8000
# then fire several requests at once
```

A well-behaved client honours `Retry-After`:

```python
import time
import httpx

def with_retry(client, payload, attempts=5):
    for _ in range(attempts):
        response = client.post("http://127.0.0.1:8000/v1/chat/completions", json=payload)
        if response.status_code != 429:
            return response
        time.sleep(float(response.headers.get("Retry-After", 1)))
    raise RuntimeError("engine stayed saturated")
```

Both the streaming and non-streaming paths return a real 429. On the streaming
path there is a narrow race where saturation can instead arrive as an in-stream
chunk carrying `{"error": ..., "code": "engine_saturated", "retry_after": 1}`;
check for that code if you want to handle every case.

---

## 7. Telemetry and metrics

### 7.1 `/v1/engine/telemetry`

```bash
curl -s http://127.0.0.1:8000/v1/engine/telemetry | python -m json.tool
```

The fields worth knowing:

| Field | Means |
| :--- | :--- |
| `tokens_per_forward_pass` | **the batching indicator.** 1.0 = no batching; 6.29 measured at 8 concurrent requests |
| `total_blocks` / `allocated_blocks` / `free_blocks` | the KV pool. `total == free + allocated`, always |
| `shared_blocks` | blocks held by more than one sequence (prefix sharing) |
| `blocks_saved` | allocations avoided by sharing |
| `memory_saved_mb` | the same, in megabytes |
| `prefix_hit_rate` | 0..1 |
| `block_owners` | `{"0": [1,2,3]}` — block id to **all** owning sequences |
| `preemptions` | sequences evicted under memory pressure and recomputed |
| `cow_copies` | copy-on-write block copies. Expected to stay 0 |

Every field is always present. When prefix sharing is off they are zeroed rather
than omitted, so a client can bind to one shape.

### 7.2 Watching prefix sharing happen

Fire several requests with the same long prefix and watch `shared_blocks` climb:

```python
import asyncio, httpx

PREFIX = ("You are MemForge, a careful and precise assistant. Answer accurately, "
          "prefer concrete detail over generalities. ") * 6

async def main():
    async with httpx.AsyncClient(timeout=120) as client:
        task = asyncio.gather(*(
            client.post("http://127.0.0.1:8000/v1/chat/completions",
                        json={"messages": [{"role": "user",
                                            "content": PREFIX + f"Question {i}?"}],
                              "max_tokens": 10})
            for i in range(4)
        ))
        while not task.done():
            await asyncio.sleep(0.25)
            t = (await client.get("http://127.0.0.1:8000/v1/engine/telemetry")).json()
            if t["shared_blocks"]:
                print(f"shared {t['shared_blocks']} blocks, "
                      f"saved {t['memory_saved_mb']} MB, "
                      f"hit rate {t['prefix_hit_rate']:.3f}")
        await task

asyncio.run(main())
```

Two real runs of exactly this script, differing only in `MEMFORGE_KV_BUDGET_MB`:
one reported 15 shared blocks and 31.6 MB saved, the other 9 blocks and 19.0 MB.
**The absolute numbers depend on the pool size, the model's bytes-per-token and
how the prompt happens to divide into 16-token blocks**, so treat them as
illustrative — what should be stable is that `shared_blocks` is greater than
zero and `block_owners` shows the leading blocks held by every sequence at once.

### 7.3 `/v1/engine/blocks`

```bash
curl -s http://127.0.0.1:8000/v1/engine/blocks | python -m json.tool
```

`occupancy[i]` names **one** owner of block `i`, or `0` if free — a flat array
cannot express a shared block, which is why `block_owners` exists beside it.

### 7.4 `/metrics`

```bash
curl -s http://127.0.0.1:8000/metrics
```

Prometheus text: `# HELP` and `# TYPE` comments, then `name value` lines. Parse
by skipping `#` lines and splitting on the first space.

```python
def parse_metrics(text):
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        out[name] = float(value)
    return out
```

**`memforge_blocks_leaked` must always be 0.** It is
`total - free - allocated`, it has no warm-up behaviour, and a nonzero value is
a genuine leak rather than noise.

```bash
# The one-line health check for memory correctness
curl -s http://127.0.0.1:8000/metrics | grep memforge_blocks_leaked
```

---

## 8. Ready-made tools

Before writing your own client, these exist:

```bash
# Interactive chat with live TTFT and tok/s
python scripts/memforge_cli.py

# One-shot, scriptable
python scripts/memforge_cli.py --prompt "What is paging?" --tokens 32

# Load generation, with mid-stream disconnects
python scripts/traffic_simulator.py --concurrency 10 --requests 30
python scripts/traffic_simulator.py --concurrency 12 --requests 100 --disconnect-rate 0.25

# Leak gate: spawns its own server
python benchmarks/run_stress_test.py --spawn-server --requests 50 --concurrency 8

# Benchmark results to Markdown
python scripts/export_report.py --out benchmarks/results/REPORT.md
```

---

## 9. Quick reference

```bash
BASE=http://127.0.0.1:8000

curl $BASE/v1/health
curl $BASE/v1/models
curl -s $BASE/v1/engine/telemetry | python -m json.tool
curl -s $BASE/v1/engine/blocks    | python -m json.tool
curl -s $BASE/metrics | grep -v '^#'

curl -X POST $BASE/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":16}'

curl -X POST $BASE/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'

curl -N -X POST $BASE/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":16,"stream":true}'
```

Setup problems belong in
[TEAM_ONBOARDING.md](TEAM_ONBOARDING.md); configuration variables are listed in
[ASHUTOSH_SANSKAR_BHUMIKA_WORKPLAN.md](../ASHUTOSH_SANSKAR_BHUMIKA_WORKPLAN.md)
§3.5.
