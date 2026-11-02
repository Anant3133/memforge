"""Per-token decode latency (batch 1, eager) per model: is a small draft actually cheap on this GPU?"""
import json, sys, time, torch
from transformers import AutoModelForCausalLM
out = {}
for name in sys.argv[2:]:
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).cuda().eval()
    ids = torch.randint(0, 1000, (1, 64), device="cuda")
    with torch.no_grad():
        o = m(ids, use_cache=True); pkv = o.past_key_values
        tok = ids[:, -1:]
        for _ in range(5): o = m(tok, past_key_values=pkv, use_cache=True)   # warmup
        torch.cuda.synchronize(); t = time.perf_counter(); n = 60
        for _ in range(n): o = m(tok, past_key_values=pkv, use_cache=True)
        torch.cuda.synchronize(); ms = (time.perf_counter() - t) / n * 1000
        # GPU-busy time of the same step, to separate launch-bound from compute-bound
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
            for _ in range(10): o = m(tok, past_key_values=pkv, use_cache=True)
            torch.cuda.synchronize()
        gpu_ms = sum(e.device_time_total for e in p.key_averages()) / 10 / 1000
    out[name] = {"layers": m.config.num_hidden_layers, "ms_per_token": round(ms, 2), "gpu_busy_ms": round(gpu_ms, 2)}
    print(name, out[name], flush=True)
    del m, o, pkv; torch.cuda.empty_cache()
json.dump(out, open(sys.argv[1], "w"), indent=2)
