"""Can a CUDA graph make a 0.6B decode step cost its GPU time instead of its launch time?"""
import sys, time, torch
from transformers import AutoModelForCausalLM, StaticCache
name = sys.argv[1]
m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).cuda().eval()
B, L = 1, 512
cache = StaticCache(config=m.config, max_batch_size=B, max_cache_len=L, device="cuda", dtype=torch.float16)
ids = torch.randint(0, 1000, (B, 64), device="cuda")
pos = torch.arange(64, device="cuda")
with torch.no_grad():
    m(ids, past_key_values=cache, cache_position=pos, use_cache=True)
    tok = torch.zeros((B, 1), dtype=torch.long, device="cuda")
    cp = torch.tensor([64], device="cuda")
    def step():
        return m(tok, past_key_values=cache, cache_position=cp, use_cache=True).logits
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = step()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); t = time.perf_counter(); n = 100
    for _ in range(n):
        cp += 1; tok.copy_(out[:, -1].argmax(-1, keepdim=True)); g.replay()
    torch.cuda.synchronize()
print(name, "CUDA-graph ms/token:", round((time.perf_counter() - t) / n * 1000, 2))
