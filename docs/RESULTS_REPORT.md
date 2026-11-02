# MemForge results report (A100 server)

Every figure is the median over independent runs with [min-max] and n. GPU timings were taken on a card shared with other tenants (100% utilisation from jobs that are not ours), so ratios between arms run back to back are meaningful; absolute tokens/s are not.


## 1. Batched CUDA-graph draft (speculative speedup vs plain decoding, K=3, fp16)

Speedup > 1 means speculation beats plain decoding. Arms are run back to back in the same round.

| pair | batch | eager draft | per-sequence graph | batched graph | **batched, one replay/round** | acceptance | output equivalent |
|---|---:|---|---|---|---|---|---|
| q3-8b-0.6b-k3 | 1 | 0.700 [0.698-0.703] n=3 | 0.782 [0.756-0.841] n=3 | 0.828 [0.811-0.875] n=3 | **n/a** | 0.545 | True |
| q3-8b-0.6b-k3 | 2 | 0.637 [0.636-0.640] n=3 | 0.729 [0.707-0.734] n=3 | 0.911 [0.894-0.914] n=3 | **n/a** | 0.633 | True |
| q3-8b-0.6b-k3 | 4 | 0.415 [0.411-0.416] n=3 | 0.505 [0.502-0.507] n=3 | 0.814 [0.806-0.826] n=3 | **n/a** | 0.496 | True |
| q3-8b-0.6b-k3 | 8 | 0.354 [0.352-0.354] n=3 | 0.434 [0.432-0.435] n=3 | 0.874 [0.835-0.877] n=3 | **n/a** | 0.520 | True |
| q7b-0.5b-k3 | 1 | 0.753 [0.747-0.776] n=3 | n/a | 0.830 [0.815-0.849] n=3 | **n/a** | 0.515 | True |
| q7b-0.5b-k3 | 2 | 0.567 [0.566-0.571] n=3 | n/a | 0.824 [0.822-0.825] n=3 | **n/a** | 0.427 | True |
| q7b-0.5b-k3 | 4 | 0.434 [0.432-0.438] n=3 | n/a | 0.775 [0.774-0.808] n=3 | **n/a** | 0.438 | True |
| q7b-0.5b-k3 | 8 | 0.381 [0.377-0.382] n=3 | n/a | 0.890 [0.876-0.894] n=3 | **n/a** | 0.456 | True |

## 2. Phase A with the paged decode step as a CUDA graph (fp16)

Target: paged within 5% of contiguous (>= 0.95). Three baselines are shown because the answer depends on which one you mean.

| model | ctx | eager paged / eager contiguous (before) | **graphed paged / eager contiguous** | graphed paged / graphed contiguous | output = eager contiguous |
|---|---:|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 128 | 0.765 [0.765-0.765] n=1 | **1.684 [1.684-1.684] n=1** | 0.645 [0.645-0.645] n=1 | True |
| Qwen2.5-0.5B-Instruct | 512 | 0.760 [0.760-0.760] n=1 | **1.254 [1.254-1.254] n=1** | 0.621 [0.621-0.621] n=1 | True |
| Qwen2.5-0.5B-Instruct | 1024 | 0.746 [0.746-0.746] n=1 | **1.139 [1.139-1.139] n=1** | 0.714 [0.714-0.714] n=1 | True |
| Qwen2.5-0.5B-Instruct | 2048 | 0.755 [0.755-0.755] n=1 | **0.940 [0.940-0.940] n=1** | 0.767 [0.767-0.767] n=1 | True |

## 3. Chunked prefill on CPU, repeated (worst decode stall / normal step; target <= 2.0)

| model | unchunked | budget 64 | budget 16 | **budget 4** | meets target at budget 4 in every run |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 8.61 [6.52-9.72] n=4 | 4.85 [3.95-4.97] n=4 | 3.00 [2.49-3.81] n=4 | **1.81 [1.65-1.93] n=4** | True |
| Qwen2.5-1.5B-Instruct | 8.46 [7.69-12.69] n=4 | 4.31 [2.07-5.98] n=4 | 2.84 [2.60-4.00] n=4 | **1.96 [1.73-2.10] n=4** | False |
| Qwen2.5-3B-Instruct | 7.42 [6.84-10.45] n=4 | 5.07 [3.35-5.39] n=4 | 2.86 [2.64-3.62] n=4 | **1.84 [1.80-2.05] n=4** | False |
| Qwen2.5-7B-Instruct | 6.34 [5.53-6.90] n=4 | 3.97 [2.83-6.12] n=4 | 2.80 [2.06-4.41] n=4 | **1.82 [1.70-2.59] n=4** | False |
| SmolLM2-360M-Instruct | 8.35 [6.74-8.82] n=4 | 5.71 [2.39-8.81] n=4 | 2.75 [2.36-4.43] n=4 | **2.10 [1.60-2.41] n=4** | False |
