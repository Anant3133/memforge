# MemForge results report (A100 server)

Every figure is the median over independent runs with [min-max] and n. GPU timings were taken on a card shared with other tenants (100% utilisation from jobs that are not ours), so ratios between arms run back to back are meaningful; absolute tokens/s are not.


## 1. Batched CUDA-graph draft (speculative speedup vs plain decoding, K=3, fp16)

Speedup > 1 means speculation beats plain decoding. Arms are run back to back in the same round.

| pair | batch | eager draft | per-sequence graph | batched graph | **batched, one replay/round** | acceptance | output equivalent |
|---|---:|---|---|---|---|---|---|
| q3-8b-0.6b | 1 | 0.700 [0.697-0.703] n=6 | 0.782 [0.756-0.841] n=3 | 0.821 [0.805-0.875] n=6 | **0.642 [0.641-0.653] n=3** | 0.545 | True |
| q3-8b-0.6b | 2 | 0.637 [0.636-0.640] n=6 | 0.729 [0.707-0.734] n=3 | 0.911 [0.888-0.914] n=6 | **0.671 [0.663-0.683] n=3** | 0.633 | True |
| q3-8b-0.6b | 4 | 0.413 [0.408-0.416] n=6 | 0.505 [0.502-0.507] n=3 | 0.816 [0.806-0.835] n=6 | **0.744 [0.732-0.751] n=3** | 0.496 | True |
| q3-8b-0.6b | 8 | 0.353 [0.352-0.357] n=6 | 0.434 [0.432-0.435] n=3 | 0.861 [0.835-0.877] n=6 | **0.737 [0.723-0.738] n=3** | 0.520 | True |
| q7b-0.5b | 1 | 0.753 [0.747-0.776] n=3 | n/a | 0.830 [0.815-0.849] n=3 | **0.688 [0.685-0.716] n=3** | 0.515 | True |
| q7b-0.5b | 2 | 0.567 [0.566-0.571] n=3 | n/a | 0.824 [0.822-0.825] n=3 | **0.716 [0.708-0.717] n=3** | 0.427 | True |
| q7b-0.5b | 4 | 0.434 [0.432-0.438] n=3 | n/a | 0.775 [0.774-0.808] n=3 | **0.681 [0.672-0.682] n=3** | 0.438 | True |
| q7b-0.5b | 8 | 0.381 [0.377-0.382] n=3 | n/a | 0.890 [0.876-0.894] n=3 | **0.851 [0.809-0.857] n=3** | 0.456 | True |

## 2. Phase A with the paged decode step as a CUDA graph (fp16)

Target: paged within 5% of contiguous (>= 0.95). Three baselines are shown because the answer depends on which one you mean.

| model | ctx | eager paged / eager contiguous (before) | **graphed paged / eager contiguous** | graphed paged / graphed contiguous | output = eager contiguous |
|---|---:|---|---|---|---|
| Qwen2.5-0.5B-Instruct (gather bucket 128) | 128 | 0.761 [0.759-0.765] n=3 | **1.669 [1.639-1.684] n=3** | 0.645 [0.645-0.646] n=3 | True |
| Qwen2.5-0.5B-Instruct (gather bucket 128) | 512 | 0.782 [0.760-0.783] n=3 | **1.404 [1.254-1.408] n=3** | 0.707 [0.621-0.708] n=3 | True |
| Qwen2.5-0.5B-Instruct (gather bucket 128) | 1024 | 0.784 [0.746-0.792] n=3 | **1.302 [1.139-1.324] n=3** | 0.713 [0.709-0.714] n=3 | True |
| Qwen2.5-0.5B-Instruct (gather bucket 128) | 2048 | 0.792 [0.755-0.800] n=3 | **1.054 [0.940-1.062] n=3** | 0.772 [0.767-0.779] n=3 | True |
| Qwen2.5-1.5B-Instruct (gather bucket 128) | 128 | 0.781 [0.780-0.781] n=3 | **1.293 [1.291-1.302] n=3** | 0.764 [0.754-0.772] n=3 | True |
| Qwen2.5-1.5B-Instruct (gather bucket 128) | 512 | 0.786 [0.775-0.789] n=3 | **1.244 [1.236-1.251] n=3** | 0.777 [0.773-0.778] n=3 | True |
| Qwen2.5-1.5B-Instruct (gather bucket 128) | 1024 | 0.787 [0.785-0.795] n=3 | **1.134 [1.123-1.169] n=3** | 0.749 [0.744-0.765] n=3 | True |
| Qwen2.5-1.5B-Instruct (gather bucket 128) | 2048 | 0.798 [0.794-0.798] n=3 | **1.007 [1.002-1.012] n=3** | 0.822 [0.805-0.825] n=3 | True |
| Qwen2.5-3B-Instruct (gather bucket 128) | 128 | 0.810 [0.809-0.813] n=3 | **1.259 [1.251-1.260] n=3** | 0.817 [0.817-0.818] n=3 | True |
| Qwen2.5-3B-Instruct (gather bucket 128) | 512 | 0.796 [0.792-0.818] n=3 | **1.175 [1.118-1.182] n=3** | 0.910 [0.842-0.913] n=3 | True |
| Qwen2.5-3B-Instruct (gather bucket 128) | 1024 | 0.821 [0.820-0.825] n=3 | **1.102 [1.052-1.131] n=3** | 0.867 [0.834-0.892] n=3 | True |
| Qwen2.5-3B-Instruct (gather bucket 128) | 2048 | 0.829 [0.825-0.840] n=3 | **0.921 [0.921-0.938] n=3** | 0.921 [0.919-0.929] n=3 | True |
| Qwen2.5-3B-Instruct (gather bucket 32) | 1024 | 0.818 [0.808-0.822] n=3 | **1.097 [1.038-1.125] n=3** | 0.877 [0.824-0.907] n=3 | True |
| Qwen2.5-3B-Instruct (gather bucket 32) | 2048 | 0.833 [0.824-0.840] n=3 | **0.963 [0.926-0.981] n=3** | 0.951 [0.909-0.970] n=3 | True |
| Qwen2.5-7B-Instruct (gather bucket 128) | 128 | 0.861 [0.857-0.862] n=3 | **1.050 [1.050-1.050] n=3** | 0.933 [0.926-0.934] n=3 | True |
| Qwen2.5-7B-Instruct (gather bucket 128) | 512 | 0.889 [0.879-0.909] n=3 | **1.053 [0.999-1.058] n=3** | 0.915 [0.899-0.939] n=3 | True |
| Qwen2.5-7B-Instruct (gather bucket 128) | 1024 | 0.908 [0.908-0.920] n=3 | **0.998 [0.984-1.006] n=3** | 0.939 [0.925-0.951] n=3 | True |
| Qwen2.5-7B-Instruct (gather bucket 128) | 2048 | 0.923 [0.923-0.928] n=3 | **0.925 [0.921-0.927] n=3** | 0.959 [0.954-0.962] n=3 | True |
| Qwen2.5-7B-Instruct (gather bucket 32) | 1024 | 0.914 [0.909-0.916] n=3 | **1.017 [0.999-1.019] n=3** | 0.959 [0.942-0.960] n=3 | True |
| Qwen2.5-7B-Instruct (gather bucket 32) | 2048 | 0.933 [0.925-0.942] n=3 | **0.931 [0.930-0.935] n=3** | 0.964 [0.958-0.965] n=3 | True |
| Qwen3-8B (gather bucket 128) | 128 | 0.895 [0.894-0.899] n=3 | **1.117 [1.114-1.123] n=3** | 0.934 [0.928-0.935] n=3 | True |
| Qwen3-8B (gather bucket 128) | 512 | 0.897 [0.897-0.898] n=3 | **1.033 [1.027-1.038] n=3** | 0.935 [0.930-0.940] n=3 | True |
| Qwen3-8B (gather bucket 128) | 1024 | 0.879 [0.875-0.880] n=3 | **0.967 [0.966-0.968] n=3** | 0.946 [0.945-0.949] n=3 | True |
| Qwen3-8B (gather bucket 128) | 2048 | 0.919 [0.918-0.920] n=3 | **0.909 [0.908-0.912] n=3** | 0.934 [0.933-0.935] n=3 | True |
| Qwen3-8B (gather bucket 32) | 1024 | 0.882 [0.878-0.902] n=3 | **0.970 [0.959-0.974] n=3** | 0.944 [0.943-0.947] n=3 | True |
| Qwen3-8B (gather bucket 32) | 2048 | 0.916 [0.914-0.921] n=3 | **0.920 [0.915-0.926] n=3** | 0.948 [0.945-0.949] n=3 | True |

## 3. Chunked prefill on CPU, repeated (worst decode stall / normal step; target <= 2.0)

| model | unchunked | budget 64 | budget 16 | **budget 4** | meets target at budget 4 in every run |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 8.61 [6.52-9.72] n=4 | 4.85 [3.95-4.97] n=4 | 3.00 [2.49-3.81] n=4 | **1.81 [1.65-1.93] n=4** | True |
| Qwen2.5-1.5B-Instruct | 8.46 [7.69-12.69] n=4 | 4.31 [2.07-5.98] n=4 | 2.84 [2.60-4.00] n=4 | **1.96 [1.73-2.10] n=4** | False |
| Qwen2.5-3B-Instruct | 7.42 [6.84-10.45] n=4 | 5.07 [3.35-5.39] n=4 | 2.86 [2.64-3.62] n=4 | **1.84 [1.80-2.05] n=4** | False |
| Qwen2.5-7B-Instruct | 6.34 [5.53-6.90] n=4 | 3.97 [2.83-6.12] n=4 | 2.80 [2.06-4.41] n=4 | **1.82 [1.70-2.59] n=4** | False |
| SmolLM2-360M-Instruct | 8.35 [6.74-8.82] n=4 | 5.71 [2.39-8.81] n=4 | 2.75 [2.36-4.43] n=4 | **2.10 [1.60-2.41] n=4** | False |
