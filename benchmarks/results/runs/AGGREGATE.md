# Aggregated results across rounds

Median [min-max] over independent rounds. GPU numbers were taken with a third-party GPU miner running (see report).


## Phase A: paged / contiguous throughput

| dev | model | ctx | paged/contig | outputs identical |
|---|---|---:|---|---|
| cpu | qwen05 | 128 | 0.997 [0.997-0.997] n=1 | True |
| cpu | qwen05 | 512 | 0.987 [0.987-0.987] n=1 | True |
| cpu | qwen05 | 1024 | 0.969 [0.969-0.969] n=1 | True |
| cpu | qwen15 | 128 | 0.978 [0.978-0.978] n=1 | True |
| cpu | qwen15 | 512 | 0.979 [0.979-0.979] n=1 | True |
| cpu | qwen15 | 1024 | 0.956 [0.956-0.956] n=1 | True |
| cpu | qwen3b | 128 | 1.008 [1.008-1.008] n=1 | True |
| cpu | qwen3b | 512 | 0.963 [0.963-0.963] n=1 | True |
| cpu | qwen7b | 128 | 1.013 [1.013-1.013] n=1 | True |
| cpu | qwen7b | 512 | 1.032 [1.032-1.032] n=1 | True |
| cpu | smol360 | 128 | 0.959 [0.959-0.959] n=1 | True |
| cpu | smol360 | 512 | 0.990 [0.990-0.990] n=1 | True |
| cpu | smol360 | 1024 | 0.964 [0.964-0.964] n=1 | True |
| gpu | qwen05 | 128 | 0.852 [0.843-0.857] n=3 | True |
| gpu | qwen05 | 512 | 0.852 [0.852-0.854] n=3 | True |
| gpu | qwen05 | 1024 | 0.852 [0.851-0.855] n=3 | True |
| gpu | qwen05 | 2048 | 0.858 [0.857-0.862] n=3 | True |
| gpu | qwen15 | 128 | 0.861 [0.858-0.866] n=3 | True |
| gpu | qwen15 | 512 | 0.857 [0.856-0.858] n=3 | True |
| gpu | qwen15 | 1024 | 0.859 [0.856-0.861] n=3 | True |
| gpu | qwen15 | 2048 | 0.875 [0.873-0.879] n=3 | True |
| gpu | qwen3-8b | 128 | 0.886 [0.884-0.898] n=3 | True |
| gpu | qwen3-8b | 512 | 0.900 [0.885-0.900] n=3 | True |
| gpu | qwen3-8b | 1024 | 0.904 [0.898-0.916] n=3 | True |
| gpu | qwen3-8b | 2048 | 0.918 [0.916-0.939] n=3 | True |
| gpu | qwen3b | 128 | 0.864 [0.810-0.877] n=3 | True |
| gpu | qwen3b | 512 | 0.857 [0.827-0.872] n=3 | True |
| gpu | qwen3b | 1024 | 0.851 [0.843-0.875] n=3 | True |
| gpu | qwen3b | 2048 | 0.884 [0.857-0.889] n=3 | True |
| gpu | qwen7b | 128 | 0.861 [0.856-0.886] n=3 | True |
| gpu | qwen7b | 512 | 0.884 [0.877-0.898] n=3 | True |
| gpu | qwen7b | 1024 | 0.899 [0.894-0.929] n=3 | True |
| gpu | qwen7b | 2048 | 0.942 [0.938-0.963] n=3 | True |
| gpu | smol135 | 128 | 0.853 [0.847-0.857] n=3 | True |
| gpu | smol135 | 512 | 0.853 [0.850-0.855] n=3 | True |
| gpu | smol135 | 1024 | 0.851 [0.850-0.851] n=3 | True |
| gpu | smol135 | 2048 | 0.853 [0.852-0.853] n=3 | True |
| gpu | smol360 | 128 | 0.851 [0.849-0.851] n=3 | True |
| gpu | smol360 | 512 | 0.853 [0.851-0.856] n=3 | True |
| gpu | smol360 | 1024 | 0.852 [0.845-0.856] n=3 | True |
| gpu | smol360 | 2048 | 0.847 [0.846-0.855] n=3 | True |

## Phase B: prefix sharing

| dev | model | N | block ratio | prefill ratio | identical | leaked |
|---|---|---:|---:|---:|---|---:|
| cpu | qwen05 | 1 | 1.000 | 1.000 | True | 0 |
| cpu | qwen05 | 4 | 0.333 | 0.270 | True | 0 |
| cpu | qwen15 | 1 | 1.000 | 1.000 | True | 0 |
| cpu | qwen15 | 4 | 0.333 | 0.270 | True | 0 |
| cpu | qwen3b | 1 | 1.000 | 1.000 | True | 0 |
| cpu | qwen3b | 4 | 0.333 | 0.270 | True | 0 |
| cpu | qwen7b | 1 | 1.000 | 1.000 | True | 0 |
| cpu | qwen7b | 4 | 0.333 | 0.270 | True | 0 |
| cpu | smol360 | 1 | 1.000 | 1.000 | True | 0 |
| cpu | smol360 | 4 | 0.333 | 0.270 | True | 0 |
| gpu | qwen05 | 1 | 1.000 | 1.000 | True | 0 |
| gpu | qwen05 | 4 | 0.333 | 0.270 | True | 0 |
| gpu | qwen15 | 1 | 1.000 | 1.000 | True | 0 |
| gpu | qwen15 | 4 | 0.333 | 0.270 | True | 0 |
| gpu | qwen3-8b | 1 | 1.000 | 1.000 | True | 0 |
| gpu | qwen3-8b | 4 | 0.333 | 0.270 | True | 0 |
| gpu | qwen3b | 1 | 1.000 | 1.000 | True | 0 |
| gpu | qwen3b | 4 | 0.333 | 0.270 | True | 0 |
| gpu | qwen7b | 1 | 1.000 | 1.000 | True | 0 |
| gpu | qwen7b | 4 | 0.333 | 0.270 | True | 0 |
| gpu | smol135 | 1 | 1.000 | 1.000 | True | 0 |
| gpu | smol135 | 4 | 0.333 | 0.270 | True | 0 |
| gpu | smol360 | 1 | 1.000 | 1.000 | True | 0 |
| gpu | smol360 | 4 | 0.333 | 0.270 | True | 0 |

## Phase C: decode stall ratio (lower is better)

| dev | model | budget | fused | stall ratio |
|---|---|---:|---|---|
| cpu | qwen05 | 0 | False | 6.520 [6.520-6.520] n=1 |
| cpu | qwen05 | 4 | False | 1.760 [1.760-1.760] n=1 |
| cpu | qwen05 | 16 | False | 2.490 [2.490-2.490] n=1 |
| cpu | qwen05 | 64 | False | 4.890 [4.890-4.890] n=1 |
| cpu | qwen15 | 0 | False | 8.170 [8.170-8.170] n=1 |
| cpu | qwen15 | 4 | False | 1.730 [1.730-1.730] n=1 |
| cpu | qwen15 | 16 | False | 4.000 [4.000-4.000] n=1 |
| cpu | qwen15 | 64 | False | 5.980 [5.980-5.980] n=1 |
| cpu | qwen3b | 0 | False | 6.840 [6.840-6.840] n=1 |
| cpu | qwen3b | 4 | False | 1.800 [1.800-1.800] n=1 |
| cpu | qwen3b | 16 | False | 3.620 [3.620-3.620] n=1 |
| cpu | qwen3b | 64 | False | 5.390 [5.390-5.390] n=1 |
| cpu | qwen7b | 0 | False | 5.970 [5.970-5.970] n=1 |
| cpu | qwen7b | 4 | False | 1.780 [1.780-1.780] n=1 |
| cpu | qwen7b | 16 | False | 4.410 [4.410-4.410] n=1 |
| cpu | qwen7b | 64 | False | 6.120 [6.120-6.120] n=1 |
| cpu | smol360 | 0 | False | 6.740 [6.740-6.740] n=1 |
| cpu | smol360 | 4 | False | 2.010 [2.010-2.010] n=1 |
| cpu | smol360 | 16 | False | 4.430 [4.430-4.430] n=1 |
| cpu | smol360 | 64 | False | 8.810 [8.810-8.810] n=1 |
| gpu | qwen05 | 0 | False | 3.380 [3.210-3.520] n=3 |
| gpu | qwen05 | 4 | False | 2.250 [2.150-2.290] n=3 |
| gpu | qwen05 | 16 | False | 2.350 [2.140-2.480] n=3 |
| gpu | qwen05 | 64 | False | 2.200 [2.180-6.890] n=3 |
| gpu | qwen15 | 0 | False | 3.240 [3.030-3.430] n=3 |
| gpu | qwen15 | 4 | False | 2.170 [2.120-2.430] n=3 |
| gpu | qwen15 | 16 | False | 2.230 [2.220-2.250] n=3 |
| gpu | qwen15 | 64 | False | 2.280 [2.280-2.520] n=3 |
| gpu | qwen3-8b | 0 | False | 2.590 [2.560-3.180] n=3 |
| gpu | qwen3-8b | 4 | False | 2.300 [2.070-2.360] n=3 |
| gpu | qwen3-8b | 16 | False | 2.160 [2.070-2.220] n=3 |
| gpu | qwen3-8b | 64 | False | 2.200 [2.190-2.220] n=3 |
| gpu | qwen3b | 0 | False | 2.920 [2.720-3.270] n=3 |
| gpu | qwen3b | 4 | False | 2.150 [2.060-2.160] n=3 |
| gpu | qwen3b | 16 | False | 2.150 [2.100-2.400] n=3 |
| gpu | qwen3b | 64 | False | 2.200 [2.070-2.280] n=3 |
| gpu | qwen7b | 0 | False | 2.990 [2.800-3.260] n=3 |
| gpu | qwen7b | 4 | False | 2.200 [1.980-2.320] n=3 |
| gpu | qwen7b | 16 | False | 2.210 [2.050-2.270] n=3 |
| gpu | qwen7b | 64 | False | 2.210 [2.160-2.250] n=3 |
| gpu | smol135 | 0 | False | 3.390 [3.310-3.640] n=3 |
| gpu | smol135 | 4 | False | 2.470 [2.270-2.620] n=3 |
| gpu | smol135 | 16 | False | 2.290 [2.240-2.300] n=3 |
| gpu | smol135 | 64 | False | 2.180 [2.020-2.180] n=3 |
| gpu | smol360 | 0 | False | 3.060 [2.930-3.310] n=3 |
| gpu | smol360 | 4 | False | 2.190 [2.080-4.030] n=3 |
| gpu | smol360 | 16 | False | 2.380 [2.200-2.420] n=3 |
| gpu | smol360 | 64 | False | 2.300 [2.200-2.300] n=3 |

## Phase D: speculative speedup (>1 = win)

| dev | model | batch | speedup | lossless |
|---|---|---:|---|---|
| cpu | qwen15 | 1 | 0.654 [0.654-0.654] n=1 | True |
| cpu | qwen15 | 2 | 0.382 [0.382-0.382] n=1 | True |
| cpu | qwen15 | 4 | 0.283 [0.283-0.283] n=1 | True |
| cpu | qwen3b | 1 | 1.453 [1.453-1.453] n=1 | True |
| cpu | qwen3b | 2 | 0.592 [0.592-0.592] n=1 | True |
| cpu | qwen3b | 4 | 0.491 [0.491-0.491] n=1 | True |
| cpu | qwen7b | 1 | 1.159 [1.159-1.159] n=1 | True |
| cpu | qwen7b | 2 | 0.731 [0.731-0.731] n=1 | True |
| cpu | qwen7b | 4 | 0.580 [0.580-0.580] n=1 | True |
| cpu | smol360 | 1 | 0.983 [0.983-0.983] n=1 | True |
| cpu | smol360 | 2 | 0.389 [0.389-0.389] n=1 | True |
| cpu | smol360 | 4 | 0.302 [0.302-0.302] n=1 | True |
| gpu | qwen15 | 1 | 0.502 [0.500-0.505] n=3 | True |
| gpu | qwen15 | 2 | 0.475 [0.432-0.480] n=3 | True |
| gpu | qwen15 | 4 | 0.346 [0.344-0.347] n=3 | True |
| gpu | qwen3-8b | 1 | 0.720 [0.710-0.721] n=3 | True |
| gpu | qwen3-8b | 2 | 0.587 [0.582-0.631] n=3 | False |
| gpu | qwen3-8b | 4 | 0.388 [0.348-0.424] n=3 | True |
| gpu | qwen3b | 1 | 1.040 [0.961-1.049] n=3 | True |
| gpu | qwen3b | 2 | 0.663 [0.653-0.672] n=3 | True |
| gpu | qwen3b | 4 | 0.473 [0.455-0.488] n=3 | True |
| gpu | qwen7b | 1 | 0.597 [0.597-0.597] n=1 | True |
| gpu | qwen7b | 2 | 0.484 [0.484-0.484] n=1 | True |
| gpu | qwen7b | 4 | 0.348 [0.348-0.348] n=1 | True |
| gpu | smol360 | 1 | 0.687 [0.681-0.687] n=3 | True |
| gpu | smol360 | 2 | 0.577 [0.577-0.581] n=3 | True |
| gpu | smol360 | 4 | 0.386 [0.352-0.387] n=3 | True |
