/** Types and client for GET /v1/results (server/results_data.py). */

export interface Band { median: number; min: number; max: number; n: number }

export interface ScoreItem {
  id: string; section: string; value: number; target: number;
  higher_is_better: boolean; meets: boolean; n: number; baseline: number | null;
}
export interface PhaseARow {
  model: string; context: number; bucket: number;
  eager_paged: Band | null; graphed_vs_stock: Band | null; graphed_vs_graphed: Band | null;
  output_identical: boolean;
}
export interface SpecRow {
  pair: string; batch: number; acceptance: Band | null;
  eager?: Band | null; perseq?: Band | null; batched?: Band | null; unrolled?: Band | null;
}
export interface ChunkRow { backend: string; model: string; budgets: Record<string, Band | null> }
export interface PrefixRow {
  backend: string; model: string; block_ratio: Band | null; prefill_ratio: Band | null;
  leaked: number; identical: boolean;
}
export interface Q14Side {
  context: Record<string, Band | null>; chunked: Record<string, Band | null>; output_identical: boolean | null;
}
export interface StressRow {
  label: string; model: string; passed: boolean; requests: number; succeeded: number;
  leaked_blocks: number; rss_growth_mb: number; p95_ttft_ms: number | null; peak_in_flight: number | null;
}
export interface InventoryRow {
  label: string; group: string; device: string; round: number | null; files: string[]; headline: string; mtime: number;
}
export interface ResultsPayload {
  generated_at: number;
  runs_dir: string;
  kpis: {
    run_directories: number; json_files: number; data_rows: number; models: number;
    targets_tracked: number; targets_met: number; hard_failures: number;
    cpu_tests_passed: number | null; gpu_tests_passed: number | null;
    best_paged_speedup: number | null; fp16_mismatches_explained: number; fp16_mismatches_unexplained: number;
  };
  models: string[];
  scoreboard: ScoreItem[];
  hard_failures: string[];
  phase_a: PhaseARow[];
  speculative: SpecRow[];
  chunked: ChunkRow[];
  prefix: PrefixRow[];
  qwen14b: { gpu_int8: Q14Side; cpu_fp32: Q14Side };
  equivalence: { identical: number; explained_ties: number; unexplained: number };
  stress: StressRow[];
  tests: Record<string, any>;
  inventory: InventoryRow[];
}

export async function fetchResults(signal?: AbortSignal): Promise<ResultsPayload> {
  const res = await fetch('/v1/results', { signal });
  if (!res.ok) throw new Error(`results endpoint returned ${res.status}`);
  return (await res.json()) as ResultsPayload;
}
