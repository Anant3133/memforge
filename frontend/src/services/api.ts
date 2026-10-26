/**
 * REST client for the MemForge engine.
 *
 * Types here mirror the server responses in server/api/routes.py.
 */

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

/** Where a generated token came from, used to colour the speculative diff. */
export type TokenSource = 'target' | 'draft' | 'resample' | 'bonus';

export interface ChatTelemetry {
  mode: 'continuous_batching' | 'speculative';
  tokens_per_second: number;
  time_to_first_token_ms: number | null;
  /** Speculative mode only. */
  acceptance_rate?: number;
  drafted_tokens?: number;
  accepted_tokens?: number;
  target_forward_passes?: number;
  tokens_per_target_pass?: number;
  rolled_back_tokens?: number;
  token_sources?: TokenSource[];
  /** Continuous batching mode only. */
  num_preemptions?: number;
  engine?: LiveMetrics;
}

export interface ChatResponse {
  id: string;
  choices: {
    index: number;
    message: ChatMessage;
    finish_reason: string;
  }[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  memforge_telemetry?: ChatTelemetry;
}

/**
 * The engine is at capacity (HTTP 429). Distinct from a generic failure because
 * the caller should back off and retry rather than surface an error.
 */
export class EngineBusyError extends Error {
  readonly retryAfterSeconds: number;
  constructor(detail: string, retryAfterSeconds: number) {
    super(detail);
    this.name = 'EngineBusyError';
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

export interface LiveMetrics {
  tokens_per_second: number;
  tokens_per_forward_pass: number;
  allocated_blocks: number;
  free_blocks: number;
  total_blocks: number;
  memory_utilization: number;
  running_sequences: number;
  waiting_sequences: number;
  preemptions: number;
  /**
   * Prefix sharing. Non-optional on purpose: the server always sends these, and
   * zeroes them when sharing is disabled rather than omitting them, so the UI
   * never needs a `?? 0` fallback.
   */
  shared_blocks: number;
  memory_saved_mb: number;
  prefix_hit_rate: number;
  /** Admission control: requests in flight against the configured ceiling. */
  in_flight: number;
  max_concurrency: number;
}

export interface EngineTelemetry extends LiveMetrics {
  /**
   * Where the pool lives and what the device holds.
   *
   * The memory fields are `null` on CPU rather than 0, because torch has no CPU
   * allocator to ask. Zero would read as "measured, and it was nothing", which
   * is the kind of number that makes a dashboard untrustworthy.
   */
  device: string;
  dtype: string;
  cuda_available: boolean;
  device_name: string | null;
  pool_mb: number;
  capacity_tokens: number;
  num_blocks: number;
  allocated_mb: number | null;
  peak_memory_mb: number | null;
  reserved_mb: number | null;
  vram_total_mb: number | null;
  vram_free_mb: number | null;

  uptime_s: number;
  steps: number;
  forward_passes: number;
  tokens_generated: number;
  prefill_tokens: number;
  lifetime_tokens_per_second: number;
  active_sequences: number;
  block_size: number;
  kv_pool_mb: number;

  // ----------------------------------------------------- prefix sharing
  /** Block allocations avoided by sharing: sum(ref_count) - distinct blocks. */
  blocks_saved: number;
  /** Full blocks currently registered as shareable in the prefix cache. */
  cached_prefix_blocks: number;
  prefix_cache_hits: number;
  prefix_cache_misses: number;
  prefix_sharing_enabled: boolean;
  /** Prefill tokens skipped because their KV was already resident. */
  cached_prefix_tokens: number;
  /** Copy-on-write block copies performed. Expected to stay 0 in practice. */
  cow_copies: number;
  /**
   * Physical block id -> every sequence referencing it.
   *
   * This is the only field that can express a *shared* block, which is why it
   * exists alongside `occupancy`: that is a flat one-owner-per-block array and
   * cannot represent three sequences pointing at one block. Keys are strings
   * because they are JSON object keys; free blocks are absent rather than
   * present-and-empty.
   */
  block_owners: Record<string, number[]>;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded';
  service: string;
  engine_ready: boolean;
  speculative_available: boolean;
  load_error: string | null;
}

export interface BlockMap {
  block_size: number;
  total_blocks: number;
  allocated_blocks: number;
  /**
   * occupancy[i] is *one* sequence id owning physical block i, or 0 if free.
   * Kept for the existing memory heatmap; it cannot show sharing.
   */
  occupancy: number[];
  /** Share-aware view: block id -> all owning sequence ids. */
  block_owners: Record<string, number[]>;
  /** Blocks held by more than one sequence. */
  shared_blocks: number;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path} failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export function fetchHealth(): Promise<HealthResponse> {
  return getJson<HealthResponse>('/v1/health');
}

export function fetchTelemetry(): Promise<EngineTelemetry> {
  return getJson<EngineTelemetry>('/v1/engine/telemetry');
}

export function fetchBlockMap(): Promise<BlockMap> {
  return getJson<BlockMap>('/v1/engine/blocks');
}

export function fetchEngineStatus(): Promise<Record<string, unknown>> {
  return getJson<Record<string, unknown>>('/v1/engine/status');
}

export async function sendChatCompletion(
  messages: ChatMessage[],
  useSpeculative = false,
  maxTokens = 128
): Promise<ChatResponse> {
  const response = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages,
      use_speculative: useSpeculative,
      max_new_tokens: maxTokens,
    }),
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* keep the status text */
    }
    // 429 means the engine hit max_concurrency: the right response is to wait
    // and retry, not to show a failure, so it gets a distinguishable error type.
    if (response.status === 429) {
      throw new EngineBusyError(detail, Number(response.headers.get('Retry-After') ?? 1));
    }
    throw new Error(`API error: ${detail}`);
  }
  return response.json();
}

/**
 * Poll engine state over REST.
 *
 * The dashboard normally gets this pushed over `/ws/telemetry`, which is
 * cheaper and more timely. This is the fallback for when that socket is down --
 * a proxy that blocks WebSocket upgrades, or a reconnect in progress -- so the
 * memory map and gauges keep moving instead of freezing at their last value.
 */
export async function pollEngineState(): Promise<{
  telemetry: EngineTelemetry;
  blocks: BlockMap;
}> {
  const [telemetry, blocks] = await Promise.all([fetchTelemetry(), fetchBlockMap()]);
  return { telemetry, blocks };
}
