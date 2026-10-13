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
}

export interface EngineTelemetry extends LiveMetrics {
  uptime_s: number;
  steps: number;
  forward_passes: number;
  tokens_generated: number;
  prefill_tokens: number;
  lifetime_tokens_per_second: number;
  active_sequences: number;
  block_size: number;
  kv_pool_mb: number;
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
  /** occupancy[i] is the sequence id owning physical block i, or 0 if free. */
  occupancy: number[];
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
    throw new Error(`API error: ${detail}`);
  }
  return response.json();
}
