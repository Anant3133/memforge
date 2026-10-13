/**
 * WebSocket clients for the MemForge engine.
 *
 * `StreamClient`  -> /ws/stream, one event per generated token.
 * `TelemetryClient` -> /ws/telemetry, engine state pushed on a timer so the
 *                      memory heatmap keeps moving even while this tab is idle.
 *
 * Events mirror server/api/websocket.py.
 */

import type { EngineTelemetry, LiveMetrics, TokenSource } from './api';

export interface StartEvent {
  type: 'start';
  mode: 'continuous_batching' | 'speculative';
  prompt: string;
  max_new_tokens: number;
}

export interface TokenEvent {
  type: 'token';
  index: number;
  token: string;
  token_id: number;
  /** target | draft | resample | bonus */
  source: TokenSource;
  /** Convenience flag: the token was proposed by the draft model and accepted. */
  is_draft: boolean;
  seq_id?: number;
  round?: number;
  metrics?: LiveMetrics;
}

export interface DoneEvent {
  type: 'done';
  generated_tokens: number;
  finish_reason: string;
  tokens_per_second: number;
  total_time_s: number;
  time_to_first_token_ms: number | null;
  /** Speculative mode only. */
  acceptance_rate?: number;
  tokens_per_target_pass?: number;
  target_forward_passes?: number;
  rolled_back_tokens?: number;
  metrics?: LiveMetrics;
}

export interface ErrorEvent {
  type: 'error';
  message: string;
}

export type StreamEvent = StartEvent | TokenEvent | DoneEvent | ErrorEvent;

export interface TelemetryEvent {
  type: 'telemetry';
  telemetry: EngineTelemetry | null;
  last_step: Record<string, unknown> | null;
  /** occupancy[i] is the sequence id owning physical block i, or 0 if free. */
  occupancy?: number[];
  ready?: boolean;
  load_error?: string | null;
}

function socketUrl(path: string): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}${path}`;
}

export interface StreamRequest {
  prompt: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  useSpeculative?: boolean;
}

export class StreamClient {
  private ws: WebSocket | null = null;
  private pending: StreamRequest | null = null;

  connect(
    onMessage: (event: StreamEvent) => void,
    onError?: (err: Event) => void,
    onOpen?: () => void
  ): void {
    this.ws = new WebSocket(socketUrl('/ws/stream'));

    this.ws.onopen = () => {
      onOpen?.();
      // A prompt submitted before the socket finished opening is sent now
      // rather than silently dropped.
      if (this.pending) {
        const request = this.pending;
        this.pending = null;
        this.sendPrompt(request);
      }
    };

    this.ws.onmessage = (evt) => {
      try {
        onMessage(JSON.parse(evt.data) as StreamEvent);
      } catch (e) {
        console.error('failed to parse stream message', e);
      }
    };

    this.ws.onerror = (err) => onError?.(err);
  }

  get isOpen(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  sendPrompt(request: StreamRequest): void {
    const payload = JSON.stringify({
      prompt: request.prompt,
      max_new_tokens: request.maxTokens ?? 64,
      temperature: request.temperature ?? 0.7,
      top_p: request.topP ?? 0.9,
      use_speculative: request.useSpeculative ?? false,
    });

    if (this.isOpen) {
      this.ws!.send(payload);
    } else {
      this.pending = request;
    }
  }

  disconnect(): void {
    this.ws?.close();
    this.ws = null;
    this.pending = null;
  }
}

export class TelemetryClient {
  private ws: WebSocket | null = null;
  private reconnectTimer: number | null = null;
  private closedByUs = false;

  connect(onMessage: (event: TelemetryEvent) => void): void {
    this.closedByUs = false;
    this.ws = new WebSocket(socketUrl('/ws/telemetry'));

    this.ws.onmessage = (evt) => {
      try {
        onMessage(JSON.parse(evt.data) as TelemetryEvent);
      } catch (e) {
        console.error('failed to parse telemetry message', e);
      }
    };

    // The telemetry feed is ambient, so a dropped connection should heal itself
    // instead of leaving the dashboard frozen.
    this.ws.onclose = () => {
      if (!this.closedByUs && this.reconnectTimer === null) {
        this.reconnectTimer = window.setTimeout(() => {
          this.reconnectTimer = null;
          this.connect(onMessage);
        }, 1500);
      }
    };
  }

  disconnect(): void {
    this.closedByUs = true;
    if (this.reconnectTimer !== null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
  }
}
