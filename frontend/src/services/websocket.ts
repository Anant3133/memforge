export interface TokenEvent {
  type: 'token';
  token: string;
  is_draft: boolean;
  index: number;
  metrics: {
    tokens_per_second: number;
    gpu_utilization_pct: number;
    allocated_blocks: number;
    free_blocks: number;
    acceptance_rate: number;
  };
}

export interface DoneEvent {
  type: 'done';
  total_tokens: number;
  total_time_ms: number;
}

export type StreamEvent = TokenEvent | DoneEvent;

export class StreamClient {
  private ws: WebSocket | null = null;

  connect(
    onMessage: (event: StreamEvent) => void,
    onError?: (err: Event) => void
  ): void {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/stream`;

    this.ws = new WebSocket(wsUrl);

    this.ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        onMessage(data);
      } catch (e) {
        console.error('Failed to parse WebSocket message', e);
      }
    };

    this.ws.onerror = (err) => {
      if (onError) onError(err);
    };
  }

  sendPrompt(prompt: string, maxTokens: number = 64): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ prompt, max_new_tokens: maxTokens }));
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}
