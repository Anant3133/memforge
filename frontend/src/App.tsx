import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Layers, Server, Cpu, AlertTriangle } from 'lucide-react';
import { TelemetryMetrics } from './components/TelemetryMetrics';
import { MemoryVisualizer } from './components/MemoryVisualizer';
import { SpeculativeDiff, StreamToken } from './components/SpeculativeDiff';
import { ChatStream } from './components/ChatStream';
import { StreamClient, TelemetryClient, StreamEvent, TelemetryEvent } from './services/websocket';
import { fetchHealth, fetchEngineStatus } from './services/api';

interface EngineInfo {
  targetModel: string;
  draftModel: string | null;
  device: string;
  speculativeAvailable: boolean;
}

export const App: React.FC = () => {
  // Live engine state, driven by the /ws/telemetry feed.
  const [tokensPerSec, setTokensPerSec] = useState(0);
  const [tokensPerForwardPass, setTokensPerForwardPass] = useState(0);
  const [occupancy, setOccupancy] = useState<number[]>([]);
  const [blockSize, setBlockSize] = useState(16);
  const [memoryUtilization, setMemoryUtilization] = useState(0);
  const [runningSequences, setRunningSequences] = useState(0);
  const [waitingSequences, setWaitingSequences] = useState(0);

  // Per-run state, driven by the /ws/stream feed.
  const [tokens, setTokens] = useState<StreamToken[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [acceptanceRate, setAcceptanceRate] = useState<number | null>(null);
  const [tokensPerTargetPass, setTokensPerTargetPass] = useState<number | null>(null);
  const [useSpeculative, setUseSpeculative] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [engine, setEngine] = useState<EngineInfo | null>(null);

  const streamRef = useRef<StreamClient | null>(null);

  // Ambient telemetry: keeps the memory map and gauges live even while idle.
  useEffect(() => {
    const telemetry = new TelemetryClient();
    telemetry.connect((event: TelemetryEvent) => {
      if (!event.telemetry) {
        if (event.load_error) setError(`engine failed to load: ${event.load_error}`);
        return;
      }
      setTokensPerSec(event.telemetry.tokens_per_second);
      setTokensPerForwardPass(event.telemetry.tokens_per_forward_pass);
      setMemoryUtilization(event.telemetry.memory_utilization);
      setRunningSequences(event.telemetry.running_sequences);
      setWaitingSequences(event.telemetry.waiting_sequences);
      setBlockSize(event.telemetry.block_size);
      if (event.occupancy) setOccupancy(event.occupancy);
    });
    return () => telemetry.disconnect();
  }, []);

  // Token stream.
  useEffect(() => {
    const client = new StreamClient();
    client.connect(
      (event: StreamEvent) => {
        switch (event.type) {
          case 'start':
            setTokens([]);
            setError(null);
            break;
          case 'token':
            setTokens((prev) => [...prev, { token: event.token, source: event.source }]);
            if (event.metrics) {
              setTokensPerSec(event.metrics.tokens_per_second);
              setTokensPerForwardPass(event.metrics.tokens_per_forward_pass);
              setMemoryUtilization(event.metrics.memory_utilization);
              setRunningSequences(event.metrics.running_sequences);
            }
            break;
          case 'done':
            setIsStreaming(false);
            setAcceptanceRate(event.acceptance_rate ?? null);
            setTokensPerTargetPass(event.tokens_per_target_pass ?? null);
            break;
          case 'error':
            setIsStreaming(false);
            setError(event.message);
            break;
        }
      },
      () => {
        setIsStreaming(false);
        setError('lost connection to the engine');
      }
    );
    streamRef.current = client;
    return () => client.disconnect();
  }, []);

  // Engine identity for the header.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [health, status] = await Promise.all([fetchHealth(), fetchEngineStatus()]);
        if (cancelled) return;
        const config = (status.config ?? {}) as Record<string, string>;
        const described = (status.engine ?? {}) as Record<string, string>;
        setEngine({
          targetModel: config.target_model ?? 'unknown',
          draftModel: health.speculative_available ? config.draft_model ?? null : null,
          device: described.device ?? 'unknown',
          speculativeAvailable: health.speculative_available,
        });
        if (health.load_error) setError(`engine failed to load: ${health.load_error}`);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleSendPrompt = useCallback(
    (prompt: string, maxTokens: number) => {
      setTokens([]);
      setAcceptanceRate(null);
      setTokensPerTargetPass(null);
      setError(null);
      setIsStreaming(true);
      streamRef.current?.sendPrompt({ prompt, maxTokens, useSpeculative });
    },
    [useSpeculative]
  );

  const badge: React.CSSProperties = {
    padding: '6px 14px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '0.8rem',
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '32px 24px' }}>
      <header
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '32px',
          flexWrap: 'wrap',
          gap: '16px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div
            style={{
              background: 'linear-gradient(135deg, var(--accent-cyan), var(--accent-emerald))',
              padding: '10px',
              borderRadius: '10px',
              boxShadow: '0 0 20px rgba(0, 255, 170, 0.4)',
            }}
          >
            <Cpu size={26} color="#000" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.6rem', fontWeight: 800, letterSpacing: '-0.5px' }}>
              Mem<span style={{ color: 'var(--accent-emerald)' }}>Forge</span>
            </h1>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              PagedAttention &middot; continuous batching &middot; speculative decoding
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
          <div className="glass-card" style={badge}>
            <Server size={14} color="var(--accent-emerald)" />
            <span style={{ color: 'var(--text-secondary)' }}>
              Target: {engine?.targetModel ?? 'loading...'}
              {engine ? ` (${engine.device})` : ''}
            </span>
          </div>
          <div className="glass-card" style={badge}>
            <Layers size={14} color="var(--accent-cyan)" />
            <span style={{ color: 'var(--text-secondary)' }}>
              Draft: {engine ? engine.draftModel ?? 'unavailable' : 'loading...'}
            </span>
          </div>
        </div>
      </header>

      {error && (
        <div
          className="glass-card"
          style={{
            padding: '12px 16px',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            borderColor: 'var(--accent-amber)',
          }}
        >
          <AlertTriangle size={16} color="var(--accent-amber)" />
          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{error}</span>
        </div>
      )}

      <TelemetryMetrics
        tokensPerSec={tokensPerSec}
        acceptanceRate={acceptanceRate}
        memoryUtilization={memoryUtilization}
        tokensPerForwardPass={tokensPerForwardPass}
        allocatedBlocks={occupancy.filter((owner) => owner !== 0).length}
        totalBlocks={occupancy.length}
        blockSize={blockSize}
        runningSequences={runningSequences}
        waitingSequences={waitingSequences}
      />

      <MemoryVisualizer occupancy={occupancy} blockSize={blockSize} />

      <SpeculativeDiff
        tokens={tokens}
        acceptanceRate={acceptanceRate}
        tokensPerTargetPass={tokensPerTargetPass}
      />

      <div style={{ marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <input
          id="speculative-toggle"
          type="checkbox"
          checked={useSpeculative}
          disabled={!engine?.speculativeAvailable || isStreaming}
          onChange={(e) => setUseSpeculative(e.target.checked)}
        />
        <label
          htmlFor="speculative-toggle"
          style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}
        >
          Use speculative decoding
          {engine && !engine.speculativeAvailable ? ' (no draft model loaded)' : ''}
        </label>
      </div>

      <ChatStream onSend={handleSendPrompt} isStreaming={isStreaming} />
    </div>
  );
};
