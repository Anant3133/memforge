import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Layers, Server, Cpu, AlertTriangle } from 'lucide-react';
import { TelemetryMetrics } from './components/TelemetryMetrics';
import { MemoryVisualizer } from './components/MemoryVisualizer';
import { SpeculativeDiff, StreamToken } from './components/SpeculativeDiff';
import { ChatStream } from './components/ChatStream';
import { PrefixVisualizer } from './components/PrefixVisualizer';
import { BenchmarkView, RequestTiming, TelemetrySample } from './components/BenchmarkView';
import { StatusBanner, ConnectionState } from './components/StatusBanner';
import { StreamClient, TelemetryClient, StreamEvent, TelemetryEvent } from './services/websocket';
import { fetchHealth, fetchEngineStatus } from './services/api';

/** Telemetry arrives every 0.4 s, so this is roughly a 48-second window. */
const MAX_SAMPLES = 120;

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

  // Prefix sharing.
  const [blockOwners, setBlockOwners] = useState<Record<string, number[]>>({});
  const [totalBlocks, setTotalBlocks] = useState(0);
  const [sharedBlocks, setSharedBlocks] = useState(0);
  const [blocksSaved, setBlocksSaved] = useState(0);
  const [memorySavedMb, setMemorySavedMb] = useState(0);
  const [prefixHitRate, setPrefixHitRate] = useState(0);
  const [cachedPrefixBlocks, setCachedPrefixBlocks] = useState(0);
  const [prefixSharingEnabled, setPrefixSharingEnabled] = useState(true);

  // Connection + capacity, for the status banner.
  const [connection, setConnection] = useState<ConnectionState>('connecting');
  const [engineReady, setEngineReady] = useState<boolean | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [inFlight, setInFlight] = useState<number | undefined>(undefined);
  const [maxConcurrency, setMaxConcurrency] = useState<number | undefined>(undefined);
  const [lastUpdate, setLastUpdate] = useState<number | null>(null);
  const [now, setNow] = useState(() => Date.now());

  // The engine stores no history, so the chart series are accumulated here.
  const [samples, setSamples] = useState<TelemetrySample[]>([]);
  const [timings, setTimings] = useState<RequestTiming[]>([]);

  // Per-run timing, measured client-side: the server's own TTFT excludes
  // transport, and inter-token latency is not reported at all.
  const runStartRef = useRef<number | null>(null);
  const firstTokenRef = useRef<number | null>(null);
  const lastTokenRef = useRef<number | null>(null);
  const interTokenGapsRef = useRef<number[]>([]);
  const runSeqRef = useRef<number>(0);

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
      setConnection('live');
      setLastUpdate(Date.now());

      // Before the model finishes loading, telemetry is null and ready /
      // load_error carry the state instead. That is a legitimate phase on a
      // cold server, not an error.
      if (!event.telemetry) {
        if (typeof event.ready === 'boolean') setEngineReady(event.ready);
        if (event.load_error) {
          setLoadError(event.load_error);
          setError(`engine failed to load: ${event.load_error}`);
        }
        return;
      }

      const t = event.telemetry;
      setEngineReady(true);
      setTokensPerSec(t.tokens_per_second);
      setTokensPerForwardPass(t.tokens_per_forward_pass);
      setMemoryUtilization(t.memory_utilization);
      setRunningSequences(t.running_sequences);
      setWaitingSequences(t.waiting_sequences);
      setBlockSize(t.block_size);
      if (event.occupancy) setOccupancy(event.occupancy);

      setBlockOwners(t.block_owners ?? {});
      setTotalBlocks(t.total_blocks);
      setSharedBlocks(t.shared_blocks);
      setBlocksSaved(t.blocks_saved);
      setMemorySavedMb(t.memory_saved_mb);
      setPrefixHitRate(t.prefix_hit_rate);
      setCachedPrefixBlocks(t.cached_prefix_blocks);
      setPrefixSharingEnabled(t.prefix_sharing_enabled);
      setInFlight(t.in_flight);
      setMaxConcurrency(t.max_concurrency);

      setSamples((prev) =>
        [
          ...prev,
          {
            t: Date.now(),
            tokensPerSecond: t.tokens_per_second,
            tokensPerForwardPass: t.tokens_per_forward_pass,
            runningSequences: t.running_sequences,
            memoryUtilization: t.memory_utilization,
          },
        ].slice(-MAX_SAMPLES)
      );
    });
    return () => telemetry.disconnect();
  }, []);

  // Drives the "last update Ns ago" staleness hint without re-rendering on
  // every frame.
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
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
          case 'token': {
            const at = performance.now();
            if (firstTokenRef.current === null) {
              firstTokenRef.current = at;
            } else if (lastTokenRef.current !== null) {
              // Inter-token latency is not reported by the server; it is the
              // gap between consecutive token events, measured here.
              interTokenGapsRef.current.push(at - lastTokenRef.current);
            }
            lastTokenRef.current = at;
            if (typeof event.seq_id === 'number') runSeqRef.current = event.seq_id;

            setTokens((prev) => [...prev, { token: event.token, source: event.source }]);
            if (event.metrics) {
              setTokensPerSec(event.metrics.tokens_per_second);
              setTokensPerForwardPass(event.metrics.tokens_per_forward_pass);
              setMemoryUtilization(event.metrics.memory_utilization);
              setRunningSequences(event.metrics.running_sequences);
              setInFlight(event.metrics.in_flight);
              setMaxConcurrency(event.metrics.max_concurrency);
            }
            break;
          }
          case 'done': {
            setIsStreaming(false);
            setAcceptanceRate(event.acceptance_rate ?? null);
            setTokensPerTargetPass(event.tokens_per_target_pass ?? null);

            // TTFT measured from the client, which is the latency a user
            // actually experiences; the server figure excludes transport.
            const start = runStartRef.current;
            const first = firstTokenRef.current;
            if (start !== null && first !== null) {
              const gaps = interTokenGapsRef.current;
              setTimings((prev) =>
                [
                  ...prev,
                  {
                    seqId: runSeqRef.current || prev.length + 1,
                    ttftMs: first - start,
                    meanInterTokenMs: gaps.length
                      ? gaps.reduce((a, b) => a + b, 0) / gaps.length
                      : null,
                    tokens: event.generated_tokens,
                  },
                ].slice(-50)
              );
            }
            runStartRef.current = null;
            firstTokenRef.current = null;
            lastTokenRef.current = null;
            interTokenGapsRef.current = [];
            break;
          }
          case 'error':
            setIsStreaming(false);
            setError(event.message);
            break;
        }
      },
      () => {
        setIsStreaming(false);
        setConnection('reconnecting');
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
        setEngineReady(health.engine_ready);
        setLoadError(health.load_error);
        if (health.load_error) setError(`engine failed to load: ${health.load_error}`);
      } catch (e) {
        if (!cancelled) {
          setConnection('disconnected');
          setError(e instanceof Error ? e.message : String(e));
        }
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
      runStartRef.current = performance.now();
      firstTokenRef.current = null;
      lastTokenRef.current = null;
      interTokenGapsRef.current = [];
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

      <StatusBanner
        connection={connection}
        engineReady={engineReady}
        loadError={loadError}
        inFlight={inFlight}
        maxConcurrency={maxConcurrency}
        secondsSinceUpdate={lastUpdate === null ? undefined : (now - lastUpdate) / 1000}
      />

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

      <PrefixVisualizer
        blockOwners={blockOwners}
        totalBlocks={totalBlocks || occupancy.length}
        blockSize={blockSize}
        sharedBlocks={sharedBlocks}
        blocksSaved={blocksSaved}
        memorySavedMb={memorySavedMb}
        prefixHitRate={prefixHitRate}
        cachedPrefixBlocks={cachedPrefixBlocks}
        enabled={prefixSharingEnabled}
      />

      <BenchmarkView samples={samples} timings={timings} />

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
