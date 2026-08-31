import React, { useState, useEffect, useRef } from 'react';
import { Layers, Terminal, Server, Cpu } from 'lucide-react';
import { TelemetryMetrics } from './components/TelemetryMetrics';
import { MemoryVisualizer } from './components/MemoryVisualizer';
import { SpeculativeDiff, StreamToken } from './components/SpeculativeDiff';
import { ChatStream } from './components/ChatStream';
import { StreamClient, StreamEvent } from './services/websocket';

export const App: React.FC = () => {
  const [tokensPerSec, setTokensPerSec] = useState(0.0);
  const [acceptanceRate, setAcceptanceRate] = useState(0.82);
  const [gpuUtilization, setGpuUtilization] = useState(0.0);
  const [allocatedBlocks, setAllocatedBlocks] = useState(16);
  const [freeBlocks, setFreeBlocks] = useState(496);

  const [tokens, setTokens] = useState<StreamToken[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const clientRef = useRef<StreamClient | null>(null);

  useEffect(() => {
    const client = new StreamClient();
    client.connect((event: StreamEvent) => {
      if (event.type === 'token') {
        setTokens((prev) => [...prev, { token: event.token, is_draft: event.is_draft }]);
        setTokensPerSec(event.metrics.tokens_per_second);
        setGpuUtilization(event.metrics.gpu_utilization_pct);
        setAllocatedBlocks(event.metrics.allocated_blocks);
        setFreeBlocks(event.metrics.free_blocks);
        setAcceptanceRate(event.metrics.acceptance_rate);
      } else if (event.type === 'done') {
        setIsStreaming(false);
      }
    });

    clientRef.current = client;
    return () => client.disconnect();
  }, []);

  const handleSendPrompt = (prompt: string, maxTokens: number) => {
    setTokens([]);
    setIsStreaming(true);
    if (clientRef.current) {
      clientRef.current.sendPrompt(prompt, maxTokens);
    }
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '32px 24px' }}>
      {/* Header */}
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '32px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{
            background: 'linear-gradient(135deg, var(--accent-cyan), var(--accent-emerald))',
            padding: '10px',
            borderRadius: '10px',
            boxShadow: '0 0 20px rgba(0, 255, 170, 0.4)',
          }}>
            <Cpu size={26} color="#000" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.6rem', fontWeight: 800, letterSpacing: '-0.5px' }}>
              Mem<span style={{ color: 'var(--accent-emerald)' }}>Forge</span>
            </h1>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              High-Performance Custom LLM Inference Engine & Memory Allocator
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <div className="glass-card" style={{ padding: '6px 14px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem' }}>
            <Server size={14} color="var(--accent-emerald)" />
            <span style={{ color: 'var(--text-secondary)' }}>Target: Llama-3-8B</span>
          </div>
          <div className="glass-card" style={{ padding: '6px 14px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem' }}>
            <Layers size={14} color="var(--accent-cyan)" />
            <span style={{ color: 'var(--text-secondary)' }}>Draft: Llama-3-1B</span>
          </div>
        </div>
      </header>

      {/* Real-time Telemetry Metrics */}
      <TelemetryMetrics
        tokensPerSec={tokensPerSec}
        acceptanceRate={acceptanceRate}
        gpuUtilization={gpuUtilization}
        allocatedBlocks={allocatedBlocks}
        freeBlocks={freeBlocks}
      />

      {/* Memory Heatmap Visualizer */}
      <MemoryVisualizer allocatedBlocks={allocatedBlocks} totalBlocks={128} />

      {/* Speculative Verification Diff Viewer */}
      <SpeculativeDiff tokens={tokens} />

      {/* Prompt Dispatcher */}
      <ChatStream onSend={handleSendPrompt} isStreaming={isStreaming} />
    </div>
  );
};
