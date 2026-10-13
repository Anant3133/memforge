import React from 'react';
import { Zap, Cpu, Layers, Activity } from 'lucide-react';

interface MetricsProps {
  tokensPerSec: number;
  /** Draft acceptance rate, or null when the last run was not speculative. */
  acceptanceRate: number | null;
  /** Fraction of the physical block pool currently handed out (0..1). */
  memoryUtilization: number;
  /** Mean tokens confirmed per model forward pass -- the batching payoff. */
  tokensPerForwardPass: number;
  allocatedBlocks: number;
  totalBlocks: number;
  blockSize: number;
  runningSequences: number;
  waitingSequences: number;
}

const card: React.CSSProperties = { padding: '20px' };
const label: React.CSSProperties = {
  color: 'var(--text-secondary)',
  fontSize: '0.85rem',
  fontWeight: 600,
};
const header: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: '8px',
};
const footnote: React.CSSProperties = {
  fontSize: '0.75rem',
  color: 'var(--text-muted)',
  marginTop: '4px',
};

function value(color: string): React.CSSProperties {
  return {
    fontSize: '1.8rem',
    fontWeight: 800,
    color,
    fontFamily: 'var(--font-mono)',
  };
}

export const TelemetryMetrics: React.FC<MetricsProps> = ({
  tokensPerSec,
  acceptanceRate,
  memoryUtilization,
  tokensPerForwardPass,
  allocatedBlocks,
  totalBlocks,
  blockSize,
  runningSequences,
  waitingSequences,
}) => {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '16px',
        marginBottom: '24px',
      }}
    >
      <div className="glass-card" style={card}>
        <div style={header}>
          <span style={label}>THROUGHPUT</span>
          <Zap size={18} color="var(--accent-emerald)" />
        </div>
        <div style={value('var(--accent-emerald)')}>
          {tokensPerSec.toFixed(1)}{' '}
          <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>tok/s</span>
        </div>
        <div style={footnote}>
          {runningSequences} running
          {waitingSequences > 0 ? `, ${waitingSequences} queued` : ''}
        </div>
      </div>

      <div className="glass-card" style={card}>
        <div style={header}>
          <span style={label}>TOKENS / FORWARD PASS</span>
          <Cpu size={18} color="var(--accent-amber)" />
        </div>
        <div style={value('var(--accent-amber)')}>{tokensPerForwardPass.toFixed(2)}</div>
        <div style={footnote}>
          {tokensPerForwardPass > 1 ? 'Batched across sequences' : 'Single sequence'}
        </div>
      </div>

      <div className="glass-card" style={card}>
        <div style={header}>
          <span style={label}>PAGED KV BLOCKS</span>
          <Layers size={18} color="var(--accent-purple)" />
        </div>
        <div style={value('var(--accent-purple)')}>
          {allocatedBlocks}{' '}
          <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>/ {totalBlocks}</span>
        </div>
        <div style={footnote}>
          {blockSize} tokens/block &middot; {(memoryUtilization * 100).toFixed(1)}% in use
        </div>
      </div>

      <div className="glass-card" style={card}>
        <div style={header}>
          <span style={label}>DRAFT ACCEPTANCE</span>
          <Activity size={18} color="var(--accent-cyan)" />
        </div>
        <div style={value('var(--accent-cyan)')}>
          {acceptanceRate === null ? (
            <span style={{ fontSize: '1.1rem', color: 'var(--text-muted)' }}>not speculative</span>
          ) : (
            `${(acceptanceRate * 100).toFixed(0)}%`
          )}
        </div>
        <div style={footnote}>
          {acceptanceRate === null
            ? 'Enable speculation to measure'
            : 'Rejection sampling keeps output lossless'}
        </div>
      </div>
    </div>
  );
};
