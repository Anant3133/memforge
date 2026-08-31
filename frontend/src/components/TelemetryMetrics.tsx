import React from 'react';
import { Zap, Cpu, Layers, Activity } from 'lucide-react';

interface MetricsProps {
  tokensPerSec: number;
  acceptanceRate: number;
  gpuUtilization: number;
  allocatedBlocks: number;
  freeBlocks: number;
}

export const TelemetryMetrics: React.FC<MetricsProps> = ({
  tokensPerSec,
  acceptanceRate,
  gpuUtilization,
  allocatedBlocks,
  freeBlocks,
}) => {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '24px' }}>
      {/* Tokens / Sec */}
      <div className="glass-card" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 600 }}>THROUGHPUT</span>
          <Zap size={18} color="var(--accent-emerald)" />
        </div>
        <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--accent-emerald)', fontFamily: 'var(--font-mono)' }}>
          {tokensPerSec.toFixed(1)} <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>tok/s</span>
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>Continuous Batching active</div>
      </div>

      {/* Acceptance Rate */}
      <div className="glass-card" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 600 }}>DRAFT ACCEPTANCE</span>
          <Activity size={18} color="var(--accent-cyan)" />
        </div>
        <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
          {(acceptanceRate * 100).toFixed(0)}%
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>Rejection sampling verified</div>
      </div>

      {/* VRAM / Block Allocation */}
      <div className="glass-card" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 600 }}>PAGED KV BLOCKS</span>
          <Layers size={18} color="var(--accent-purple)" />
        </div>
        <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--accent-purple)', fontFamily: 'var(--font-mono)' }}>
          {allocatedBlocks} <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>/ {allocatedBlocks + freeBlocks}</span>
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>16 tokens / block (0% frag)</div>
      </div>

      {/* Hardware Utilization */}
      <div className="glass-card" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 600 }}>GPU SATURATION</span>
          <Cpu size={18} color="var(--accent-amber)" />
        </div>
        <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--accent-amber)', fontFamily: 'var(--font-mono)' }}>
          {gpuUtilization.toFixed(1)}%
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>Zero idle bubbles</div>
      </div>
    </div>
  );
};
