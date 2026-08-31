import React from 'react';
import { Database } from 'lucide-react';

interface MemoryVisualizerProps {
  allocatedBlocks: number;
  totalBlocks?: number;
}

export const MemoryVisualizer: React.FC<MemoryVisualizerProps> = ({
  allocatedBlocks,
  totalBlocks = 128,
}) => {
  const blocks = Array.from({ length: totalBlocks }, (_, i) => i < allocatedBlocks);

  return (
    <div className="glass-card" style={{ padding: '20px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Database size={18} color="var(--accent-cyan)" />
          <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>PagedAttention Physical Memory Map</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '0.8rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '10px', height: '10px', backgroundColor: 'var(--accent-emerald)', borderRadius: '2px', display: 'inline-block' }}></span>
            <span>Allocated Block</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '10px', height: '10px', backgroundColor: 'rgba(255,255,255,0.08)', borderRadius: '2px', display: 'inline-block' }}></span>
            <span>Free Pool</span>
          </div>
        </div>
      </div>

      <div className="block-grid" style={{ maxHeight: '140px', overflowY: 'auto' }}>
        {blocks.map((isAllocated, idx) => (
          <div
            key={idx}
            className={`block-cell ${isAllocated ? 'block-allocated' : 'block-free'}`}
            title={`Physical Block #${idx}: ${isAllocated ? 'ALLOCATED (16 tokens active)' : 'FREE'}`}
          />
        ))}
      </div>
    </div>
  );
};
