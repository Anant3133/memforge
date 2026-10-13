import React from 'react';
import { Database } from 'lucide-react';

interface MemoryVisualizerProps {
  /**
   * occupancy[i] is the sequence id that owns physical block i, or 0 if the
   * block is free. Colouring by owner is what makes paging visible: one
   * sequence's blocks are scattered across the pool rather than contiguous.
   */
  occupancy: number[];
  blockSize: number;
}

/** Stable per-sequence hue, so a sequence keeps its colour as it grows. */
const OWNER_COLORS = [
  'var(--accent-emerald)',
  'var(--accent-cyan)',
  'var(--accent-purple)',
  'var(--accent-amber)',
  '#ff6b9d',
  '#5eead4',
  '#a78bfa',
  '#fbbf24',
];

function colorFor(owner: number): string {
  return OWNER_COLORS[(owner - 1) % OWNER_COLORS.length];
}

export const MemoryVisualizer: React.FC<MemoryVisualizerProps> = ({
  occupancy,
  blockSize,
}) => {
  const allocated = occupancy.filter((owner) => owner !== 0).length;
  const owners = Array.from(new Set(occupancy.filter((owner) => owner !== 0))).sort(
    (a, b) => a - b
  );

  return (
    <div className="glass-card" style={{ padding: '20px', marginBottom: '24px' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '16px',
          flexWrap: 'wrap',
          gap: '8px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Database size={18} color="var(--accent-cyan)" />
          <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>
            PagedAttention Physical Memory Map
          </span>
          <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
            {allocated} / {occupancy.length} blocks &middot; {blockSize} tokens each
          </span>
        </div>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '14px',
            fontSize: '0.8rem',
            flexWrap: 'wrap',
          }}
        >
          {owners.length === 0 ? (
            <span style={{ color: 'var(--text-muted)' }}>pool idle</span>
          ) : (
            owners.map((owner) => (
              <div key={owner} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span
                  style={{
                    width: '10px',
                    height: '10px',
                    backgroundColor: colorFor(owner),
                    borderRadius: '2px',
                    display: 'inline-block',
                  }}
                />
                <span>seq #{owner}</span>
              </div>
            ))
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span
              style={{
                width: '10px',
                height: '10px',
                backgroundColor: 'rgba(255,255,255,0.08)',
                borderRadius: '2px',
                display: 'inline-block',
              }}
            />
            <span>free</span>
          </div>
        </div>
      </div>

      <div className="block-grid" style={{ maxHeight: '160px', overflowY: 'auto' }}>
        {occupancy.map((owner, idx) => (
          <div
            key={idx}
            className={`block-cell ${owner ? 'block-allocated' : 'block-free'}`}
            style={owner ? { backgroundColor: colorFor(owner), boxShadow: 'none' } : undefined}
            title={
              owner
                ? `Physical block #${idx}: owned by sequence #${owner}`
                : `Physical block #${idx}: free`
            }
          />
        ))}
      </div>
    </div>
  );
};
