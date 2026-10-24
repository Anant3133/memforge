import React, { useMemo, useState } from 'react';
import { Copy, Layers, TrendingDown } from 'lucide-react';

/**
 * Prefix sharing, made visible.
 *
 * Driven by `block_owners` (block id -> every sequence referencing it) and NOT
 * by `occupancy`. That distinction is the whole component: `occupancy` is a flat
 * one-owner-per-block array, so it physically cannot represent three sequences
 * pointing at the same physical block -- which is exactly the thing being
 * demonstrated. The existing MemoryVisualizer keeps using `occupancy` and shows
 * a different, still-useful thing: how scattered one sequence's blocks are.
 *
 * Measured on real weights, 8 concurrent requests sharing a 512-token system
 * prompt: 272 physical blocks down to 48. This is what that looks like.
 */

export interface PrefixVisualizerProps {
  /** Block id (as a string key) -> owning sequence ids. Free blocks are absent. */
  blockOwners: Record<string, number[]>;
  totalBlocks: number;
  blockSize: number;
  sharedBlocks: number;
  blocksSaved: number;
  memorySavedMb: number;
  /** 0..1 */
  prefixHitRate: number;
  cachedPrefixBlocks: number;
  enabled: boolean;
}

type CellState = 'free' | 'private' | 'shared';

interface Cell {
  index: number;
  state: CellState;
  owners: number[];
}

/** Stable per-sequence hue, matching MemoryVisualizer so colours agree across panels. */
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

function ownerColor(owner: number): string {
  return OWNER_COLORS[(owner - 1) % OWNER_COLORS.length];
}

const Stat: React.FC<{
  icon: React.ReactNode;
  label: string;
  value: string;
  hint?: string;
}> = ({ icon, label, value, hint }) => (
  <div
    title={hint}
    style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '4px',
      padding: '10px 14px',
      borderRadius: '10px',
      background: 'var(--bg-secondary)',
      border: '1px solid var(--border-color)',
      minWidth: '130px',
    }}
  >
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        color: 'var(--text-muted)',
        fontSize: '0.72rem',
        textTransform: 'uppercase',
        letterSpacing: '0.04em',
      }}
    >
      {icon}
      {label}
    </div>
    <div
      style={{
        fontFamily: 'var(--font-mono)',
        fontSize: '1.15rem',
        fontWeight: 700,
        color: 'var(--text-primary)',
      }}
    >
      {value}
    </div>
  </div>
);

export const PrefixVisualizer: React.FC<PrefixVisualizerProps> = ({
  blockOwners,
  totalBlocks,
  blockSize,
  sharedBlocks,
  blocksSaved,
  memorySavedMb,
  prefixHitRate,
  cachedPrefixBlocks,
  enabled,
}) => {
  const [hovered, setHovered] = useState<Cell | null>(null);

  const cells = useMemo<Cell[]>(() => {
    const out: Cell[] = [];
    for (let i = 0; i < totalBlocks; i += 1) {
      const owners = blockOwners[String(i)] ?? [];
      const state: CellState =
        owners.length === 0 ? 'free' : owners.length === 1 ? 'private' : 'shared';
      out.push({ index: i, state, owners });
    }
    return out;
  }, [blockOwners, totalBlocks]);

  const counts = useMemo(() => {
    let free = 0;
    let privateCount = 0;
    let shared = 0;
    for (const cell of cells) {
      if (cell.state === 'free') free += 1;
      else if (cell.state === 'private') privateCount += 1;
      else shared += 1;
    }
    return { free, private: privateCount, shared };
  }, [cells]);

  /**
   * A shared block is filled with its first owner's colour but outlined and
   * brightened, so it reads as "shared" at a glance in a grid of hundreds. The
   * feature being demonstrated should be the first thing the eye lands on.
   */
  function cellStyle(cell: Cell): React.CSSProperties {
    if (cell.state === 'free') {
      return {
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border-color)',
      };
    }
    if (cell.state === 'private') {
      return {
        background: ownerColor(cell.owners[0]),
        opacity: 0.45,
        border: '1px solid transparent',
      };
    }
    return {
      background: ownerColor(cell.owners[0]),
      opacity: 1,
      border: '2px solid var(--text-primary)',
      boxShadow: '0 0 8px var(--border-highlight)',
    };
  }

  const savingsRatio =
    blocksSaved + counts.private + counts.shared > 0
      ? (blocksSaved + counts.private + counts.shared) / (counts.private + counts.shared)
      : 1;

  return (
    <div className="glass-card" style={{ padding: '20px', marginBottom: '24px' }}>
      {/* ------------------------------------------------------------ header */}
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
          <Copy size={18} color="var(--accent-purple)" />
          <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>
            Prefix Sharing &mdash; Copy-on-Write Block Reuse
          </span>
          <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
            {counts.shared + counts.private} / {totalBlocks} blocks in use &middot; {blockSize}{' '}
            tokens each
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', fontSize: '0.78rem' }}>
          <LegendSwatch
            label={`shared (${counts.shared})`}
            style={{
              background: 'var(--accent-emerald)',
              border: '2px solid var(--text-primary)',
            }}
          />
          <LegendSwatch
            label={`private (${counts.private})`}
            style={{ background: 'var(--accent-emerald)', opacity: 0.45 }}
          />
          <LegendSwatch
            label={`free (${counts.free})`}
            style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}
          />
        </div>
      </div>

      {/* --------------------------------------------------- disabled notice */}
      {!enabled ? (
        <div
          style={{
            padding: '28px',
            textAlign: 'center',
            color: 'var(--text-muted)',
            fontSize: '0.88rem',
            background: 'var(--bg-secondary)',
            borderRadius: '10px',
            border: '1px dashed var(--border-color)',
          }}
        >
          Prefix sharing is disabled on this engine.
          <div style={{ fontSize: '0.78rem', marginTop: '6px' }}>
            Every sequence stores its own copy of its prompt&rsquo;s KV cache.
          </div>
        </div>
      ) : (
        <>
          {/* ------------------------------------------------------- stats */}
          <div
            style={{
              display: 'flex',
              gap: '10px',
              flexWrap: 'wrap',
              marginBottom: '18px',
            }}
          >
            <Stat
              icon={<TrendingDown size={12} />}
              label="memory saved"
              value={`${memorySavedMb.toFixed(2)} MB`}
              hint="KV bytes the pool would have spent without sharing, minus what it actually spent."
            />
            <Stat
              icon={<Layers size={12} />}
              label="blocks saved"
              value={String(blocksSaved)}
              hint="sum(ref_count) - distinct allocated blocks: allocations avoided by sharing."
            />
            {/*
              Two sources for this number: `sharedBlocks` is the allocator's
              own reference-count scan, which is authoritative, while
              counts.shared is derived from the block tables the engine
              reports. They should agree; if they ever do not, the discrepancy
              is worth surfacing rather than silently picking one, because it
              would mean the telemetry and the allocator disagree about who
              owns what.
            */}
            <Stat
              icon={<Copy size={12} />}
              label="shared blocks"
              value={
                sharedBlocks === counts.shared
                  ? String(sharedBlocks)
                  : `${sharedBlocks} (map: ${counts.shared})`
              }
              hint={
                sharedBlocks === counts.shared
                  ? 'Physical blocks referenced by more than one sequence.'
                  : 'Allocator and block map disagree on the shared count - please report this.'
              }
            />
            <Stat
              icon={<Layers size={12} />}
              label="hit rate"
              value={`${(prefixHitRate * 100).toFixed(1)}%`}
              hint="Prefix-cache lookups that found a reusable block."
            />
            <Stat
              icon={<Layers size={12} />}
              label="cached prefixes"
              value={String(cachedPrefixBlocks)}
              hint="Full blocks currently registered as shareable."
            />
            {blocksSaved > 0 && (
              <Stat
                icon={<TrendingDown size={12} />}
                label="effective"
                value={`${savingsRatio.toFixed(1)}x`}
                hint="Blocks that would have been needed, divided by blocks actually held."
              />
            )}
          </div>

          {/* -------------------------------------------------------- grid */}
          {counts.shared === 0 && counts.private === 0 ? (
            <div
              style={{
                padding: '28px',
                textAlign: 'center',
                color: 'var(--text-muted)',
                fontSize: '0.88rem',
                background: 'var(--bg-secondary)',
                borderRadius: '10px',
                border: '1px dashed var(--border-color)',
              }}
            >
              Pool is empty &mdash; no sequences in flight.
            </div>
          ) : (
            <>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(14px, 1fr))',
                  gap: '3px',
                }}
              >
                {cells.map((cell) => (
                  <div
                    key={cell.index}
                    onMouseEnter={() => setHovered(cell)}
                    onMouseLeave={() => setHovered(null)}
                    style={{
                      aspectRatio: '1 / 1',
                      borderRadius: '3px',
                      cursor: cell.state === 'free' ? 'default' : 'pointer',
                      transition: 'transform 0.12s ease',
                      ...cellStyle(cell),
                    }}
                  />
                ))}
              </div>

              {counts.shared === 0 && (
                <div
                  style={{
                    marginTop: '14px',
                    fontSize: '0.8rem',
                    color: 'var(--text-muted)',
                  }}
                >
                  No shared prefixes yet. Sharing needs two or more in-flight requests whose
                  prompts begin with the same 16-token block &mdash; a single request has nothing
                  to share with.
                </div>
              )}
            </>
          )}

          {/* ----------------------------------------------------- tooltip */}
          <div
            style={{
              marginTop: '14px',
              minHeight: '40px',
              padding: '10px 14px',
              borderRadius: '8px',
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-color)',
              fontSize: '0.8rem',
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-secondary)',
            }}
          >
            {hovered === null ? (
              <span style={{ color: 'var(--text-muted)' }}>
                Hover a block to see which sequences own it.
              </span>
            ) : hovered.state === 'free' ? (
              <>
                block <strong style={{ color: 'var(--text-primary)' }}>{hovered.index}</strong>{' '}
                &mdash; free
              </>
            ) : (
              <>
                block <strong style={{ color: 'var(--text-primary)' }}>{hovered.index}</strong>{' '}
                &mdash;{' '}
                {hovered.state === 'shared' ? (
                  <span style={{ color: 'var(--accent-emerald)' }}>
                    shared by {hovered.owners.length} sequences
                  </span>
                ) : (
                  <span>private</span>
                )}
                {': '}
                {hovered.owners.map((owner, i) => (
                  <span key={owner} style={{ color: ownerColor(owner) }}>
                    seq {owner}
                    {i < hovered.owners.length - 1 ? ', ' : ''}
                  </span>
                ))}
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
};

const LegendSwatch: React.FC<{ label: string; style: React.CSSProperties }> = ({
  label,
  style,
}) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
    <div style={{ width: '12px', height: '12px', borderRadius: '3px', ...style }} />
    <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
  </div>
);
