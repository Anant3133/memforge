import React from 'react';
import { LearnMore } from './LearnMore';
import { SPECULATIVE_TOKENS } from './explanations';
import { Eye } from 'lucide-react';
import type { TokenSource } from '../services/api';

export interface StreamToken {
  token: string;
  source: TokenSource;
}

interface SpeculativeDiffProps {
  tokens: StreamToken[];
  /** Shown once a speculative run finishes. */
  acceptanceRate?: number | null;
  tokensPerTargetPass?: number | null;
}

/**
 * How each token provenance is rendered.
 *
 * draft    -- proposed by the draft model and accepted by the verifier
 * resample -- the draft was rejected; this came from the corrected distribution
 * bonus    -- the whole proposal was accepted, so verification yielded a free token
 * target   -- plain (non-speculative) decoding
 */
const SOURCE_STYLE: Record<TokenSource, { className: string; label: string }> = {
  draft: { className: 'token-draft', label: 'Draft accepted' },
  resample: { className: 'token-target', label: 'Rejected, resampled' },
  bonus: { className: 'token-draft', label: 'Bonus token' },
  target: { className: 'token-target', label: 'Target generated' },
};

const legendOrder: TokenSource[] = ['draft', 'resample', 'bonus', 'target'];

export const SpeculativeDiff: React.FC<SpeculativeDiffProps> = ({
  tokens,
  acceptanceRate,
  tokensPerTargetPass,
}) => {
  const counts = tokens.reduce<Partial<Record<TokenSource, number>>>((acc, t) => {
    acc[t.source] = (acc[t.source] ?? 0) + 1;
    return acc;
  }, {});

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
          <Eye size={18} color="var(--accent-emerald)" />
          <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>
            Speculative Verification Stream
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
          {legendOrder
            .filter((source) => counts[source])
            .map((source) => (
              <span key={source} className={SOURCE_STYLE[source].className} style={{ fontSize: '0.75rem' }}>
                {SOURCE_STYLE[source].label} ({counts[source]})
              </span>
            ))}
        </div>
      </div>

      <div
        style={{
          minHeight: '80px',
          maxHeight: '200px',
          overflowY: 'auto',
          padding: '12px',
          backgroundColor: 'rgba(0,0,0,0.3)',
          borderRadius: '8px',
          border: '1px solid var(--border-color)',
          lineHeight: 1.8,
          whiteSpace: 'pre-wrap',
        }}
      >
        {tokens.length === 0 ? (
          <span style={{ color: 'var(--text-muted)', fontStyle: 'italic', fontSize: '0.9rem' }}>
            Submit a prompt to watch tokens arrive. With speculation enabled, each token is
            coloured by whether the draft model proposed it or the target model corrected it.
          </span>
        ) : (
          tokens.map((item, idx) => (
            <span
              key={idx}
              className={SOURCE_STYLE[item.source].className}
              title={SOURCE_STYLE[item.source].label}
            >
              {item.token}
            </span>
          ))
        )}
      </div>

      {(acceptanceRate != null || tokensPerTargetPass != null) && (
        <div style={{ marginTop: '12px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          {acceptanceRate != null && (
            <span>acceptance {(acceptanceRate * 100).toFixed(0)}%</span>
          )}
          {acceptanceRate != null && tokensPerTargetPass != null && <span> &middot; </span>}
          {tokensPerTargetPass != null && (
            <span>{tokensPerTargetPass.toFixed(2)} tokens per target forward pass</span>
          )}
          <span> &middot; output is statistically identical to the target model alone</span>
        </div>
      )}
      <LearnMore {...SPECULATIVE_TOKENS} />
    </div>
  );
};
