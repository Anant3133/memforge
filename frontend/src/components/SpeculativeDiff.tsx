import React from 'react';
import { Eye } from 'lucide-react';

export interface StreamToken {
  token: string;
  is_draft: boolean;
}

interface SpeculativeDiffProps {
  tokens: StreamToken[];
}

export const SpeculativeDiff: React.FC<SpeculativeDiffProps> = ({ tokens }) => {
  return (
    <div className="glass-card" style={{ padding: '20px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Eye size={18} color="var(--accent-emerald)" />
          <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>Speculative Verification Stream</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '0.8rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span className="token-draft" style={{ fontSize: '0.75rem' }}>Draft Accepted</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span className="token-target" style={{ fontSize: '0.75rem' }}>Target Verified / Resampled</span>
          </div>
        </div>
      </div>

      <div style={{
        minHeight: '80px',
        maxHeight: '200px',
        overflowY: 'auto',
        padding: '12px',
        backgroundColor: 'rgba(0,0,0,0.3)',
        borderRadius: '8px',
        border: '1px solid var(--border-color)',
        lineHeight: 1.8,
      }}>
        {tokens.length === 0 ? (
          <span style={{ color: 'var(--text-muted)', fontStyle: 'italic', fontSize: '0.9rem' }}>
            Submit a prompt to inspect real-time speculative token verification...
          </span>
        ) : (
          tokens.map((item, idx) => (
            <span
              key={idx}
              className={item.is_draft ? 'token-draft' : 'token-target'}
            >
              {item.token}
            </span>
          ))
        )}
      </div>
    </div>
  );
};
