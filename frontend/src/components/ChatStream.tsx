import React, { useState } from 'react';
import { Send, Sparkles, StopCircle } from 'lucide-react';

interface ChatStreamProps {
  onSend: (prompt: string, maxTokens: number) => void;
  isStreaming: boolean;
}

export const ChatStream: React.FC<ChatStreamProps> = ({ onSend, isStreaming }) => {
  const [prompt, setPrompt] = useState('');
  const [maxTokens, setMaxTokens] = useState(64);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || isStreaming) return;
    onSend(prompt, maxTokens);
  };

  return (
    <div className="glass-card" style={{ padding: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
        <Sparkles size={18} color="var(--accent-purple)" />
        <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>Prompt Dispatcher</span>
      </div>

      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '12px' }}>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Type an inference prompt (e.g. 'Explain PagedAttention memory layout in C++')..."
            rows={3}
            style={{
              width: '100%',
              backgroundColor: 'rgba(0,0,0,0.3)',
              border: '1px solid var(--border-color)',
              borderRadius: '8px',
              color: 'var(--text-primary)',
              padding: '12px',
              fontFamily: 'var(--font-sans)',
              fontSize: '0.95rem',
              resize: 'none',
              outline: 'none',
            }}
          />
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Max Tokens:</label>
            <input
              type="number"
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
              min={16}
              max={256}
              style={{
                width: '70px',
                backgroundColor: 'rgba(0,0,0,0.3)',
                border: '1px solid var(--border-color)',
                borderRadius: '6px',
                color: 'var(--text-primary)',
                padding: '6px 10px',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.85rem',
              }}
            />
          </div>

          <button
            type="submit"
            disabled={isStreaming || !prompt.trim()}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              backgroundColor: isStreaming ? 'var(--text-muted)' : 'var(--accent-emerald)',
              color: '#000',
              fontWeight: 700,
              padding: '10px 20px',
              borderRadius: '8px',
              border: 'none',
              cursor: isStreaming ? 'not-allowed' : 'pointer',
              transition: 'all 0.2s ease',
            }}
          >
            {isStreaming ? (
              <>
                <StopCircle size={16} />
                <span>Streaming...</span>
              </>
            ) : (
              <>
                <Send size={16} />
                <span>Execute Step</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};
