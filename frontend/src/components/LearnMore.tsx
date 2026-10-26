import React, { useState } from 'react';
import { BookOpen, ChevronDown, ChevronRight } from 'lucide-react';

/**
 * An expandable explanation attached to a panel.
 *
 * Every number this dashboard shows is the output of a specific experiment with
 * a specific method, and a reader who does not know the method cannot tell a
 * good number from a bad one. `tokens / forward pass` reading 1.0 looks fine
 * until you know that 1.0 means nothing is being batched at all. So each panel
 * carries the method next to the measurement rather than in a document nobody
 * opens.
 *
 * Collapsed by default: the dashboard's job is the live numbers, and prose that
 * cannot be dismissed competes with them. Rendered as a real <button> with
 * aria-expanded so it is keyboard reachable and screen readers announce the
 * state, which a clickable <div> would not be.
 */

export interface LearnMoreSection {
  heading: string;
  /** Each entry renders as its own paragraph. */
  body: string[];
}

export interface LearnMoreProps {
  /** What this explains, e.g. "tokens / forward pass". */
  subject: string;
  sections: LearnMoreSection[];
  /** Figures measured on this project, shown as a definition list. */
  measured?: Array<{ label: string; value: string }>;
}

export const LearnMore: React.FC<LearnMoreProps> = ({ subject, sections, measured }) => {
  const [open, setOpen] = useState(false);

  return (
    <div style={{ marginTop: '10px' }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '5px',
          padding: '4px 9px',
          borderRadius: '6px',
          border: '1px solid var(--border-color)',
          background: 'transparent',
          color: 'var(--text-muted)',
          font: 'inherit',
          fontSize: '0.72rem',
          cursor: 'pointer',
        }}
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <BookOpen size={12} />
        Learn more
      </button>

      {open && (
        <div
          style={{
            marginTop: '8px',
            padding: '12px 14px',
            borderRadius: '8px',
            border: '1px solid var(--border-color)',
            background: 'var(--bg-primary)',
            fontSize: '0.78rem',
            lineHeight: 1.6,
            color: 'var(--text-secondary)',
          }}
        >
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.72rem',
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
              color: 'var(--text-muted)',
              marginBottom: '10px',
            }}
          >
            {subject}
          </div>

          {sections.map((section) => (
            <div key={section.heading} style={{ marginBottom: '12px' }}>
              <div
                style={{
                  fontWeight: 700,
                  color: 'var(--text-primary)',
                  marginBottom: '3px',
                }}
              >
                {section.heading}
              </div>
              {section.body.map((paragraph, i) => (
                <p key={i} style={{ margin: '0 0 6px' }}>
                  {paragraph}
                </p>
              ))}
            </div>
          ))}

          {measured && measured.length > 0 && (
            <div
              style={{
                marginTop: '4px',
                paddingTop: '10px',
                borderTop: '1px solid var(--border-color)',
              }}
            >
              <div
                style={{
                  fontWeight: 700,
                  color: 'var(--text-primary)',
                  marginBottom: '6px',
                }}
              >
                Measured on this project
              </div>
              {measured.map((row) => (
                <div
                  key={row.label}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    gap: '12px',
                    padding: '2px 0',
                  }}
                >
                  <span>{row.label}</span>
                  <span
                    style={{
                      fontFamily: 'var(--font-mono)',
                      color: 'var(--text-primary)',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {row.value}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
