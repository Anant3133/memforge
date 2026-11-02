import React, { useState } from 'react';
import { BookOpen } from 'lucide-react';
import { Modal } from './analytics/Modal';
import './analytics/analytics.css';

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
 * Opens as a pop-up modal (the same one the Run Analytics page uses) rather than
 * expanding in place: prose that pushes the live numbers around the page competes
 * with them, and a modal can be dismissed with Esc or a click outside. Rendered as a real <button> with
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
      <button type="button" className="mf-know" onClick={() => setOpen(true)} aria-haspopup="dialog">
        <BookOpen size={12} />
        Know more
      </button>

      {open && (
        <Modal title={subject} onClose={() => setOpen(false)}>
          <div className="mf-modal-head">
            <h2>{subject}</h2>
          </div>
          <div className="mf-modal-body">
            {sections.map((section) => (
              <section key={section.heading}>
                <h3>{section.heading}</h3>
                {section.body.map((paragraph, i) => (
                  <p key={i}>{paragraph}</p>
                ))}
              </section>
            ))}
            {measured && measured.length > 0 && (
              <section>
                <h3>Measured on this project</h3>
                <dl className="mf-modal-live">
                  {measured.map((row) => (
                    <React.Fragment key={row.label}>
                      <dt>{row.label}</dt>
                      <dd>{row.value}</dd>
                    </React.Fragment>
                  ))}
                </dl>
              </section>
            )}
          </div>
        </Modal>
      )}
    </div>
  );
};
