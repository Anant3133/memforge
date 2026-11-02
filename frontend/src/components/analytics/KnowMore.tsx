import React, { useState } from 'react';
import { Info, FileCode2 } from 'lucide-react';
import { Modal } from './Modal';
import { STAT_INFO } from './statInfo';
import type { ResultsPayload } from '../../services/results';

interface KnowMoreProps {
  /** Key into STAT_INFO. */
  id: string;
  data?: ResultsPayload | null;
  /** Extra "Our result" lines for this particular instance (e.g. the model shown). */
  extra?: Array<[string, string]>;
  label?: string;
}

/** A "Know more" button that opens a pop-up explaining one statistic in full. */
export const KnowMore: React.FC<KnowMoreProps> = ({ id, data, extra, label = 'Know more' }) => {
  const [open, setOpen] = useState(false);
  const info = STAT_INFO[id];
  if (!info) return null;
  const live = [...(extra ?? []), ...(data && info.live ? info.live(data) : [])];

  return (
    <>
      <button type="button" className="mf-know" onClick={() => setOpen(true)} aria-haspopup="dialog">
        <Info size={12} />
        {label}
      </button>
      {open && (
        <Modal title={info.title} onClose={() => setOpen(false)}>
          <div className="mf-modal-head">
            <h2>{info.title}</h2>
            <p>{info.tagline}</p>
          </div>
          <div className="mf-modal-body">
            <section><h3>What it is</h3><p>{info.what}</p></section>
            <section><h3>Why we track it</h3><p>{info.why}</p></section>
            <section><h3>How it is measured</h3><p>{info.how}</p></section>
            <section><h3>How to read it</h3><p>{info.read}</p></section>
            {live.length > 0 && (
              <section>
                <h3>Our result</h3>
                <dl className="mf-modal-live">
                  {live.map(([k, v], i) => (
                    <React.Fragment key={i}>
                      <dt>{k}</dt>
                      <dd>{v}</dd>
                    </React.Fragment>
                  ))}
                </dl>
              </section>
            )}
            {info.caveats.length > 0 && (
              <section>
                <h3>Caveats</h3>
                <ul>{info.caveats.map((c, i) => <li key={i}>{c}</li>)}</ul>
              </section>
            )}
            <section>
              <h3>Where to look</h3>
              <ul className="mf-files">
                {info.files.map((f) => <li key={f}><FileCode2 size={12} /> <code>{f}</code></li>)}
              </ul>
            </section>
          </div>
        </Modal>
      )}
    </>
  );
};
