import React, { useState } from 'react';
import { useInView, useWidth } from './hooks';
import type { Band } from '../../services/results';

export const PALETTE = ['var(--accent-cyan)', 'var(--accent-emerald)', 'var(--accent-purple)', 'var(--accent-amber)', 'var(--accent-red)'];

const fmt = (v: number, d = 2) => v.toFixed(d);

/** Round-number axis: a top and step such that ticks land on tidy values (0.2, 0.25, 0.5 ...), never 0.325. */
export function niceScale(maxVal: number): { top: number; ticks: number[] } {
  for (const step of [0.1, 0.2, 0.25, 0.5, 1, 2, 5]) {
    const top = Math.ceil((maxVal * 1.04) / step) * step;
    if (top / step <= 6) {
      return { top, ticks: Array.from({ length: Math.round(top / step) + 1 }, (_, i) => +(i * step).toFixed(4)) };
    }
  }
  return { top: maxVal, ticks: [0, maxVal] };
}

/** y positions for reference-line labels: a label goes below its line when another line sits just above it. */
function labelOffsets(ys: number[]): number[] {
  return ys.map((y, i) => (ys.some((o, j) => j !== i && o < y && y - o < 16) ? 12 : -4));
}

/* ----------------------------------------------------------------- bars */

export interface BarSeries { name: string; color: string; values: Array<Band | null> }

interface GroupedBarsProps {
  groups: string[];
  series: BarSeries[];
  /** Reference lines, e.g. the 0.95 target. */
  lines?: Array<{ y: number; label: string; color: string }>;
  yMax?: number;
}

/** Vertical grouped bars with min-max whiskers and optional target lines. Responsive via viewBox. */
export const GroupedBars: React.FC<GroupedBarsProps> = ({ groups, series, lines = [], yMax }) => {
  const [ref, seen] = useInView<HTMLDivElement>();
  const [wref, W] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{ g: number; s: number } | null>(null);
  const compact = W < 520;
  const H = compact ? 280 : 300, padL = compact ? 34 : 44, padR = 10, padT = 16, padB = compact ? 74 : 56;
  const { top, ticks } = niceScale(yMax ?? Math.max(1.2, ...series.flatMap((s) => s.values.map((b) => (b ? b.max : 0))), ...lines.map((l) => l.y)));
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const gw = plotW / Math.max(1, groups.length);
  const bw = Math.min(34, (gw * 0.78) / Math.max(1, series.length));
  const y = (v: number) => padT + plotH - (v / top) * plotH;
  const lineOff = labelOffsets(lines.map((l) => y(l.y)));

  return (
    <div ref={ref} className="mf-chart" role="img" aria-label={`Grouped bar chart: ${series.map((s) => s.name).join(', ')}`}>
      <div ref={wref} className="mf-chart-box">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={padL} x2={W - padR} y1={y(t)} y2={y(t)} className="mf-grid" />
            <text x={padL - 6} y={y(t) + 3} textAnchor="end" className="mf-axis">{t}</text>
          </g>
        ))}
        {groups.map((g, gi) => {
          const cx = padL + gi * gw + gw / 2;
          const x0 = cx - (bw * series.length) / 2;
          return (
            <g key={g}>
              {series.map((s, si) => {
                const b = s.values[gi];
                if (!b) return null;
                const h = Math.max(1, (b.median / top) * plotH);
                const x = x0 + si * bw;
                const active = hover?.g === gi && hover.s === si;
                return (
                  <g key={s.name} onMouseEnter={() => setHover({ g: gi, s: si })} onMouseLeave={() => setHover(null)}>
                    <rect
                      x={x + 1} width={bw - 2} rx={3}
                      y={seen ? y(b.median) : padT + plotH} height={seen ? h : 0}
                      fill={s.color} opacity={active ? 1 : 0.85}
                      style={{ transition: `y .7s cubic-bezier(.2,.8,.2,1) ${gi * 60 + si * 40}ms, height .7s cubic-bezier(.2,.8,.2,1) ${gi * 60 + si * 40}ms` }}
                    />
                    {b.n > 1 && seen && (
                      <g className="mf-whisker">
                        <line x1={x + bw / 2} x2={x + bw / 2} y1={y(b.min)} y2={y(b.max)} />
                        <line x1={x + bw / 2 - 3} x2={x + bw / 2 + 3} y1={y(b.max)} y2={y(b.max)} />
                        <line x1={x + bw / 2 - 3} x2={x + bw / 2 + 3} y1={y(b.min)} y2={y(b.min)} />
                      </g>
                    )}
                    {(active || series.length <= 2) && seen && (
                      <text x={x + bw / 2} y={y(b.max) - 5} textAnchor="middle" className="mf-barlabel">{fmt(b.median)}</text>
                    )}
                  </g>
                );
              })}
              {gw < 84 ? (
                <text transform={`translate(${cx + 4},${H - padB + 12}) rotate(-35)`} textAnchor="end" className="mf-axis mf-xlabel">
                  {g.replace(/-Instruct$/, '')}
                  <title>{g}</title>
                </text>
              ) : (
                <text x={cx} y={H - padB + 16} textAnchor="middle" className="mf-axis mf-xlabel">
                  {g.length > 22 ? g.slice(0, 21) + '…' : g}
                  <title>{g}</title>
                </text>
              )}
            </g>
          );
        })}
        {lines.map((l, li) => (
          <g key={l.label}>
            <line x1={padL} x2={W - padR} y1={y(l.y)} y2={y(l.y)} stroke={l.color} strokeDasharray="5 4" strokeWidth={1.3} />
            <text x={W - padR} y={y(l.y) + lineOff[li]} textAnchor="end" className="mf-axis" fill={l.color}>{l.label}</text>
          </g>
        ))}
      </svg>
      </div>
      {hover && (
        <div className="mf-tip" aria-live="polite">
          {(() => {
            const b = series[hover.s].values[hover.g]!;
            return `${series[hover.s].name} · ${groups[hover.g]}: ${fmt(b.median, 3)} (range ${fmt(b.min, 3)}–${fmt(b.max, 3)}, n=${b.n})`;
          })()}
        </div>
      )}
      <Legend items={series.map((s) => ({ name: s.name, color: s.color }))} />
    </div>
  );
};

export const Legend: React.FC<{ items: Array<{ name: string; color: string }> }> = ({ items }) => (
  <div className="mf-legend">
    {items.map((i) => (
      <span key={i.name}><i style={{ background: i.color }} />{i.name}</span>
    ))}
  </div>
);

/* ----------------------------------------------------------------- lines */

interface LineChartProps {
  xs: number[];
  series: Array<{ name: string; color: string; values: Array<Band | null> }>;
  refLine?: { y: number; label: string };
  xLabel: string;
}

export const LineChart: React.FC<LineChartProps> = ({ xs, series, refLine, xLabel }) => {
  const [ref, seen] = useInView<HTMLDivElement>();
  const [wref, W] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const compact = W < 520;
  const H = compact ? 250 : 280, padL = compact ? 34 : 44, padR = 14, padT = 16, padB = 46;
  const { top, ticks } = niceScale(Math.max(1.1, refLine?.y ?? 0, ...series.flatMap((s) => s.values.map((b) => (b ? b.max : 0)))));
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const px = (i: number) => padL + (xs.length === 1 ? plotW / 2 : (i / (xs.length - 1)) * plotW);
  const py = (v: number) => padT + plotH - (v / top) * plotH;

  return (
    <div ref={ref} className="mf-chart" role="img" aria-label={`Line chart over ${xLabel}`}>
      <div ref={wref} className="mf-chart-box">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={padL} x2={W - padR} y1={py(t)} y2={py(t)} className="mf-grid" />
            <text x={padL - 6} y={py(t) + 3} textAnchor="end" className="mf-axis">{t}</text>
          </g>
        ))}
        {xs.map((x, i) => (
          <text key={x} x={px(i)} y={H - padB + 16} textAnchor="middle" className="mf-axis">{x}</text>
        ))}
        <text x={padL + plotW / 2} y={H - 8} textAnchor="middle" className="mf-axis">{xLabel}</text>
        {refLine && (
          <g>
            <line x1={padL} x2={W - padR} y1={py(refLine.y)} y2={py(refLine.y)} stroke="var(--text-muted)" strokeDasharray="5 4" />
            <text x={W - padR} y={py(refLine.y) - 4} textAnchor="end" className="mf-axis">{refLine.label}</text>
          </g>
        )}
        {series.map((s, si) => {
          const pts = s.values.map((b, i) => (b ? [px(i), py(b.median)] : null));
          const d = pts.filter(Boolean).map((p, i) => `${i ? 'L' : 'M'}${p![0]},${p![1]}`).join(' ');
          return (
            <g key={s.name}>
              {d && (
                <path d={d} fill="none" stroke={s.color} strokeWidth={2.4} strokeLinejoin="round" strokeLinecap="round"
                  pathLength={1} className={seen ? 'mf-draw' : ''} style={{ strokeDasharray: 1, strokeDashoffset: seen ? 0 : 1, animationDelay: `${si * 120}ms` }} />
              )}
              {pts.map((p, i) => p && (
                <circle key={i} cx={p[0]} cy={p[1]} r={hover === i ? 5.5 : 3.8} fill={s.color}
                  className="mf-dot" style={{ opacity: seen ? 1 : 0, transitionDelay: `${300 + si * 120 + i * 80}ms` }} />
              ))}
            </g>
          );
        })}
        {xs.map((x, i) => (
          <rect key={x} x={px(i) - 24} y={padT} width={48} height={plotH} fill="transparent"
            onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} />
        ))}
      </svg>
      </div>
      {hover !== null && (
        <div className="mf-tip" aria-live="polite">
          {`${xLabel} ${xs[hover]}: `}
          {series.map((s) => (s.values[hover] ? `${s.name} ${fmt(s.values[hover]!.median, 3)}` : null)).filter(Boolean).join(' · ')}
        </div>
      )}
      <Legend items={series.map((s) => ({ name: s.name, color: s.color }))} />
    </div>
  );
};

/* ----------------------------------------------------------------- donut */

export const Donut: React.FC<{ parts: Array<{ name: string; value: number; color: string }> }> = ({ parts }) => {
  const [ref, seen] = useInView<HTMLDivElement>();
  const total = parts.reduce((a, p) => a + p.value, 0) || 1;
  const R = 54, C = 2 * Math.PI * R;
  let acc = 0;
  return (
    <div ref={ref} className="mf-donut">
      <svg viewBox="0 0 140 140" role="img" aria-label={parts.map((p) => `${p.name} ${p.value}`).join(', ')}>
        <circle cx={70} cy={70} r={R} fill="none" stroke="var(--border-color)" strokeWidth={16} />
        {parts.map((p) => {
          const len = (p.value / total) * C;
          const el = (
            <circle key={p.name} cx={70} cy={70} r={R} fill="none" stroke={p.color} strokeWidth={16}
              strokeDasharray={`${seen ? len : 0} ${C}`} strokeDashoffset={-acc} transform="rotate(-90 70 70)"
              style={{ transition: 'stroke-dasharray 1s cubic-bezier(.2,.8,.2,1)' }} />
          );
          acc += len;
          return el;
        })}
        <text x={70} y={68} textAnchor="middle" className="mf-donut-n">{total}</text>
        <text x={70} y={84} textAnchor="middle" className="mf-axis">runs checked</text>
      </svg>
      <ul>
        {parts.map((p) => (
          <li key={p.name}><i style={{ background: p.color }} /><b>{p.value}</b> {p.name}</li>
        ))}
      </ul>
    </div>
  );
};

/* ------------------------------------------------------------ progress */

export const Meter: React.FC<{ value: number; target: number; higher: boolean; max?: number }> = ({ value, target, higher, max }) => {
  const [ref, seen] = useInView<HTMLDivElement>();
  const top = max ?? Math.max(value, target) * 1.25;
  const meets = higher ? value >= target : value <= target;
  return (
    <div ref={ref} className="mf-meter" aria-hidden="true">
      <div className="mf-meter-fill" style={{ width: seen ? `${Math.min(100, (value / top) * 100)}%` : 0, background: meets ? 'var(--accent-emerald)' : 'var(--accent-amber)' }} />
      <div className="mf-meter-target" style={{ left: `${Math.min(100, (target / top) * 100)}%` }} />
    </div>
  );
};
