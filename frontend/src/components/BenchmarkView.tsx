import React, { useMemo } from 'react';
import { Activity, Clock, Gauge, Timer } from 'lucide-react';

/**
 * Latency and throughput over time.
 *
 * The engine keeps no history -- every telemetry payload is a snapshot -- so the
 * series here are accumulated client-side by App.tsx and passed in. At the
 * 0.4 s telemetry cadence, 120 samples is roughly 48 seconds of window.
 *
 * Charts are hand-rolled inline SVG rather than a charting library, which is a
 * deliberate choice: adding a dependency means editing package.json and
 * package-lock.json, both shared files, and a line chart is a scale plus a
 * `points` string. MemoryVisualizer already establishes that hand-rolled
 * visuals work in this codebase.
 *
 * `tokens_per_forward_pass` is the metric worth watching most closely: 1.0 means
 * nothing is being batched, and 6.29 was measured at 8 concurrent requests. It
 * is the most direct evidence that continuous batching is doing its job.
 */

export interface TelemetrySample {
  /** performance.now() or Date.now(), only used for ordering and spacing. */
  t: number;
  tokensPerSecond: number;
  tokensPerForwardPass: number;
  runningSequences: number;
  memoryUtilization: number;
}

export interface RequestTiming {
  seqId: number;
  /** Time to first token, in milliseconds. */
  ttftMs: number;
  /** Mean gap between consecutive tokens, in milliseconds. */
  meanInterTokenMs: number | null;
  tokens: number;
}

export interface BenchmarkViewProps {
  samples: TelemetrySample[];
  timings: RequestTiming[];
  maxSamples?: number;
}

// ---------------------------------------------------------------- primitives

interface SparklineProps {
  values: number[];
  color: string;
  height?: number;
  /** Force the y-axis maximum; otherwise the series maximum is used. */
  yMax?: number;
  /** A reference line, e.g. 1.0 for "no batching". */
  baseline?: number;
  baselineLabel?: string;
}

/**
 * A filled line chart.
 *
 * The y-axis always starts at zero. Auto-scaling the minimum would make noise
 * look like a trend, which is exactly the kind of thing that makes a chart
 * misleading rather than informative.
 */
const Sparkline: React.FC<SparklineProps> = ({
  values,
  color,
  height = 64,
  yMax,
  baseline,
  baselineLabel,
}) => {
  const width = 100; // viewBox units; the SVG scales to its container
  const top = yMax ?? Math.max(...values, 0.0001);
  const scaleY = (v: number) => height - (Math.max(0, v) / (top || 1)) * height;

  if (values.length < 2) {
    return (
      <div
        style={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-muted)',
          fontSize: '0.75rem',
        }}
      >
        collecting&hellip;
      </div>
    );
  }

  const step = width / (values.length - 1);
  const points = values.map((v, i) => `${(i * step).toFixed(2)},${scaleY(v).toFixed(2)}`);
  const area = `0,${height} ${points.join(' ')} ${width},${height}`;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      style={{ width: '100%', height, display: 'block', overflow: 'visible' }}
      role="img"
      aria-label="time series"
    >
      {baseline !== undefined && baseline <= top && (
        <>
          <line
            x1={0}
            x2={width}
            y1={scaleY(baseline)}
            y2={scaleY(baseline)}
            stroke="var(--text-muted)"
            strokeWidth={0.4}
            strokeDasharray="2 2"
            vectorEffect="non-scaling-stroke"
          />
          {baselineLabel && (
            <text
              x={1}
              y={scaleY(baseline) - 2}
              fill="var(--text-muted)"
              style={{ fontSize: '5px', fontFamily: 'var(--font-mono)' }}
            >
              {baselineLabel}
            </text>
          )}
        </>
      )}
      <polygon points={area} fill={color} opacity={0.14} />
      <polyline
        points={points.join(' ')}
        fill="none"
        stroke={color}
        strokeWidth={1.6}
        strokeLinejoin="round"
        strokeLinecap="round"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
};

const ChartCard: React.FC<{
  icon: React.ReactNode;
  title: string;
  current: string;
  hint: string;
  children: React.ReactNode;
}> = ({ icon, title, current, hint, children }) => (
  <div
    style={{
      flex: '1 1 260px',
      minWidth: 0,
      padding: '14px',
      borderRadius: '10px',
      background: 'var(--bg-secondary)',
      border: '1px solid var(--border-color)',
    }}
  >
    <div
      style={{
        display: 'flex',
        alignItems: 'baseline',
        justifyContent: 'space-between',
        gap: '8px',
        marginBottom: '10px',
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
        title={hint}
      >
        {icon}
        {title}
      </div>
      <div
        style={{
          fontFamily: 'var(--font-mono)',
          fontWeight: 700,
          fontSize: '1.05rem',
          color: 'var(--text-primary)',
        }}
      >
        {current}
      </div>
    </div>
    {children}
  </div>
);

// ------------------------------------------------------------------ the view

function percentile(sorted: number[], p: number): number | null {
  if (sorted.length === 0) return null;
  if (sorted.length === 1) return sorted[0];
  const rank = (p / 100) * (sorted.length - 1);
  const low = Math.floor(rank);
  const high = Math.ceil(rank);
  if (low === high) return sorted[low];
  return sorted[low] + (sorted[high] - sorted[low]) * (rank - low);
}

export const BenchmarkView: React.FC<BenchmarkViewProps> = ({
  samples,
  timings,
  maxSamples = 120,
}) => {
  const window = useMemo(() => samples.slice(-maxSamples), [samples, maxSamples]);

  const series = useMemo(
    () => ({
      tps: window.map((s) => s.tokensPerSecond),
      tpfp: window.map((s) => s.tokensPerForwardPass),
      running: window.map((s) => s.runningSequences),
      util: window.map((s) => s.memoryUtilization * 100),
    }),
    [window]
  );

  const latest = window.length ? window[window.length - 1] : null;

  const ttftStats = useMemo(() => {
    const sorted = timings.map((t) => t.ttftMs).sort((a, b) => a - b);
    return {
      p50: percentile(sorted, 50),
      p95: percentile(sorted, 95),
      p99: percentile(sorted, 99),
      count: sorted.length,
    };
  }, [timings]);

  const itlMean = useMemo(() => {
    const values = timings
      .map((t) => t.meanInterTokenMs)
      .filter((v): v is number => v !== null);
    if (!values.length) return null;
    return values.reduce((a, b) => a + b, 0) / values.length;
  }, [timings]);

  const fmt = (v: number | null, unit = 'ms', digits = 0) =>
    v === null ? '--' : `${v.toFixed(digits)} ${unit}`;

  return (
    <div className="glass-card" style={{ padding: '20px', marginBottom: '24px' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          marginBottom: '16px',
          flexWrap: 'wrap',
        }}
      >
        <Gauge size={18} color="var(--accent-cyan)" />
        <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>
          Latency &amp; Throughput
        </span>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
          {window.length} samples &middot; {timings.length} completed request
          {timings.length === 1 ? '' : 's'}
        </span>
      </div>

      <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginBottom: '16px' }}>
        <ChartCard
          icon={<Activity size={12} />}
          title="tokens / forward pass"
          current={latest ? latest.tokensPerForwardPass.toFixed(2) : '--'}
          hint="The clearest evidence batching is working. 1.0 means no batching at all; 6.29 was measured at 8 concurrent requests."
        >
          <Sparkline
            values={series.tpfp}
            color="var(--accent-emerald)"
            baseline={1}
            baselineLabel="1.0 = no batching"
          />
        </ChartCard>

        <ChartCard
          icon={<Timer size={12} />}
          title="tokens / second"
          current={latest ? latest.tokensPerSecond.toFixed(1) : '--'}
          hint="Aggregate generation rate over a 5-second window."
        >
          <Sparkline values={series.tps} color="var(--accent-cyan)" />
        </ChartCard>

        <ChartCard
          icon={<Activity size={12} />}
          title="running sequences"
          current={latest ? String(latest.runningSequences) : '--'}
          hint="Sequences the scheduler is advancing this step."
        >
          <Sparkline values={series.running} color="var(--accent-purple)" />
        </ChartCard>

        <ChartCard
          icon={<Gauge size={12} />}
          title="KV pool used"
          current={latest ? `${(latest.memoryUtilization * 100).toFixed(1)}%` : '--'}
          hint="Fraction of physical blocks allocated."
        >
          <Sparkline values={series.util} color="var(--accent-amber)" yMax={100} />
        </ChartCard>
      </div>

      {/* --------------------------------------------- per-request latencies */}
      <div
        style={{
          display: 'flex',
          gap: '10px',
          flexWrap: 'wrap',
          alignItems: 'stretch',
        }}
      >
        <div
          style={{
            flex: '1 1 320px',
            padding: '14px',
            borderRadius: '10px',
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
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
              marginBottom: '10px',
            }}
          >
            <Clock size={12} />
            time to first token
          </div>

          <div style={{ display: 'flex', gap: '18px', flexWrap: 'wrap' }}>
            {(['p50', 'p95', 'p99'] as const).map((key) => (
              <div key={key}>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>{key}</div>
                <div
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 700,
                    fontSize: '1.05rem',
                    color: 'var(--text-primary)',
                  }}
                >
                  {fmt(ttftStats[key])}
                </div>
              </div>
            ))}
            <div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>
                mean inter-token
              </div>
              <div
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontWeight: 700,
                  fontSize: '1.05rem',
                  color: 'var(--text-primary)',
                }}
              >
                {fmt(itlMean, 'ms', 1)}
              </div>
            </div>
          </div>

          {ttftStats.count > 0 && ttftStats.count < 20 && (
            <div
              style={{
                marginTop: '10px',
                fontSize: '0.72rem',
                color: 'var(--text-muted)',
              }}
            >
              Only {ttftStats.count} sample{ttftStats.count === 1 ? '' : 's'} &mdash; p95 and p99
              are single data points at this count and should not be read as trends.
            </div>
          )}
        </div>

        {/* A horizontal bar per completed request, so an outlier is obvious. */}
        <div
          style={{
            flex: '1 1 320px',
            padding: '14px',
            borderRadius: '10px',
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
          }}
        >
          <div
            style={{
              color: 'var(--text-muted)',
              fontSize: '0.72rem',
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
              marginBottom: '10px',
            }}
          >
            per-request TTFT
          </div>

          {timings.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>
              No completed requests yet.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
              {timings.slice(-8).map((timing) => {
                const worst = Math.max(...timings.map((t) => t.ttftMs), 1);
                const pct = (timing.ttftMs / worst) * 100;
                return (
                  <div
                    key={timing.seqId}
                    style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
                    title={`${timing.tokens} tokens`}
                  >
                    <span
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.7rem',
                        color: 'var(--text-muted)',
                        width: '46px',
                      }}
                    >
                      seq {timing.seqId}
                    </span>
                    <div
                      style={{
                        flex: 1,
                        height: '10px',
                        borderRadius: '5px',
                        background: 'var(--bg-primary)',
                        overflow: 'hidden',
                      }}
                    >
                      <div
                        style={{
                          width: `${pct}%`,
                          height: '100%',
                          background: 'var(--accent-cyan)',
                          opacity: 0.75,
                        }}
                      />
                    </div>
                    <span
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.7rem',
                        color: 'var(--text-secondary)',
                        width: '62px',
                        textAlign: 'right',
                      }}
                    >
                      {timing.ttftMs.toFixed(0)} ms
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
