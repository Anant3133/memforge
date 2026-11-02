import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { RefreshCw, Search, CheckCircle2, AlertTriangle, Layers, Gauge, FlaskConical, ShieldCheck, Database, Cpu } from 'lucide-react';
import './analytics.css';
import { fetchResults, ResultsPayload, ScoreItem, Band } from '../../services/results';
import { KnowMore } from './KnowMore';
import { useCountUp } from './hooks';
import { Donut, GroupedBars, LineChart, Meter, PALETTE } from './Charts';

/* ------------------------------------------------------------------ helpers */

const f = (v: number | null | undefined, d = 2) => (v == null ? '—' : v.toFixed(d));
const bandText = (b: Band | null | undefined, d = 2) => (b ? `${b.median.toFixed(d)}` : '—');
const bandRange = (b: Band | null | undefined, d = 2) => (b ? `${b.min.toFixed(d)}–${b.max.toFixed(d)} · n=${b.n}` : '');

function humanize(id: string): string {
  const m = id.match(/^([^[]+)\[(.+)\]$/);
  if (!m) return id;
  const [, kind, rest] = m;
  const p = rest.split('|');
  switch (kind) {
    case 'phaseA.graphed_vs_stock':
      return `Paged decode vs stock · ${p[0]} · ${p[1]} tok · ${p[2] === 'b32' ? '32-token bucket' : '128-token bucket'}`;
    case 'prefix.block_ratio': return `Prefix sharing blocks · ${p[0]}`;
    case 'chunked.cpu_stall_b4': return `Decode stall (CPU, budget 4) · ${p[0]}`;
    case 'spec.batched_graph': return `Speculative speedup (batched graph) · ${p[0]} · ${p[1].replace('b', 'batch ')}`;
    default: return id;
  }
}

const SECTION_LABEL: Record<string, string> = {
  phaseA: 'Paged decode', prefix: 'Prefix sharing', chunked: 'Chunked prefill', spec: 'Speculative',
};

/* ------------------------------------------------------------------ pieces */

const Section: React.FC<{ id: string; title: string; icon: React.ReactNode; blurb: string; know?: React.ReactNode; children: React.ReactNode }> = ({ id, title, icon, blurb, know, children }) => (
  <section id={id} className="mf-section glass-card">
    <header className="mf-section-head">
      <div className="mf-section-title">
        <span className="mf-icon">{icon}</span>
        <div>
          <h2>{title}</h2>
          <p>{blurb}</p>
        </div>
      </div>
      {know}
    </header>
    {children}
  </section>
);

const Kpi: React.FC<{ label: string; value: number; decimals?: number; suffix?: string; sub?: string; tone?: 'good' | 'warn' | 'neutral'; know: React.ReactNode; delay: number }> = ({ label, value, decimals = 0, suffix = '', sub, tone = 'neutral', know, delay }) => {
  const v = useCountUp(value);
  return (
    <div className={`mf-kpi glass-card mf-tone-${tone}`} style={{ animationDelay: `${delay}ms` }}>
      <div className="mf-kpi-label">{label}</div>
      <div className="mf-kpi-value">{decimals ? v.toFixed(decimals) : Math.round(v).toLocaleString()}{suffix}</div>
      {sub && <div className="mf-kpi-sub">{sub}</div>}
      <div className="mf-kpi-know">{know}</div>
    </div>
  );
};

const Chips: React.FC<{ options: Array<{ value: string; label: string }>; value: string; onChange: (v: string) => void; label: string }> = ({ options, value, onChange, label }) => (
  <div className="mf-chips" role="group" aria-label={label}>
    {options.map((o) => (
      <button key={o.value} type="button" className={o.value === value ? 'on' : ''} aria-pressed={o.value === value} onClick={() => onChange(o.value)}>{o.label}</button>
    ))}
  </div>
);

/* ------------------------------------------------------------------ main */

export const ResultsDashboard: React.FC = () => {
  const [data, setData] = useState<ResultsPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(() => {
    const ctrl = new AbortController();
    setLoading(true);
    fetchResults(ctrl.signal)
      .then((d) => { setData(d); setError(null); })
      .catch((e) => { if (e.name !== 'AbortError') setError(e instanceof Error ? e.message : String(e)); })
      .finally(() => setLoading(false));
    return () => ctrl.abort();
  }, []);
  useEffect(() => load(), [load]);

  if (!data) {
    return (
      <div className="mf-analytics">
        <div className="mf-skeleton-grid" aria-busy={!error}>
          {Array.from({ length: 6 }, (_, i) => <div key={i} className="mf-skeleton glass-card" />)}
        </div>
        {error && (
          <div className="glass-card mf-error" role="alert">
            <AlertTriangle size={16} /> Could not load results: {error}. Start the server with <code>python -m server.app</code>.
            <button type="button" className="mf-btn" onClick={load}>Retry</button>
          </div>
        )}
      </div>
    );
  }
  return <Loaded data={data} reload={load} loading={loading} />;
};

const Loaded: React.FC<{ data: ResultsPayload; reload: () => void; loading: boolean }> = ({ data, reload, loading }) => {
  const k = data.kpis;
  return (
    <div className="mf-analytics">
      <div className="mf-hero">
        <div>
          <h2>Run Analytics</h2>
          <p>Every number below comes from an archived benchmark run on the A100 server. Medians with min–max ranges; the “Know more” buttons explain each statistic.</p>
        </div>
        <div className="mf-hero-actions">
          <span className="mf-updated">Updated {new Date(data.generated_at * 1000).toLocaleTimeString()}</span>
          <button type="button" className="mf-btn" onClick={reload} disabled={loading} aria-label="Refresh results">
            <RefreshCw size={14} className={loading ? 'mf-spin' : ''} /> Refresh
          </button>
          <KnowMore id="contention" data={data} label="Shared-GPU caveat" />
        </div>
      </div>

      <div className="mf-kpis">
        <Kpi delay={0} label="Archived runs" value={k.run_directories} sub={`${k.json_files} result files`} know={<KnowMore id="runs" data={data} />} />
        <Kpi delay={60} label="Data rows" value={k.data_rows} sub={`${k.models} models`} know={<KnowMore id="rows" data={data} />} />
        <Kpi delay={120} label="Targets met" value={k.targets_met} suffix={` / ${k.targets_tracked}`} tone={k.targets_met === k.targets_tracked ? 'good' : 'warn'} sub="known misses stay visible" know={<KnowMore id="targets" data={data} />} />
        <Kpi delay={180} label="Best paged speedup" value={k.best_paged_speedup ?? 0} decimals={2} suffix="×" tone="good" sub="vs stock decoding" know={<KnowMore id="best_speedup" data={data} />} />
        <Kpi delay={240} label="Tests passing" value={(k.cpu_tests_passed ?? 0) + (k.gpu_tests_passed ?? 0)} tone="good" sub={`${k.cpu_tests_passed} CPU · ${k.gpu_tests_passed} GPU`} know={<KnowMore id="tests" data={data} />} />
        <Kpi delay={300} label="Hard failures" value={k.hard_failures} tone={k.hard_failures === 0 ? 'good' : 'warn'} sub={`${k.fp16_mismatches_unexplained} unexplained mismatches`} know={<KnowMore id="hard_failures" data={data} />} />
      </div>

      <nav className="mf-jump" aria-label="Sections">
        {[['scoreboard', 'Scoreboard'], ['paged', 'Paged decode'], ['spec', 'Speculative'], ['chunk', 'Chunked prefill'], ['prefix', 'Prefix sharing'], ['q14', 'Qwen3-14B'], ['equiv', 'Equivalence'], ['stress', 'Soak'], ['inventory', 'Runs']].map(([id, label]) => (
          <a key={id} href={`#${id}`} onClick={(e) => { e.preventDefault(); document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' }); }}>{label}</a>
        ))}
      </nav>

      <Scoreboard data={data} />
      <PagedSection data={data} />
      <SpecSection data={data} />
      <ChunkSection data={data} />
      <PrefixSection data={data} />
      <Q14Section data={data} />
      <EquivSection data={data} />
      <StressSection data={data} />
      <InventorySection data={data} />
    </div>
  );
};

/* ---------------------------------------------------------------- scoreboard */

const Scoreboard: React.FC<{ data: ResultsPayload }> = ({ data }) => {
  const [filter, setFilter] = useState('all');
  const [section, setSection] = useState('all');
  const rows = useMemo(
    () => data.scoreboard.filter((s) => (filter === 'all' || (filter === 'met' ? s.meets : !s.meets)) && (section === 'all' || s.section === section)),
    [data, filter, section]
  );
  const sections = Array.from(new Set(data.scoreboard.map((s) => s.section)));
  return (
    <Section id="scoreboard" title="Target scoreboard" icon={<ShieldCheck size={18} />}
      blurb="Each tracked metric against the goal set before measuring. Misses stay on the board."
      know={<KnowMore id="gate" data={data} label="About the gate" />}>
      <div className="mf-toolbar">
        <Chips label="Status filter" value={filter} onChange={setFilter} options={[{ value: 'all', label: `All ${data.scoreboard.length}` }, { value: 'met', label: `Met ${data.kpis.targets_met}` }, { value: 'miss', label: `Missing ${data.kpis.targets_tracked - data.kpis.targets_met}` }]} />
        <Chips label="Area filter" value={section} onChange={setSection} options={[{ value: 'all', label: 'All areas' }, ...sections.map((s) => ({ value: s, label: SECTION_LABEL[s] ?? s }))]} />
      </div>
      <ul className="mf-score">
        {rows.map((s: ScoreItem, i) => (
          <li key={s.id} style={{ animationDelay: `${Math.min(i, 20) * 25}ms` }}>
            <div className="mf-score-main">
              <span className="mf-score-name" title={s.id}>{humanize(s.id)}</span>
              <span className={`mf-pill ${s.meets ? 'ok' : 'miss'}`}>{s.meets ? <CheckCircle2 size={12} /> : <AlertTriangle size={12} />}{s.meets ? 'met' : 'known miss'}</span>
            </div>
            <div className="mf-score-nums"><b>{s.value.toFixed(3)}</b><span>target {s.higher_is_better ? '≥' : '≤'} {s.target} · n={s.n}</span></div>
            <Meter value={s.value} target={s.target} higher={s.higher_is_better} />
          </li>
        ))}
        {rows.length === 0 && <li className="mf-empty">Nothing matches this filter.</li>}
      </ul>
    </Section>
  );
};

/* ------------------------------------------------------------------- paged */

const PagedSection: React.FC<{ data: ResultsPayload }> = ({ data }) => {
  const contexts = Array.from(new Set(data.phase_a.map((p) => p.context))).sort((a, b) => a - b);
  const [ctx, setCtx] = useState(String(contexts[contexts.length - 1] ?? 2048));
  const [bucket, setBucket] = useState('32');
  const rows = data.phase_a.filter((p) => p.context === Number(ctx));
  // Prefer the finer bucket where it was measured, else the 128 bucket.
  const byModel = new Map<string, typeof rows[number]>();
  rows.forEach((r) => {
    const cur = byModel.get(r.model);
    if (!cur || (bucket === '32' ? r.bucket === 32 : r.bucket === 128)) byModel.set(r.model, r);
  });
  const models = Array.from(byModel.keys());
  return (
    <Section id="paged" title="Paged decode vs stock" icon={<Gauge size={18} />}
      blurb="Throughput of the paged KV cache relative to stock Hugging Face decoding. Before = eager paged; after = the whole step as a CUDA graph."
      know={<KnowMore id="paged_vs_stock" data={data} />}>
      <div className="mf-toolbar">
        <Chips label="Context length" value={ctx} onChange={setCtx} options={contexts.map((c) => ({ value: String(c), label: `${c} tok` }))} />
        <Chips label="Gather bucket" value={bucket} onChange={setBucket} options={[{ value: '32', label: '32-token bucket' }, { value: '128', label: '128-token bucket' }]} />
        <KnowMore id="cuda_graph" data={data} label="What is a CUDA graph?" />
      </div>
      <GroupedBars
        groups={models}
        series={[
          { name: 'Before: eager paged', color: PALETTE[3], values: models.map((m) => byModel.get(m)!.eager_paged) },
          { name: 'After: graphed paged', color: PALETTE[1], values: models.map((m) => byModel.get(m)!.graphed_vs_stock) },
          { name: 'vs graphed contiguous', color: PALETTE[2], values: models.map((m) => byModel.get(m)!.graphed_vs_graphed) },
        ]}
        lines={[{ y: 0.95, label: 'target 0.95', color: 'var(--accent-red)' }, { y: 1, label: 'stock = 1.0', color: 'var(--text-muted)' }]}
      />
      <div className="mf-table-wrap">
        <table className="mf-table">
          <thead><tr><th>Model</th><th>Eager paged</th><th>Graphed vs stock</th><th>Graphed vs graphed</th><th>Output</th></tr></thead>
          <tbody>
            {models.map((m) => {
              const r = byModel.get(m)!;
              return (
                <tr key={m}>
                  <td>{m}</td>
                  <td>{bandText(r.eager_paged)}</td>
                  <td className={r.graphed_vs_stock && r.graphed_vs_stock.median >= 0.95 ? 'good' : 'warn'}><b>{bandText(r.graphed_vs_stock)}</b> <small>{bandRange(r.graphed_vs_stock)}</small></td>
                  <td>{bandText(r.graphed_vs_graphed)}</td>
                  <td>{r.output_identical ? <span className="mf-pill ok">identical</span> : <span className="mf-pill miss">differs</span>}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mf-note">Against an equally graphed contiguous baseline paged runs at {f(Math.min(...rows.map((r) => r.graphed_vs_graphed?.median ?? 9)))}–{f(Math.max(...rows.map((r) => r.graphed_vs_graphed?.median ?? 0)))}×: the remaining gap is the gather copy. <KnowMore id="paged_vs_graphed" data={data} label="Why two baselines?" /></p>
    </Section>
  );
};

/* -------------------------------------------------------------- speculative */

const ARMS: Array<[keyof import('../../services/results').SpecRow, string, string]> = [
  ['eager', 'Eager draft', PALETTE[4]], ['perseq', 'Per-sequence graph', PALETTE[3]],
  ['batched', 'Batched graph', PALETTE[1]], ['unrolled', 'Unrolled (1 replay)', PALETTE[2]],
];

const SpecSection: React.FC<{ data: ResultsPayload }> = ({ data }) => {
  const pairs = Array.from(new Set(data.speculative.map((s) => s.pair)));
  const [pair, setPair] = useState(pairs[0] ?? '');
  const rows = data.speculative.filter((s) => s.pair === pair).sort((a, b) => a.batch - b.batch);
  return (
    <Section id="spec" title="Speculative decoding" icon={<FlaskConical size={18} />}
      blurb="Speed with a small draft model, relative to plain decoding (1.0 = no gain). The eager draft collapses with batch size; the batched graph stays flat."
      know={<KnowMore id="spec_speedup" data={data} />}>
      <div className="mf-toolbar">
        <Chips label="Model pair" value={pair} onChange={setPair} options={pairs.map((p) => ({ value: p, label: p }))} />
        <KnowMore id="spec_arms" data={data} label="The four strategies" />
        <KnowMore id="spec_accept" data={data} label="Acceptance rate" />
      </div>
      <LineChart xLabel="batch size" xs={rows.map((r) => r.batch)} refLine={{ y: 1, label: 'plain decoding = 1.0' }}
        series={ARMS.map(([key, name, color]) => ({ name, color, values: rows.map((r) => (r[key] as Band | null | undefined) ?? null) })).filter((s) => s.values.some(Boolean))} />
      <div className="mf-table-wrap">
        <table className="mf-table">
          <thead><tr><th>Batch</th>{ARMS.map(([, n]) => <th key={n}>{n}</th>)}<th>Acceptance</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.batch}><td>{r.batch}</td>{ARMS.map(([key]) => <td key={key}>{bandText(r[key] as Band | null | undefined, 3)}</td>)}<td>{bandText(r.acceptance, 3)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  );
};

/* ------------------------------------------------------------------ chunked */

const heat = (v: number | undefined) => {
  if (v === undefined) return 'none';
  return v <= 2 ? 'ok' : v <= 3 ? 'mid' : 'bad';
};

const ChunkSection: React.FC<{ data: ResultsPayload }> = ({ data }) => {
  const backends = Array.from(new Set(data.chunked.map((c) => c.backend)));
  const [backend, setBackend] = useState(backends.includes('GPU') ? 'GPU' : backends[0] ?? 'CPU');
  const rows = data.chunked.filter((c) => c.backend === backend);
  const budgets = ['0', '64', '16', '4'];
  return (
    <Section id="chunk" title="Chunked prefill" icon={<Layers size={18} />}
      blurb="Worst delay a running request suffers when a 512-token prompt arrives, as a multiple of its normal step. Goal ≤ 2.0."
      know={<KnowMore id="chunk_stall" data={data} />}>
      <div className="mf-toolbar">
        <Chips label="Backend" value={backend} onChange={setBackend} options={backends.map((b) => ({ value: b, label: b }))} />
        <KnowMore id="chunk_budget" data={data} label="What is the budget?" />
      </div>
      <div className="mf-table-wrap">
        <table className="mf-table mf-heat">
          <thead><tr><th>Model</th>{budgets.map((b) => <th key={b}>{b === '0' ? 'Unchunked' : `Budget ${b}`}</th>)}</tr></thead>
          <tbody>
            {rows.map((r, ri) => (
              <tr key={r.model}>
                <td>{r.model}</td>
                {budgets.map((b, bi) => {
                  const band = r.budgets[b];
                  return (
                    <td key={b} className={`cell ${heat(band?.median)}`} style={{ animationDelay: `${(ri * 4 + bi) * 30}ms` }} title={band ? `${bandRange(band)}` : ''}>
                      {band ? band.median.toFixed(2) : '—'}<small>{band ? `n=${band.n}` : ''}</small>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mf-note">Green meets the 2.0 goal, amber is up to 3.0, red is worse. On GPU a step carrying a chunk runs two forward passes, which sets a floor near 2×.</p>
    </Section>
  );
};

/* ------------------------------------------------------------------- prefix */

const PrefixSection: React.FC<{ data: ResultsPayload }> = ({ data }) => {
  const rows = data.prefix.filter((p) => p.backend !== 'GPU int8');
  const gpu = rows.filter((p) => p.backend === 'GPU');
  const list = gpu.length ? gpu : rows;
  return (
    <Section id="prefix" title="Prefix sharing" icon={<Database size={18} />}
      blurb="Four requests share one prompt prefix. Lower ratios mean less memory and less repeated work; leaks must be zero."
      know={<KnowMore id="prefix_blocks" data={data} />}>
      <div className="mf-toolbar"><KnowMore id="prefix_prefill" data={data} label="Prefill work" /><KnowMore id="leaked" data={data} label="Leaked blocks" /></div>
      <div className="mf-cards">
        {list.map((p, i) => (
          <div key={p.backend + p.model} className="mf-mini glass-card" style={{ animationDelay: `${i * 50}ms` }}>
            <div className="mf-mini-title">{p.model} <small>{p.backend}</small></div>
            <div className="mf-mini-row"><span>Blocks used</span><b>{bandText(p.block_ratio, 3)}×</b></div>
            <Meter value={p.block_ratio?.median ?? 0} target={0.5} higher={false} max={1} />
            <div className="mf-mini-row"><span>Prefill work</span><b>{bandText(p.prefill_ratio, 2)}×</b></div>
            <Meter value={p.prefill_ratio?.median ?? 0} target={0.5} higher={false} max={1} />
            <div className="mf-mini-foot">
              <span className={`mf-pill ${p.leaked === 0 ? 'ok' : 'miss'}`}>{p.leaked} leaked</span>
              <span className={`mf-pill ${p.identical ? 'ok' : 'miss'}`}>{p.identical ? 'identical output' : 'output differs'}</span>
            </div>
          </div>
        ))}
      </div>
    </Section>
  );
};

/* -------------------------------------------------------------------- 14B */

const Q14Section: React.FC<{ data: ResultsPayload }> = ({ data }) => {
  const g = data.qwen14b.gpu_int8, c = data.qwen14b.cpu_fp32;
  const ctxs = Object.keys(g.context);
  return (
    <Section id="q14" title="Qwen3-14B" icon={<Cpu size={18} />}
      blurb="The largest model tested. fp16 needs about 29.5 GB, so the GPU runs use int8 weights; the CPU runs use fp32."
      know={<KnowMore id="q14" data={data} />}>
      <div className="mf-toolbar"><KnowMore id="int8" data={data} label="What is int8 here?" /></div>
      {ctxs.length > 0 && (
        <GroupedBars groups={ctxs.map((x) => `${x} tok`)}
          series={[
            { name: 'GPU int8 · paged / contiguous', color: PALETTE[1], values: ctxs.map((x) => g.context[x]) },
            { name: 'CPU fp32 · paged / contiguous', color: PALETTE[0], values: ctxs.map((x) => c.context[x] ?? null) },
          ]}
          lines={[{ y: 0.95, label: 'target 0.95', color: 'var(--accent-red)' }]} />
      )}
      <div className="mf-cards">
        <div className="mf-mini glass-card"><div className="mf-mini-title">GPU int8 · chunked prefill</div>
          <div className="mf-mini-row"><span>Unchunked stall</span><b>{bandText(g.chunked['0'])}×</b></div>
          <div className="mf-mini-row"><span>Budget 4 stall</span><b>{bandText(g.chunked['4'])}×</b></div>
          <div className="mf-mini-foot"><span className={`mf-pill ${g.output_identical ? 'ok' : 'miss'}`}>{g.output_identical ? 'identical output' : 'output differs'}</span></div></div>
        <div className="mf-mini glass-card"><div className="mf-mini-title">CPU fp32 · chunked prefill</div>
          <div className="mf-mini-row"><span>Unchunked stall</span><b>{bandText(c.chunked['0'])}×</b></div>
          <div className="mf-mini-row"><span>Budget 4 stall</span><b>{bandText(c.chunked['4'])}×</b></div>
          <div className="mf-mini-foot"><span className="mf-pill ok">1 run</span></div></div>
      </div>
    </Section>
  );
};

/* -------------------------------------------------------------- equivalence */

const EquivSection: React.FC<{ data: ResultsPayload }> = ({ data }) => (
  <Section id="equiv" title="fp16 output equivalence" icon={<ShieldCheck size={18} />}
    blurb="When half-precision output differs, is it a bug or a coin-toss between two equally scored tokens?"
    know={<KnowMore id="fp16_ties" data={data} />}>
    <Donut parts={[
      { name: 'byte-identical', value: data.equivalence.identical, color: PALETTE[1] },
      { name: 'explained exact ties', value: data.equivalence.explained_ties, color: PALETTE[3] },
      { name: 'unexplained (bugs)', value: data.equivalence.unexplained, color: PALETTE[4] },
    ]} />
    <p className="mf-note">At the first mismatch the two tokens had the same logit in a clean single-sequence pass: fp16 can’t separate them, so the winner depends on batch shape. No mismatch was unexplained.</p>
  </Section>
);

/* ------------------------------------------------------------------- stress */

const StressSection: React.FC<{ data: ResultsPayload }> = ({ data }) => (
  <Section id="stress" title="Soak tests" icon={<ShieldCheck size={18} />}
    blurb="110+ requests per model, sequential then 8 at a time: no leaked blocks and flat memory."
    know={<KnowMore id="stress_rss" data={data} />}>
    <div className="mf-table-wrap">
      <table className="mf-table">
        <thead><tr><th>Run</th><th>Requests</th><th>Leaked blocks</th><th>Memory growth</th><th>p95 first token</th><th>Result</th></tr></thead>
        <tbody>
          {data.stress.map((s) => (
            <tr key={s.label}>
              <td>{s.model}</td><td>{s.succeeded}/{s.requests}</td><td>{s.leaked_blocks}</td>
              <td><div className="mf-inline-meter"><Meter value={Math.max(0, s.rss_growth_mb)} target={50} higher={false} max={60} /><span>{s.rss_growth_mb.toFixed(1)} MB</span></div></td>
              <td>{s.p95_ttft_ms == null ? '—' : `${Math.round(s.p95_ttft_ms)} ms`}</td>
              <td><span className={`mf-pill ${s.passed ? 'ok' : 'miss'}`}>{s.passed ? 'pass' : 'fail'}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </Section>
);

/* ---------------------------------------------------------------- inventory */

const InventorySection: React.FC<{ data: ResultsPayload }> = ({ data }) => {
  const [q, setQ] = useState('');
  const [group, setGroup] = useState('all');
  const [limit, setLimit] = useState(15);
  const groups = Array.from(new Set(data.inventory.map((r) => r.group))).sort();
  const rows = data.inventory.filter((r) => (group === 'all' || r.group === group) && (!q || (r.label + r.headline).toLowerCase().includes(q.toLowerCase())));
  useEffect(() => setLimit(15), [q, group]);
  return (
    <Section id="inventory" title="Run inventory" icon={<Database size={18} />}
      blurb="Every archived run, so any figure above can be traced to the files behind it."
      know={<KnowMore id="inventory" data={data} />}>
      <div className="mf-toolbar">
        <label className="mf-search"><Search size={14} /><input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search runs (model, group, result)…" aria-label="Search runs" /></label>
        <select value={group} onChange={(e) => setGroup(e.target.value)} aria-label="Filter by group" className="mf-select">
          <option value="all">All groups</option>
          {groups.map((g) => <option key={g} value={g}>{g}</option>)}
        </select>
        <span className="mf-count">{rows.length} runs</span>
      </div>
      <div className="mf-table-wrap">
        <table className="mf-table">
          <thead><tr><th>Run</th><th>Group</th><th>Device</th><th>Round</th><th>Headline</th></tr></thead>
          <tbody>
            {rows.slice(0, limit).map((r) => (
              <tr key={r.label}><td className="mono" title={r.files.join(', ')}>{r.label}</td><td>{r.group}</td><td>{r.device}</td><td>{r.round ?? '—'}</td><td>{r.headline || '—'}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      {limit < rows.length && <button type="button" className="mf-btn mf-more" onClick={() => setLimit((l) => l + 25)}>Show more ({rows.length - limit} left)</button>}
    </Section>
  );
};
