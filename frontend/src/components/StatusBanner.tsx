import React, { useCallback, useEffect, useState } from 'react';
import { Activity, AlertTriangle, Loader2, Moon, Sun, Wifi, WifiOff } from 'lucide-react';

/**
 * Connection, engine-load and capacity state, plus the theme toggle.
 *
 * Three states the dashboard previously handled only loosely, each of which
 * looks like a broken app if it is not shown:
 *
 *  - **The WebSocket dropped.** `TelemetryClient` already retries every 1.5 s,
 *    but silently, so a reconnecting dashboard was indistinguishable from a
 *    frozen one.
 *  - **The engine is still loading.** A cold server spends 10-30 s loading model
 *    weights, during which /v1/health reports engine_ready: false and every
 *    telemetry push carries `telemetry: null`. Without a banner this reads as a
 *    dead backend.
 *  - **The engine is at capacity.** `max_concurrency` admissions are in flight
 *    and further requests get a 429, which is correct backpressure rather than
 *    a failure -- but only if the UI says so.
 */

export type ConnectionState = 'connecting' | 'live' | 'reconnecting' | 'disconnected';

export interface StatusBannerProps {
  connection: ConnectionState;
  /** From /v1/health. Null while the first probe is in flight. */
  engineReady: boolean | null;
  loadError: string | null;
  /** From telemetry. Omitted before the first payload arrives. */
  inFlight?: number;
  maxConcurrency?: number;
  /** Seconds since the last telemetry payload, for a staleness hint. */
  secondsSinceUpdate?: number;
}

type Tone = 'good' | 'warn' | 'bad' | 'idle';

const TONE_COLOR: Record<Tone, string> = {
  good: 'var(--accent-emerald)',
  warn: 'var(--accent-amber)',
  bad: 'var(--accent-red)',
  idle: 'var(--text-muted)',
};

const THEME_KEY = 'memforge-theme';

function readStoredTheme(): 'dark' | 'light' {
  // localStorage throws in private-browsing modes and when site data is
  // blocked, so a failure here must fall back rather than break the render.
  try {
    return window.localStorage.getItem(THEME_KEY) === 'light' ? 'light' : 'dark';
  } catch {
    return 'dark';
  }
}

function applyTheme(theme: 'dark' | 'light'): void {
  // Dark is the default declared on :root, so it needs no attribute.
  if (theme === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
  } else {
    document.documentElement.removeAttribute('data-theme');
  }
  try {
    window.localStorage.setItem(THEME_KEY, theme);
  } catch {
    /* the toggle still works for this session */
  }
}

const Pill: React.FC<{
  tone: Tone;
  icon: React.ReactNode;
  label: string;
  value: string;
  title?: string;
}> = ({ tone, icon, label, value, title }) => (
  <div
    title={title}
    style={{
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      padding: '6px 12px',
      borderRadius: '999px',
      border: `1px solid ${TONE_COLOR[tone]}`,
      background: 'var(--bg-card)',
      fontSize: '0.78rem',
      whiteSpace: 'nowrap',
    }}
  >
    <span style={{ display: 'flex', alignItems: 'center', color: TONE_COLOR[tone] }}>
      {icon}
    </span>
    <span style={{ color: 'var(--text-muted)' }}>{label}</span>
    <span style={{ color: 'var(--text-primary)', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
      {value}
    </span>
  </div>
);

export const StatusBanner: React.FC<StatusBannerProps> = ({
  connection,
  engineReady,
  loadError,
  inFlight,
  maxConcurrency,
  secondsSinceUpdate,
}) => {
  const [theme, setTheme] = useState<'dark' | 'light'>(readStoredTheme);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((current) => (current === 'dark' ? 'light' : 'dark'));
  }, []);

  const connectionPill = (() => {
    switch (connection) {
      case 'live':
        return { tone: 'good' as Tone, icon: <Wifi size={14} />, value: 'live' };
      case 'connecting':
        return {
          tone: 'idle' as Tone,
          icon: <Loader2 size={14} className="spin" />,
          value: 'connecting',
        };
      case 'reconnecting':
        return {
          tone: 'warn' as Tone,
          icon: <Loader2 size={14} className="spin" />,
          value: 'reconnecting',
        };
      default:
        return { tone: 'bad' as Tone, icon: <WifiOff size={14} />, value: 'offline' };
    }
  })();

  // engineReady === null means the first /v1/health probe has not answered yet,
  // which is different from "not ready" and should not look like a problem.
  const enginePill = (() => {
    if (loadError) {
      return { tone: 'bad' as Tone, icon: <AlertTriangle size={14} />, value: 'load failed' };
    }
    if (engineReady === null) {
      return { tone: 'idle' as Tone, icon: <Loader2 size={14} className="spin" />, value: 'checking' };
    }
    if (!engineReady) {
      return {
        tone: 'warn' as Tone,
        icon: <Loader2 size={14} className="spin" />,
        value: 'loading model',
      };
    }
    return { tone: 'good' as Tone, icon: <Activity size={14} />, value: 'ready' };
  })();

  const hasCapacity = typeof inFlight === 'number' && typeof maxConcurrency === 'number';
  const capacityTone: Tone = (() => {
    if (!hasCapacity || maxConcurrency === 0) return 'idle';
    const ratio = inFlight! / maxConcurrency!;
    if (ratio >= 1) return 'bad';
    if (ratio >= 0.75) return 'warn';
    return 'good';
  })();

  const stale = typeof secondsSinceUpdate === 'number' && secondsSinceUpdate > 3;

  return (
    <div
      className="glass-card"
      style={{
        padding: '12px 16px',
        marginBottom: '20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '12px',
        flexWrap: 'wrap',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
        <Pill
          tone={connectionPill.tone}
          icon={connectionPill.icon}
          label="telemetry"
          value={connectionPill.value}
          title={
            connection === 'reconnecting'
              ? 'The telemetry socket dropped and is retrying every 1.5 s'
              : undefined
          }
        />
        <Pill
          tone={enginePill.tone}
          icon={enginePill.icon}
          label="engine"
          value={enginePill.value}
          title={loadError ?? undefined}
        />
        {hasCapacity && (
          <Pill
            tone={capacityTone}
            icon={<Activity size={14} />}
            label="in flight"
            value={`${inFlight} / ${maxConcurrency}`}
            title={
              capacityTone === 'bad'
                ? 'At capacity: further requests receive HTTP 429 with Retry-After. This is backpressure working, not a failure.'
                : 'Requests admitted to the engine against the configured max_concurrency'
            }
          />
        )}
        {stale && (
          <Pill
            tone="warn"
            icon={<AlertTriangle size={14} />}
            label="last update"
            value={`${secondsSinceUpdate!.toFixed(0)}s ago`}
            title="Telemetry should arrive every 0.4 s. A longer gap means the feed has stalled."
          />
        )}
      </div>

      <button
        onClick={toggleTheme}
        aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
        title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '7px 14px',
          borderRadius: '999px',
          border: '1px solid var(--border-color)',
          background: 'var(--bg-card)',
          color: 'var(--text-secondary)',
          cursor: 'pointer',
          fontSize: '0.78rem',
          fontFamily: 'var(--font-sans)',
        }}
      >
        {theme === 'dark' ? <Moon size={14} /> : <Sun size={14} />}
        {theme === 'dark' ? 'Dark' : 'Light'}
      </button>

      {loadError && (
        <div
          style={{
            flexBasis: '100%',
            color: 'var(--accent-red)',
            fontSize: '0.8rem',
            fontFamily: 'var(--font-mono)',
          }}
        >
          {loadError}
        </div>
      )}
    </div>
  );
};
