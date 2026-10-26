import React from 'react';
import { Cpu, HardDrive, Layers, Zap } from 'lucide-react';
import { LearnMore } from './LearnMore';
import { DEVICE_PANEL } from './explanations';

/**
 * What hardware the engine is actually on, and what it is holding there.
 *
 * Worth its own panel because "is this running on the GPU?" is the first
 * question anyone asks of an inference engine, and until now the dashboard
 * could not answer it -- the telemetry payload carried no device field at all.
 *
 * The memory figures are `null` on CPU, not zero, and this renders that as a
 * dash with an explicit note rather than "0 MB". torch has no CPU allocator to
 * query, so zero here would mean "measured, and it was nothing" when the truth
 * is "nothing measured it". That distinction is the whole reason Phase E's
 * memory telemetry was blocked on a CUDA build.
 */

export interface DevicePanelProps {
  device: string;
  deviceName: string | null;
  dtype: string;
  cudaAvailable: boolean;
  poolMb: number;
  numBlocks: number;
  capacityTokens: number;
  allocatedMb: number | null;
  peakMemoryMb: number | null;
  vramTotalMb: number | null;
  vramFreeMb: number | null;
}

const Stat: React.FC<{
  icon: React.ReactNode;
  label: string;
  value: string;
  note?: string;
}> = ({ icon, label, value, note }) => (
  <div style={{ flex: '1 1 150px', minWidth: 0 }}>
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '5px',
        color: 'var(--text-muted)',
        fontSize: '0.7rem',
        textTransform: 'uppercase',
        letterSpacing: '0.04em',
        marginBottom: '3px',
      }}
    >
      {icon}
      {label}
    </div>
    <div
      style={{
        fontFamily: 'var(--font-mono)',
        fontWeight: 700,
        fontSize: '0.95rem',
        color: 'var(--text-primary)',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
      }}
      title={value}
    >
      {value}
    </div>
    {note && (
      <div style={{ color: 'var(--text-muted)', fontSize: '0.68rem', marginTop: '1px' }}>
        {note}
      </div>
    )}
  </div>
);

const mb = (value: number | null): string =>
  value === null ? '--' : `${value.toFixed(0)} MB`;

export const DevicePanel: React.FC<DevicePanelProps> = ({
  device,
  deviceName,
  dtype,
  cudaAvailable,
  poolMb,
  numBlocks,
  capacityTokens,
  allocatedMb,
  peakMemoryMb,
  vramTotalMb,
  vramFreeMb,
}) => {
  const onGpu = device.startsWith('cuda');
  // fp16 halves the KV cache against fp32, so the dtype is a memory fact and
  // not just a precision one. Worth surfacing next to the pool size.
  const shortDtype = dtype.replace('torch.', '');

  return (
    <div
      style={{
        padding: '14px 16px',
        borderRadius: '10px',
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border-color)',
        marginBottom: '16px',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          marginBottom: '12px',
        }}
      >
        <Zap size={14} color={onGpu ? 'var(--accent-emerald)' : 'var(--text-muted)'} />
        <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>Execution device</span>
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.72rem',
            padding: '1px 7px',
            borderRadius: '10px',
            background: onGpu ? 'var(--accent-emerald)' : 'var(--bg-primary)',
            color: onGpu ? 'var(--bg-primary)' : 'var(--text-muted)',
            border: onGpu ? 'none' : '1px solid var(--border-color)',
          }}
        >
          {onGpu ? 'GPU' : 'CPU'}
        </span>
        {!onGpu && cudaAvailable && (
          // A real state, not a hypothetical: CUDA can be available while the
          // model was still loaded onto the CPU.
          <span style={{ color: 'var(--accent-amber)', fontSize: '0.72rem' }}>
            CUDA is available but the model is on CPU
          </span>
        )}
      </div>

      <div style={{ display: 'flex', gap: '14px', flexWrap: 'wrap' }}>
        <Stat
          icon={<Cpu size={11} />}
          label="device"
          value={deviceName ?? device}
          note={`${device} · ${shortDtype}`}
        />
        <Stat
          icon={<Layers size={11} />}
          label="KV pool"
          value={`${poolMb.toFixed(0)} MB`}
          note={`${numBlocks} blocks · ${capacityTokens.toLocaleString()} tokens`}
        />
        <Stat
          icon={<HardDrive size={11} />}
          label="torch allocated"
          value={mb(allocatedMb)}
          note={onGpu ? 'weights + pool + activations' : 'no CPU allocator to query'}
        />
        <Stat
          icon={<HardDrive size={11} />}
          label="peak allocated"
          value={mb(peakMemoryMb)}
          note={onGpu ? 'high-water mark since start' : 'CPU: not measurable'}
        />
        {onGpu && vramTotalMb !== null && (
          <Stat
            icon={<HardDrive size={11} />}
            label="VRAM free"
            value={mb(vramFreeMb)}
            note={`of ${vramTotalMb.toFixed(0)} MB total`}
          />
        )}
      </div>

      <LearnMore {...DEVICE_PANEL} />
    </div>
  );
};
