import { useEffect, useState } from 'react';

/**
 * Staged progress plus a skeleton of the results layout.
 *
 * LLM extraction can take several seconds. A bare spinner gives no sense of
 * whether anything is happening, so the stages advance on a timer and the
 * skeleton shows the shape of what is coming.
 */
const STAGES = [
  'Reading resume',
  'Extracting requirements',
  'Scoring',
  'Writing explanation',
] as const;

const STAGE_MS = 1400;

interface Props {
  onCancel: () => void;
}

export default function LoadingView({ onCancel }: Props) {
  const [stage, setStage] = useState(0);

  useEffect(() => {
    // Hold on the last stage rather than looping, which would imply restarting.
    if (stage >= STAGES.length - 1) return;
    const timer = setTimeout(() => setStage((current) => current + 1), STAGE_MS);
    return () => clearTimeout(timer);
  }, [stage]);

  return (
    <div className="space-y-6">
      <section className="card" aria-labelledby="progress-heading">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 id="progress-heading" className="text-base font-semibold">
            Analysing
          </h2>
          <button type="button" className="btn-secondary" onClick={onCancel}>
            Cancel
          </button>
        </div>

        <ol className="mt-4 space-y-2" aria-live="polite">
          {STAGES.map((label, index) => {
            const done = index < stage;
            const active = index === stage;
            return (
              <li key={label} className="flex items-center gap-3 text-sm">
                <span
                  aria-hidden="true"
                  className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full border text-xs ${
                    done
                      ? 'border-accent bg-accent text-white'
                      : active
                        ? 'border-accent text-accent'
                        : 'border-line text-ink-muted'
                  }`}
                >
                  {done ? '✓' : index + 1}
                </span>
                <span className={active || done ? 'text-ink' : 'text-ink-muted'}>
                  {label}
                  {active && '…'}
                </span>
              </li>
            );
          })}
        </ol>
      </section>

      {/* Skeleton of the results layout, so the page does not jump on arrival. */}
      <div aria-hidden="true" className="space-y-6">
        <div className="card">
          <div className="h-12 w-28 animate-pulse rounded bg-line" />
          <div className="mt-3 h-4 w-48 animate-pulse rounded bg-line" />
        </div>
        <div className="card space-y-3">
          {[0, 1, 2, 3].map((row) => (
            <div key={row} className="space-y-2">
              <div className="h-3 w-32 animate-pulse rounded bg-line" />
              <div className="flex gap-2">
                <div className="h-7 w-20 animate-pulse rounded-full bg-line" />
                <div className="h-7 w-24 animate-pulse rounded-full bg-line" />
                <div className="h-7 w-16 animate-pulse rounded-full bg-line" />
              </div>
            </div>
          ))}
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="card h-48 animate-pulse" />
          <div className="card h-48 animate-pulse" />
        </div>
      </div>
    </div>
  );
}
