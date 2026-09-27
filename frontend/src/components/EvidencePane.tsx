import { useEffect, useRef } from 'react';

import type { EvidenceSpan, SourceDocument } from '../services/types';

interface Props {
  document: SourceDocument;
  /** The span to highlight, or null when nothing is selected. */
  span: EvidenceSpan | null;
  /** True when a chip is selected but this document has no span for it. */
  missingHere: boolean;
}

/**
 * Renders a document with one highlighted span, scrolled into view.
 *
 * The offsets come from the backend and index into exactly this text, so the
 * highlight is a slice rather than a search. That is what keeps the highlight
 * honest: the UI cannot decide a skill is evidenced somewhere the backend did not.
 */
export default function EvidencePane({ document: source, span, missingHere }: Props) {
  const markRef = useRef<HTMLElement>(null);

  // The adapter already rejects unusable offsets, but this component is also
  // rendered directly, so it validates rather than trusting its input. Bad
  // offsets would otherwise render an invisible, empty <mark>.
  const usable =
    span !== null &&
    span.start !== null &&
    span.end !== null &&
    span.start >= 0 &&
    span.end > span.start &&
    span.end <= source.text.length;

  useEffect(() => {
    if (!usable) return;
    const mark = markRef.current;
    // Guarded because scrollIntoView is absent in some environments, including
    // jsdom. Failing to scroll must never take down the results view.
    if (mark && typeof mark.scrollIntoView === 'function') {
      mark.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
  }, [usable, span?.start, span?.end]);

  const body = (() => {
    if (!usable || span === null || span.start === null || span.end === null) {
      return source.text;
    }
    return (
      <>
        {source.text.slice(0, span.start)}
        <mark
          ref={markRef}
          className="rounded bg-accent/25 px-0.5 font-semibold text-ink underline decoration-accent decoration-2"
        >
          {source.text.slice(span.start, span.end)}
        </mark>
        {source.text.slice(span.end)}
      </>
    );
  })();

  return (
    <section className="card flex min-h-0 flex-col" aria-labelledby={`pane-${source.kind}`}>
      <div className="flex items-baseline justify-between gap-2">
        <h3 id={`pane-${source.kind}`} className="text-sm font-semibold">
          {source.heading}
        </h3>
        {missingHere && (
          <span className="text-xs text-ink-muted">Not mentioned here</span>
        )}
      </div>
      <div
        className="mt-3 max-h-80 overflow-auto rounded border border-line bg-surface p-3"
        tabIndex={0}
        role="region"
        aria-label={`${source.heading} text`}
      >
        {source.text.trim() ? (
          <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed">
            {body}
          </pre>
        ) : (
          <p className="text-xs text-ink-muted">
            No text was extracted from this document.
          </p>
        )}
      </div>
    </section>
  );
}
