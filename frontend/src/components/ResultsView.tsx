import { useState } from 'react';

import EvidencePane from './EvidencePane';
import type { ExtractionPath, MatchView, SkillChip } from '../services/types';

const PATH_EXPLANATION: Record<ExtractionPath, string> = {
  llm: 'Skills were extracted by the language model, and every one was checked against a quote from the source document.',
  hybrid:
    'Skills came from both the language model and the regex and ontology parser. Where the two disagreed on whether something is required, the model won.',
  deterministic:
    'Skills were extracted by the regex and ontology parser alone, with no language model involved.',
  fallback:
    'The language model was tried and did not return usable output, so the regex and ontology parser produced this result instead.',
};

const PATH_LABEL: Record<ExtractionPath, string> = {
  llm: 'LLM extraction',
  hybrid: 'Hybrid extraction',
  deterministic: 'Deterministic extraction',
  fallback: 'Deterministic fallback',
};

interface GroupProps {
  heading: string;
  icon: string;
  hint: string;
  chips: SkillChip[];
  selected: string | null;
  onSelect: (skill: string) => void;
  tone: 'matched' | 'missing';
}

/**
 * One chip group.
 *
 * Colour is never the only signal: each group has its own heading and icon, and
 * every chip repeats its state in text for screen readers.
 */
function ChipGroup({
  heading,
  icon,
  hint,
  chips,
  selected,
  onSelect,
  tone,
}: GroupProps) {
  return (
    <div>
      <h3 className="flex items-center gap-2 text-sm font-semibold">
        <span aria-hidden="true">{icon}</span>
        {heading}
        <span className="font-normal text-ink-muted">({chips.length})</span>
      </h3>
      {chips.length === 0 ? (
        <p className="mt-2 text-sm text-ink-muted">{hint}</p>
      ) : (
        <ul className="mt-2 flex flex-wrap gap-2">
          {chips.map((chip) => {
            const isSelected = selected === chip.skill;
            const hasEvidence = chip.jdSpan?.start != null || chip.resumeSpan?.start != null;
            return (
              <li key={`${heading}-${chip.skill}`}>
                <button
                  type="button"
                  onClick={() => onSelect(chip.skill)}
                  onMouseEnter={() => onSelect(chip.skill)}
                  aria-pressed={isSelected}
                  className={`chip ${
                    tone === 'matched'
                      ? 'border-emerald-600/40 bg-emerald-600/10'
                      : 'border-amber-600/50 bg-amber-600/10'
                  } ${isSelected ? 'ring-2 ring-accent' : ''}`}
                >
                  <span aria-hidden="true">{tone === 'matched' ? '✓' : '○'}</span>
                  <span>{chip.skill}</span>
                  <span className="sr-only">
                    {tone === 'matched' ? 'present in resume' : 'not found in resume'}.
                    {hasEvidence ? ' Select to show the evidence.' : ' No evidence span available.'}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

interface Props {
  result: MatchView;
  onStartOver: () => void;
  onEditInputs: () => void;
}

export default function ResultsView({ result, onStartOver, onEditInputs }: Props) {
  const [selected, setSelected] = useState<string | null>(null);

  const allChips = [
    ...result.matchedMust,
    ...result.missingMust,
    ...result.matchedNice,
    ...result.missingNice,
  ];
  const active = allChips.find((chip) => chip.skill === selected) ?? null;

  return (
    <div className="space-y-6">
      <section className="card" aria-labelledby="score-heading">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h2 id="score-heading" className="text-sm font-medium text-ink-muted">
              Match score
              {result.jobTitle ? ` for ${result.jobTitle}` : ''}
            </h2>
            <p className="mt-1 flex items-baseline gap-3">
              <span className="text-5xl font-bold tabular-nums">
                {result.scorePercent}%
              </span>
              <span className="text-base text-ink-muted">{result.scoreLabel}</span>
            </p>
          </div>
          <dl className="flex gap-6 text-sm">
            <div>
              <dt className="text-ink-muted">Must-have</dt>
              <dd className="text-lg font-semibold tabular-nums">
                {result.breakdown.mustMatched} of {result.breakdown.mustTotal}
              </dd>
            </div>
            <div>
              <dt className="text-ink-muted">Nice-to-have</dt>
              <dd className="text-lg font-semibold tabular-nums">
                {result.breakdown.niceMatched} of {result.breakdown.niceTotal}
              </dd>
            </div>
          </dl>
        </div>

        <p className="mt-4 flex flex-wrap items-center gap-2 text-xs">
          <span
            className="inline-flex items-center gap-1 rounded-full border border-line px-2 py-0.5"
            title={PATH_EXPLANATION[result.extractionPath]}
          >
            <span aria-hidden="true">🔎</span>
            {PATH_LABEL[result.extractionPath]}
          </span>
          <span className="text-ink-muted">
            {PATH_EXPLANATION[result.extractionPath]}
          </span>
        </p>
      </section>

      <section className="card space-y-5" aria-labelledby="skills-heading">
        <div>
          <h2 id="skills-heading" className="text-base font-semibold">
            Skills
          </h2>
          <p className="mt-1 text-sm text-ink-muted">
            Select a skill to highlight the text it came from. Must-have skills count double.
          </p>
        </div>
        <ChipGroup
          heading="Matched must-have"
          icon="✓"
          hint="None of the required skills were found in the resume."
          chips={result.matchedMust}
          selected={selected}
          onSelect={setSelected}
          tone="matched"
        />
        <ChipGroup
          heading="Missing must-have"
          icon="⚠"
          hint="Every required skill was found. Nothing critical is missing."
          chips={result.missingMust}
          selected={selected}
          onSelect={setSelected}
          tone="missing"
        />
        <ChipGroup
          heading="Matched nice-to-have"
          icon="✓"
          hint="None of the preferred skills were found in the resume."
          chips={result.matchedNice}
          selected={selected}
          onSelect={setSelected}
          tone="matched"
        />
        <ChipGroup
          heading="Missing nice-to-have"
          icon="○"
          hint="Every preferred skill was found as well."
          chips={result.missingNice}
          selected={selected}
          onSelect={setSelected}
          tone="missing"
        />
      </section>

      <section aria-labelledby="evidence-heading">
        <h2 id="evidence-heading" className="text-base font-semibold">
          Evidence
        </h2>
        <p className="mt-1 text-sm text-ink-muted" aria-live="polite">
          {active
            ? `Showing where ${active.skill} appears.`
            : 'Select a skill above to see the exact text it was extracted from.'}
        </p>
        {/* Stacks on narrow screens, side by side from md up. */}
        <div className="mt-3 grid gap-4 md:grid-cols-2">
          <EvidencePane
            document={result.jd}
            span={active?.jdSpan ?? null}
            missingHere={active !== null && active.jdSpan?.start == null}
          />
          <EvidencePane
            document={result.resume}
            span={active?.resumeSpan ?? null}
            missingHere={active !== null && active.resumeSpan?.start == null}
          />
        </div>
      </section>

      <section className="card" aria-labelledby="explanation-heading">
        <h2 id="explanation-heading" className="text-base font-semibold">
          Explanation
        </h2>
        {result.llmStatus === 'ok' && result.rationale ? (
          <>
            <p className="mt-2 text-sm leading-relaxed">{result.rationale}</p>
            {result.suggestions.length > 0 && (
              <>
                <h3 className="mt-4 text-sm font-semibold">Suggestions</h3>
                <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">
                  {result.suggestions.map((suggestion) => (
                    <li key={suggestion}>{suggestion}</li>
                  ))}
                </ul>
              </>
            )}
          </>
        ) : (
          <p className="mt-2 text-sm text-ink-muted">
            {result.llmStatus === 'disabled'
              ? 'No language model is configured, so there is no written explanation. The score above is computed the same way either way.'
              : 'The written explanation is unavailable right now. The score above still stands: it is computed from the skill lists, not by the language model.'}
          </p>
        )}
      </section>

      <div className="flex flex-wrap gap-3">
        <button type="button" className="btn-secondary" onClick={onEditInputs}>
          Edit inputs
        </button>
        <button type="button" className="btn-secondary" onClick={onStartOver}>
          Start over
        </button>
        <p className="w-full text-xs text-ink-muted">Request {result.requestId}</p>
      </div>
    </div>
  );
}
