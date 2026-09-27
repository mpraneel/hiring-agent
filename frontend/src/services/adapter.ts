/**
 * The only code in the app that touches raw API responses.
 *
 * Every response is validated with Zod and mapped into the UI model. A malformed
 * response therefore becomes a typed error state rather than a crash or a blank
 * screen, and components never need to defend against a missing field.
 */
import { z } from 'zod';

import {
  examplesResponseSchema,
  matchResponseSchema,
  errorResponseSchema,
  type spanSchema,
} from './schemas';
import type {
  AppError,
  EvidenceSpan,
  ExampleView,
  MatchView,
  Priority,
  SkillChip,
} from './types';

export class AdapterError extends Error {
  readonly appError: AppError;

  constructor(appError: AppError) {
    super(appError.message);
    this.name = 'AdapterError';
    this.appError = appError;
  }
}

type RawSpan = z.infer<typeof spanSchema>;

const SCORE_BANDS: ReadonlyArray<readonly [number, string]> = [
  [0.85, 'Very strong match'],
  [0.7, 'Strong match'],
  [0.5, 'Partial match'],
  [0.3, 'Weak match'],
  [0, 'Poor match'],
];

export function scoreLabel(score: number): string {
  for (const [threshold, label] of SCORE_BANDS) {
    if (score >= threshold) return label;
  }
  return 'Poor match';
}

/**
 * Keep a span only when its offsets are usable against the given text.
 *
 * A skill can be extracted without a locatable span, and offsets must be checked
 * against the actual text length: a highlight that runs past the end of the
 * document would silently render nothing.
 */
function toSpan(raw: RawSpan | undefined, textLength: number): EvidenceSpan | null {
  if (!raw) return null;
  const { start, end } = raw;
  const usable =
    typeof start === 'number' &&
    typeof end === 'number' &&
    start >= 0 &&
    end > start &&
    end <= textLength;
  return {
    skill: raw.skill,
    evidence: raw.evidence ?? '',
    start: usable ? start : null,
    end: usable ? end : null,
  };
}

function indexSpans(spans: RawSpan[], textLength: number): Map<string, EvidenceSpan> {
  const bySkill = new Map<string, EvidenceSpan>();
  for (const raw of spans) {
    const span = toSpan(raw, textLength);
    if (!span) continue;
    const existing = bySkill.get(span.skill);
    // Prefer a span that actually resolved over one that did not.
    if (!existing || (existing.start === null && span.start !== null)) {
      bySkill.set(span.skill, span);
    }
  }
  return bySkill;
}

function buildChips(
  skills: string[],
  priority: Priority,
  matched: boolean,
  jdSpans: Map<string, EvidenceSpan>,
  resumeSpans: Map<string, EvidenceSpan>,
): SkillChip[] {
  return skills.map((skill) => ({
    skill,
    priority,
    matched,
    jdSpan: jdSpans.get(skill) ?? null,
    resumeSpan: resumeSpans.get(skill) ?? null,
  }));
}

/** Validate and map a /api/v1/match response. */
export function adaptMatchResponse(payload: unknown): MatchView {
  const parsed = matchResponseSchema.safeParse(payload);
  if (!parsed.success) {
    throw new AdapterError({
      kind: 'malformed',
      message:
        'The server replied in a format this page does not understand. It may be running a different version.',
    });
  }

  const data = parsed.data;
  const jdText = data.documents.jd.source_text;
  const resumeText = data.documents.resume.source_text;

  const jdSpans = indexSpans(data.documents.jd.requirements, jdText.length);
  const resumeSpans = indexSpans(data.documents.resume.skills, resumeText.length);

  const score = Number.isFinite(data.match_score)
    ? Math.min(1, Math.max(0, data.match_score))
    : 0;

  return {
    score,
    scorePercent: Math.round(score * 100),
    scoreLabel: scoreLabel(score),
    breakdown: {
      mustMatched: data.breakdown.must_matched.length,
      mustTotal: data.breakdown.must_total,
      niceMatched: data.breakdown.nice_matched.length,
      niceTotal: data.breakdown.nice_total,
    },
    matchedMust: buildChips(data.breakdown.must_matched, 'must', true, jdSpans, resumeSpans),
    missingMust: buildChips(data.breakdown.must_missing, 'must', false, jdSpans, resumeSpans),
    matchedNice: buildChips(data.breakdown.nice_matched, 'nice', true, jdSpans, resumeSpans),
    missingNice: buildChips(data.breakdown.nice_missing, 'nice', false, jdSpans, resumeSpans),
    rationale: data.llm_rationale ?? null,
    suggestions: data.suggestions.slice(0, 3),
    extractionPath: data.extraction_path,
    llmStatus: data.llm_status,
    requestId: data.request_id,
    jobTitle: data.documents.jd.title ?? null,
    jd: { kind: 'jd', heading: 'Job description', text: jdText },
    resume: { kind: 'resume', heading: 'Resume', text: resumeText },
  };
}

/** Validate and map a /api/v1/examples response. */
export function adaptExamplesResponse(payload: unknown): ExampleView[] {
  const parsed = examplesResponseSchema.safeParse(payload);
  if (!parsed.success) {
    throw new AdapterError({
      kind: 'malformed',
      message: 'The example data could not be read, so the examples are unavailable.',
    });
  }
  return parsed.data.examples.map((example) => ({
    id: example.id,
    label: example.label,
    description: example.description,
    jobDescription: example.job_description,
    resumeText: example.resume_text,
  }));
}

/** Turn a non-2xx response body into a message a person can act on. */
export function adaptErrorResponse(status: number, payload: unknown): AppError {
  const parsed = errorResponseSchema.safeParse(payload);
  const detail = parsed.success ? parsed.data.detail : undefined;

  const fromDetail =
    typeof detail === 'string' ? detail : (detail?.error ?? detail?.message ?? undefined);
  const requestId =
    (typeof detail === 'object' ? detail?.request_id : undefined) ??
    (parsed.success ? parsed.data.request_id : undefined);

  if (status >= 400 && status < 500) {
    return {
      kind: 'validation',
      message: fromDetail ?? 'The request was rejected. Check the resume and job description.',
      ...(requestId ? { requestId } : {}),
    };
  }

  return {
    kind: 'server',
    message: 'Something went wrong on the server. The score could not be computed.',
    ...(requestId ? { requestId } : {}),
  };
}
