/**
 * Adapter tests.
 *
 * The adapter is the only place raw API shapes are trusted, so these cover the
 * malformed cases explicitly: that is the difference between a typed error state
 * and a blank screen.
 */
import { describe, expect, it } from 'vitest';

import {
  AdapterError,
  adaptErrorResponse,
  adaptExamplesResponse,
  adaptMatchResponse,
  scoreLabel,
} from './adapter';

const JD_TEXT = 'Requirements\n- Strong Python\n- Docker';
const RESUME_TEXT = 'Skills\nPython, Redis';

/** Offsets are derived, not hand counted, so the fixture cannot drift. */
function offsetsOf(text: string, word: string): { start: number; end: number } {
  const start = text.indexOf(word);
  if (start < 0) throw new Error(`fixture bug: ${word} not found in text`);
  return { start, end: start + word.length };
}

const JD_PYTHON = offsetsOf(JD_TEXT, 'Python');
const JD_DOCKER = offsetsOf(JD_TEXT, 'Docker');
const RESUME_PYTHON = offsetsOf(RESUME_TEXT, 'Python');

interface RawSpanFixture {
  skill: string;
  priority: 'must' | 'nice' | null;
  evidence: string;
  start: number | null;
  end: number | null;
}

interface MatchFixture {
  match_score: unknown;
  baseline_score: number;
  matched_skills: string[];
  missing_skills: string[];
  nice_matches: string[];
  llm_rationale: string | null;
  suggestions: string[];
  extraction_path: string;
  llm_status: string;
  request_id: string;
  breakdown: Record<string, unknown>;
  documents: {
    jd: { title: string | null; source_text: string; requirements: RawSpanFixture[] };
    resume: { name: string | null; source_text: string; skills: RawSpanFixture[] };
  };
  [key: string]: unknown;
}

function validMatch(overrides: Record<string, unknown> = {}): MatchFixture {
  return {
    match_score: 0.6,
    baseline_score: 0.6,
    matched_skills: ['python'],
    missing_skills: ['docker'],
    nice_matches: [],
    llm_rationale: 'Good overlap.',
    suggestions: ['Add Docker experience.'],
    extraction_path: 'hybrid',
    llm_status: 'ok',
    request_id: 'req-1',
    breakdown: {
      must_matched: ['python'],
      must_missing: ['docker'],
      must_total: 2,
      nice_matched: [],
      nice_missing: ['redis'],
      nice_total: 1,
    },
    documents: {
      jd: {
        title: 'Backend Engineer',
        source_text: JD_TEXT,
        requirements: [
          { skill: 'python', priority: 'must' as const, evidence: '- Strong Python', ...JD_PYTHON },
          { skill: 'docker', priority: 'must' as const, evidence: '- Docker', ...JD_DOCKER },
        ],
      },
      resume: {
        name: 'Alex',
        source_text: RESUME_TEXT,
        skills: [
          { skill: 'python', priority: null, evidence: 'Python, Redis', ...RESUME_PYTHON },
        ],
      },
    },
    ...overrides,
  };
}

describe('scoreLabel', () => {
  it.each([
    [0.95, 'Very strong match'],
    [0.75, 'Strong match'],
    [0.55, 'Partial match'],
    [0.35, 'Weak match'],
    [0.1, 'Poor match'],
    [0, 'Poor match'],
  ])('maps %s to %s', (score, label) => {
    expect(scoreLabel(score)).toBe(label);
  });
});

describe('adaptMatchResponse', () => {
  it('maps a valid response into the UI model', () => {
    const view = adaptMatchResponse(validMatch());

    expect(view.scorePercent).toBe(60);
    expect(view.scoreLabel).toBe('Partial match');
    expect(view.jobTitle).toBe('Backend Engineer');
    expect(view.requestId).toBe('req-1');
    expect(view.extractionPath).toBe('hybrid');
    expect(view.llmStatus).toBe('ok');
    expect(view.rationale).toBe('Good overlap.');
  });

  it('groups chips by priority and match state', () => {
    const view = adaptMatchResponse(validMatch());

    expect(view.matchedMust.map((c) => c.skill)).toEqual(['python']);
    expect(view.missingMust.map((c) => c.skill)).toEqual(['docker']);
    expect(view.matchedNice).toEqual([]);
    expect(view.missingNice.map((c) => c.skill)).toEqual(['redis']);
  });

  it('reports matched-of-total counts', () => {
    expect(adaptMatchResponse(validMatch()).breakdown).toEqual({
      mustMatched: 1,
      mustTotal: 2,
      niceMatched: 0,
      niceTotal: 1,
    });
  });

  it('attaches spans from both documents to a chip', () => {
    const chip = adaptMatchResponse(validMatch()).matchedMust[0]!;
    expect(chip.jdSpan?.start).toBe(JD_PYTHON.start);
    expect(chip.resumeSpan?.start).toBe(RESUME_PYTHON.start);
  });

  it('offsets slice to the expected words', () => {
    const view = adaptMatchResponse(validMatch());
    const jdSpan = view.matchedMust[0]!.jdSpan!;
    expect(view.jd.text.slice(jdSpan.start!, jdSpan.end!)).toBe('Python');
  });

  it('nulls offsets that run past the end of the text', () => {
    const payload = validMatch();
    // A stale or buggy backend could report an out-of-range span. Highlighting
    // it would silently render nothing, so it is treated as unlocatable.
    payload.documents.jd.requirements[0]!.end = 99999;
    const chip = adaptMatchResponse(payload).matchedMust[0]!;
    expect(chip.jdSpan?.start).toBeNull();
    expect(chip.jdSpan?.end).toBeNull();
  });

  it('nulls inverted offsets', () => {
    const payload = validMatch();
    payload.documents.jd.requirements[0]!.start = 30;
    payload.documents.jd.requirements[0]!.end = 10;
    expect(adaptMatchResponse(payload).matchedMust[0]!.jdSpan?.start).toBeNull();
  });

  it('accepts null offsets for a skill with no locatable evidence', () => {
    const payload = validMatch();
    payload.documents.jd.requirements[0] = {
      skill: 'python',
      priority: 'must',
      evidence: '',
      start: null,
      end: null,
    };
    const chip = adaptMatchResponse(payload).matchedMust[0]!;
    expect(chip.skill).toBe('python');
    expect(chip.jdSpan?.start).toBeNull();
  });

  it('tolerates missing optional fields', () => {
    const payload = validMatch({ llm_rationale: null, suggestions: [] });
    const view = adaptMatchResponse(payload);
    expect(view.rationale).toBeNull();
    expect(view.suggestions).toEqual([]);
  });

  it('ignores extra unknown fields', () => {
    const view = adaptMatchResponse(validMatch({ some_future_field: { nested: true } }));
    expect(view.scorePercent).toBe(60);
  });

  it('caps suggestions at three', () => {
    const view = adaptMatchResponse(
      validMatch({ suggestions: ['a', 'b', 'c', 'd', 'e'] }),
    );
    expect(view.suggestions).toHaveLength(3);
  });

  it('clamps a score outside the unit range', () => {
    expect(adaptMatchResponse(validMatch({ match_score: 1.4 })).scorePercent).toBe(100);
    expect(adaptMatchResponse(validMatch({ match_score: -0.2 })).scorePercent).toBe(0);
  });

  it('throws a typed error when a required field is missing', () => {
    const payload = validMatch();
    delete (payload as Partial<MatchFixture>).breakdown;
    expect(() => adaptMatchResponse(payload)).toThrow(AdapterError);
    try {
      adaptMatchResponse(payload);
    } catch (error) {
      expect((error as AdapterError).appError.kind).toBe('malformed');
    }
  });

  it('throws a typed error on a wrong field type', () => {
    expect(() => adaptMatchResponse(validMatch({ match_score: 'high' }))).toThrow(
      AdapterError,
    );
  });

  it('throws a typed error on an unknown extraction path', () => {
    expect(() => adaptMatchResponse(validMatch({ extraction_path: 'magic' }))).toThrow(
      AdapterError,
    );
  });

  it('throws a typed error on a null payload', () => {
    expect(() => adaptMatchResponse(null)).toThrow(AdapterError);
    expect(() => adaptMatchResponse('nonsense')).toThrow(AdapterError);
  });
});

describe('adaptExamplesResponse', () => {
  it('maps examples into the UI model', () => {
    const examples = adaptExamplesResponse({
      examples: [
        {
          id: 'a',
          label: 'Backend',
          description: 'desc',
          job_description: 'jd',
          resume_text: 'resume',
        },
      ],
    });
    expect(examples).toEqual([
      {
        id: 'a',
        label: 'Backend',
        description: 'desc',
        jobDescription: 'jd',
        resumeText: 'resume',
      },
    ]);
  });

  it('returns an empty list when there are no examples', () => {
    expect(adaptExamplesResponse({ examples: [] })).toEqual([]);
  });

  it('throws a typed error on a malformed payload', () => {
    expect(() => adaptExamplesResponse({ examples: [{ id: 1 }] })).toThrow(AdapterError);
  });
});

describe('adaptErrorResponse', () => {
  it('surfaces a 400 detail message to the user', () => {
    const error = adaptErrorResponse(400, {
      detail: { error: 'File must be a PDF', request_id: 'req-9' },
    });
    expect(error.kind).toBe('validation');
    expect(error.message).toBe('File must be a PDF');
    expect(error.requestId).toBe('req-9');
  });

  it('handles a plain string detail', () => {
    expect(adaptErrorResponse(400, { detail: 'Bad input' }).message).toBe('Bad input');
  });

  it('falls back to generic wording for a 400 with no detail', () => {
    expect(adaptErrorResponse(400, {}).message).toContain('rejected');
  });

  it('never surfaces server internals on a 500', () => {
    const error = adaptErrorResponse(500, {
      detail: {
        error: 'Internal server error',
        message: 'Traceback: secret-key at 10.0.0.5',
        request_id: 'req-5',
      },
    });
    expect(error.kind).toBe('server');
    expect(error.message).not.toContain('secret-key');
    expect(error.message).not.toContain('Traceback');
    expect(error.requestId).toBe('req-5');
  });

  it('handles an unparseable error body', () => {
    expect(adaptErrorResponse(500, null).kind).toBe('server');
    expect(adaptErrorResponse(503, 'gateway down').kind).toBe('server');
  });
});
