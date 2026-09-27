/**
 * Component tests.
 *
 * The API is mocked at the fetch boundary, so no backend is needed. Coverage
 * targets what a user actually depends on: input validation, the example flow,
 * the evidence highlight, and every error state.
 */
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import App from '../App';
import ErrorBanner from './ErrorBanner';
import ResultsView from './ResultsView';
import { validateFile } from './InputView';
import type { MatchView, SkillChip } from '../services/types';

const JD_TEXT = 'Requirements\n- Strong Python\n- Docker required';
const RESUME_TEXT = 'Skills\nPython, Redis';

/** Locate a skill in text case-insensitively, mirroring what the backend does. */
function spanFor(text: string, skill: string) {
  const start = text.toLowerCase().indexOf(skill.toLowerCase());
  if (start < 0) return null;
  return { skill, evidence: text, start, end: start + skill.length };
}

function chip(skill: string, priority: 'must' | 'nice', matched: boolean): SkillChip {
  return {
    skill,
    priority,
    matched,
    jdSpan: spanFor(JD_TEXT, skill),
    resumeSpan: spanFor(RESUME_TEXT, skill),
  };
}

function matchView(overrides: Partial<MatchView> = {}): MatchView {
  return {
    score: 0.6,
    scorePercent: 60,
    scoreLabel: 'Partial match',
    breakdown: { mustMatched: 1, mustTotal: 2, niceMatched: 0, niceTotal: 1 },
    matchedMust: [chip('python', 'must', true)],
    missingMust: [chip('docker', 'must', false)],
    matchedNice: [],
    missingNice: [chip('kafka', 'nice', false)],
    rationale: 'Solid Python background, but Docker is missing.',
    suggestions: ['Add a bullet about Docker deployment.'],
    extractionPath: 'hybrid',
    llmStatus: 'ok',
    requestId: 'req-42',
    jobTitle: 'Backend Engineer',
    jd: { kind: 'jd', heading: 'Job description', text: JD_TEXT },
    resume: { kind: 'resume', heading: 'Resume', text: RESUME_TEXT },
    ...overrides,
  };
}

const EXAMPLES_PAYLOAD = {
  examples: [
    {
      id: 'backend',
      label: 'Backend engineer',
      description: 'A backend role.',
      job_description: JD_TEXT,
      resume_text: RESUME_TEXT,
    },
  ],
};

function mockFetch(handlers: {
  examples?: () => Response | Promise<Response>;
  match?: () => Response | Promise<Response>;
}) {
  const fetchMock = vi.fn((input: RequestInfo | URL, _init?: RequestInit) => {
    const url = String(input);
    if (url.includes('/examples')) {
      return Promise.resolve(
        handlers.examples?.() ?? jsonResponse(EXAMPLES_PAYLOAD),
      );
    }
    if (url.includes('/match')) {
      if (!handlers.match) throw new Error('unexpected match call');
      return Promise.resolve(handlers.match());
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function validMatchPayload() {
  return {
    match_score: 0.6,
    baseline_score: 0.6,
    matched_skills: ['python'],
    missing_skills: ['docker'],
    nice_matches: [],
    llm_rationale: 'Solid Python background.',
    suggestions: ['Add a bullet about Docker.'],
    extraction_path: 'deterministic',
    llm_status: 'disabled',
    request_id: 'req-1',
    breakdown: {
      must_matched: ['python'],
      must_missing: ['docker'],
      must_total: 2,
      nice_matched: [],
      nice_missing: [],
      nice_total: 0,
    },
    documents: {
      jd: { title: 'Backend', source_text: JD_TEXT, requirements: [] },
      resume: { name: null, source_text: RESUME_TEXT, skills: [] },
    },
  };
}

beforeEach(() => {
  vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------
// File validation
// ---------------------------------------------------------------------------

describe('validateFile', () => {
  const makeFile = (name: string, type: string, size: number) => {
    const file = new File(['x'], name, { type });
    Object.defineProperty(file, 'size', { value: size });
    return file;
  };

  it('accepts a reasonable PDF', () => {
    expect(validateFile(makeFile('cv.pdf', 'application/pdf', 1000))).toBeNull();
  });

  it('accepts a PDF by extension when the type is missing', () => {
    expect(validateFile(makeFile('cv.pdf', '', 1000))).toBeNull();
  });

  it('rejects a non-PDF and says what to do instead', () => {
    const message = validateFile(makeFile('cv.docx', 'application/msword', 1000));
    expect(message).toMatch(/not a PDF/i);
    expect(message).toMatch(/paste/i);
  });

  it('rejects a file over 5 MB and names the actual size', () => {
    const message = validateFile(makeFile('big.pdf', 'application/pdf', 6 * 1024 * 1024));
    expect(message).toMatch(/6\.0 MB/);
    expect(message).toMatch(/5 MB/);
  });

  it('rejects an empty file', () => {
    expect(validateFile(makeFile('empty.pdf', 'application/pdf', 0))).toMatch(/empty/i);
  });
});

// ---------------------------------------------------------------------------
// Input view
// ---------------------------------------------------------------------------

describe('input view', () => {
  it('disables submit until both inputs are present', async () => {
    mockFetch({});
    render(<App />);

    const submit = await screen.findByRole('button', { name: /analyse match/i });
    expect(submit).toBeDisabled();

    await userEvent.click(screen.getByRole('button', { name: /paste text instead/i }));
    await userEvent.type(screen.getByLabelText(/paste the resume text/i), 'Python');
    expect(submit).toBeDisabled();

    await userEvent.type(screen.getByLabelText(/paste the full posting/i), 'Need Python');
    expect(submit).toBeEnabled();
  });

  it('treats whitespace-only input as empty', async () => {
    mockFetch({});
    render(<App />);
    await userEvent.click(await screen.findByRole('button', { name: /paste text instead/i }));
    await userEvent.type(screen.getByLabelText(/paste the resume text/i), '   ');
    await userEvent.type(screen.getByLabelText(/paste the full posting/i), '   ');
    expect(screen.getByRole('button', { name: /analyse match/i })).toBeDisabled();
  });

  it('shows a character count for the job description', async () => {
    mockFetch({});
    render(<App />);
    await userEvent.type(await screen.findByLabelText(/paste the full posting/i), 'Python');
    expect(screen.getByText('6 characters')).toBeInTheDocument();
  });

  it('loads an example into both fields', async () => {
    mockFetch({});
    render(<App />);

    await userEvent.click(
      await screen.findByRole('button', { name: /try an example: backend engineer/i }),
    );

    expect(screen.getByLabelText(/paste the resume text/i)).toHaveValue(RESUME_TEXT);
    expect(screen.getByLabelText(/paste the full posting/i)).toHaveValue(JD_TEXT);
    expect(screen.getByRole('button', { name: /analyse match/i })).toBeEnabled();
  });

  it('survives the examples endpoint failing', async () => {
    mockFetch({ examples: () => jsonResponse({ detail: 'boom' }, 500) });
    render(<App />);
    // The page must still be usable without examples.
    expect(await screen.findByLabelText(/paste the full posting/i)).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.queryByRole('button', { name: /try an example/i })).not.toBeInTheDocument(),
    );
  });

  it('exposes the drop zone as a keyboard reachable button', async () => {
    mockFetch({});
    render(<App />);
    const dropZone = await screen.findByRole('button', { name: /drop a pdf here/i });
    expect(dropZone).toBeInTheDocument();
    dropZone.focus();
    expect(dropZone).toHaveFocus();
  });
});

// ---------------------------------------------------------------------------
// Results view
// ---------------------------------------------------------------------------

describe('results view', () => {
  const noop = () => {};

  it('shows the score, label and breakdown', () => {
    render(<ResultsView result={matchView()} onStartOver={noop} onEditInputs={noop} />);
    expect(screen.getByText('60%')).toBeInTheDocument();
    expect(screen.getByText('Partial match')).toBeInTheDocument();
    expect(screen.getByText('1 of 2')).toBeInTheDocument();
    expect(screen.getByText('0 of 1')).toBeInTheDocument();
  });

  it('renders all four chip groups with headings, not colour alone', () => {
    render(<ResultsView result={matchView()} onStartOver={noop} onEditInputs={noop} />);
    expect(screen.getByRole('heading', { name: /matched must-have/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /missing must-have/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /matched nice-to-have/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /missing nice-to-have/i })).toBeInTheDocument();
  });

  it('renders chips as buttons so they are keyboard reachable', () => {
    render(<ResultsView result={matchView()} onStartOver={noop} onEditInputs={noop} />);
    expect(screen.getByRole('button', { name: /python/i })).toBeInTheDocument();
  });

  it('gives an empty group real copy rather than blank space', () => {
    render(<ResultsView result={matchView()} onStartOver={noop} onEditInputs={noop} />);
    expect(screen.getByText(/none of the preferred skills were found/i)).toBeInTheDocument();
  });

  it('highlights the evidence span when a chip is selected', async () => {
    const { container } = render(
      <ResultsView result={matchView()} onStartOver={noop} onEditInputs={noop} />,
    );
    expect(container.querySelector('mark')).toBeNull();

    await userEvent.click(screen.getByRole('button', { name: /python/i }));

    const marks = Array.from(container.querySelectorAll('mark'));
    expect(marks.length).toBeGreaterThan(0);
    // The highlight must be the skill itself, sliced by backend offsets.
    expect(marks.map((m) => m.textContent)).toContain('Python');
    expect(screen.getByText(/showing where python appears/i)).toBeInTheDocument();
  });

  it('says so when a selected skill is absent from one document', async () => {
    render(<ResultsView result={matchView()} onStartOver={noop} onEditInputs={noop} />);
    await userEvent.click(screen.getByRole('button', { name: /docker/i }));
    expect(screen.getByText(/not mentioned here/i)).toBeInTheDocument();
  });

  it('shows the rationale and suggestions on the LLM path', () => {
    render(<ResultsView result={matchView()} onStartOver={noop} onEditInputs={noop} />);
    expect(screen.getByText(/solid python background/i)).toBeInTheDocument();
    expect(screen.getByText(/add a bullet about docker deployment/i)).toBeInTheDocument();
  });

  it('explains that the score stands when the LLM is unavailable', () => {
    render(
      <ResultsView
        result={matchView({ llmStatus: 'unavailable', rationale: null, suggestions: [] })}
        onStartOver={noop}
        onEditInputs={noop}
      />,
    );
    expect(screen.getByText(/explanation is unavailable/i)).toBeInTheDocument();
    expect(screen.getByText(/score above still stands/i)).toBeInTheDocument();
    // The score itself is untouched.
    expect(screen.getByText('60%')).toBeInTheDocument();
  });

  it('distinguishes no key configured from a failed call', () => {
    render(
      <ResultsView
        result={matchView({ llmStatus: 'disabled', rationale: null, suggestions: [] })}
        onStartOver={noop}
        onEditInputs={noop}
      />,
    );
    expect(screen.getByText(/no language model is configured/i)).toBeInTheDocument();
  });

  it.each([
    ['llm', /LLM extraction/i],
    ['hybrid', /Hybrid extraction/i],
    ['deterministic', /Deterministic extraction/i],
    ['fallback', /Deterministic fallback/i],
  ] as const)('badges the %s extraction path', (path, pattern) => {
    render(
      <ResultsView
        result={matchView({ extractionPath: path })}
        onStartOver={noop}
        onEditInputs={noop}
      />,
    );
    expect(screen.getAllByText(pattern).length).toBeGreaterThan(0);
  });

  it('explains what a fallback means', () => {
    render(
      <ResultsView
        result={matchView({ extractionPath: 'fallback' })}
        onStartOver={noop}
        onEditInputs={noop}
      />,
    );
    expect(screen.getByText(/did not return usable output/i)).toBeInTheDocument();
  });

  it('shows the request id for support', () => {
    render(<ResultsView result={matchView()} onStartOver={noop} onEditInputs={noop} />);
    expect(screen.getByText(/req-42/)).toBeInTheDocument();
  });

  it('offers start over and edit inputs', async () => {
    const onStartOver = vi.fn();
    const onEditInputs = vi.fn();
    render(
      <ResultsView result={matchView()} onStartOver={onStartOver} onEditInputs={onEditInputs} />,
    );
    await userEvent.click(screen.getByRole('button', { name: /edit inputs/i }));
    await userEvent.click(screen.getByRole('button', { name: /start over/i }));
    expect(onEditInputs).toHaveBeenCalledOnce();
    expect(onStartOver).toHaveBeenCalledOnce();
  });
});

// ---------------------------------------------------------------------------
// Error states
// ---------------------------------------------------------------------------

describe('error banner', () => {
  it.each([
    ['network', /could not reach the server/i],
    ['validation', /check the inputs/i],
    ['server', /server hit a problem/i],
    ['malformed', /unexpected reply/i],
  ] as const)('gives %s its own heading', (kind, pattern) => {
    render(<ErrorBanner error={{ kind, message: 'Something happened.' }} />);
    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(pattern)).toBeInTheDocument();
  });

  it('shows a request id for server errors only', () => {
    const { unmount } = render(
      <ErrorBanner error={{ kind: 'server', message: 'Failed.', requestId: 'req-7' }} />,
    );
    expect(screen.getByText(/req-7/)).toBeInTheDocument();
    unmount();

    render(
      <ErrorBanner error={{ kind: 'validation', message: 'Bad.', requestId: 'req-8' }} />,
    );
    expect(screen.queryByText(/req-8/)).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// End to end flow, API mocked
// ---------------------------------------------------------------------------

describe('flow', () => {
  async function fillAndSubmit() {
    await userEvent.click(
      await screen.findByRole('button', { name: /try an example: backend engineer/i }),
    );
    await userEvent.click(screen.getByRole('button', { name: /analyse match/i }));
  }

  it('goes from example to results', async () => {
    mockFetch({ match: () => jsonResponse(validMatchPayload()) });
    render(<App />);
    await fillAndSubmit();

    expect(await screen.findByText('60%')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /^skills$/i })).toBeInTheDocument();
  });

  it('shows staged progress while loading', async () => {
    let release: (value: Response) => void = () => {};
    const pending = new Promise<Response>((resolve) => {
      release = resolve;
    });
    mockFetch({ match: () => pending });
    render(<App />);
    await fillAndSubmit();

    expect(await screen.findByText(/reading resume/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument();

    release(jsonResponse(validMatchPayload()));
    expect(await screen.findByText('60%')).toBeInTheDocument();
  });

  it('shows a validation message for a 400', async () => {
    mockFetch({
      match: () => jsonResponse({ detail: { error: 'File must be a PDF' } }, 400),
    });
    render(<App />);
    await fillAndSubmit();

    expect(await screen.findByText(/file must be a pdf/i)).toBeInTheDocument();
    expect(screen.getByRole('alert')).toBeInTheDocument();
  });

  it('hides server internals on a 500 but keeps the request id', async () => {
    mockFetch({
      match: () =>
        jsonResponse(
          {
            detail: {
              error: 'Internal server error',
              message: 'Traceback: secret at 10.0.0.5',
              request_id: 'req-99',
            },
          },
          500,
        ),
    });
    render(<App />);
    await fillAndSubmit();

    expect(await screen.findByText(/server hit a problem/i)).toBeInTheDocument();
    expect(screen.getByText(/req-99/)).toBeInTheDocument();
    expect(screen.queryByText(/secret at 10\.0\.0\.5/)).not.toBeInTheDocument();
    expect(screen.queryByText(/traceback/i)).not.toBeInTheDocument();
  });

  it('reports a network failure', async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, _init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/examples')) return Promise.resolve(jsonResponse(EXAMPLES_PAYLOAD));
      return Promise.reject(new TypeError('Failed to fetch'));
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<App />);
    await fillAndSubmit();

    const alert = await screen.findByRole('alert');
    expect(
      within(alert).getByRole('heading', { name: /could not reach the server/i }),
    ).toBeInTheDocument();
    expect(within(alert).getByText(/check your connection/i)).toBeInTheDocument();
  });

  it('turns a malformed success response into an error state, not a blank screen', async () => {
    mockFetch({ match: () => jsonResponse({ match_score: 'lots' }) });
    render(<App />);
    await fillAndSubmit();

    expect(await screen.findByText(/unexpected reply/i)).toBeInTheDocument();
    // The inputs are still there to retry with.
    expect(screen.getByLabelText(/paste the full posting/i)).toBeInTheDocument();
  });

  it('keeps inputs when editing after a result', async () => {
    mockFetch({ match: () => jsonResponse(validMatchPayload()) });
    render(<App />);
    await fillAndSubmit();
    await screen.findByText('60%');

    await userEvent.click(screen.getByRole('button', { name: /edit inputs/i }));
    expect(screen.getByLabelText(/paste the full posting/i)).toHaveValue(JD_TEXT);
  });

  it('clears inputs on start over', async () => {
    mockFetch({ match: () => jsonResponse(validMatchPayload()) });
    render(<App />);
    await fillAndSubmit();
    await screen.findByText('60%');

    await userEvent.click(screen.getByRole('button', { name: /start over/i }));
    expect(await screen.findByLabelText(/paste the full posting/i)).toHaveValue('');
  });

  it('returns to the inputs on cancel without an error banner', async () => {
    mockFetch({ match: () => new Promise<Response>(() => {}) });
    render(<App />);
    await fillAndSubmit();

    await userEvent.click(await screen.findByRole('button', { name: /cancel/i }));
    expect(await screen.findByLabelText(/paste the full posting/i)).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('sends resume_text rather than a file when pasting', async () => {
    const fetchMock = mockFetch({ match: () => jsonResponse(validMatchPayload()) });
    render(<App />);
    await fillAndSubmit();
    await screen.findByText('60%');

    const matchCall = fetchMock.mock.calls.find(([url]) => String(url).includes('/match'));
    const body = matchCall?.[1]?.body as FormData;
    expect(body.get('resume_text')).toBe(RESUME_TEXT);
    expect(body.get('resume_pdf')).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Page furniture
// ---------------------------------------------------------------------------

describe('page', () => {
  it('has one top level heading and a limitation note', async () => {
    mockFetch({});
    render(<App />);
    expect(await screen.findByRole('heading', { level: 1 })).toHaveTextContent('Hiring Agent');
    expect(
      screen.getByText(/not the same as depth of experience/i),
    ).toBeInTheDocument();
  });

  it('labels the example section for what it does', async () => {
    mockFetch({});
    render(<App />);
    const section = await screen.findByRole('region', { name: /start with an example/i }).catch(
      () => null,
    );
    // The heading is what matters; the region role is incidental.
    expect(
      section ?? screen.getByRole('heading', { name: /start with an example/i }),
    ).toBeTruthy();
  });
});
