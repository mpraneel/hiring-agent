/**
 * HTTP transport. Everything it returns has already passed through the adapter.
 */
import {
  AdapterError,
  adaptErrorResponse,
  adaptExamplesResponse,
  adaptMatchResponse,
} from './adapter';
import type { ExampleView, MatchView } from './types';

export interface MatchRequest {
  jobDescription: string;
  resumeFile?: File | null;
  resumeText?: string | null;
}

function networkError(): AdapterError {
  return new AdapterError({
    kind: 'network',
    message: 'Could not reach the server. Check your connection and try again.',
  });
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

export async function requestMatch(
  request: MatchRequest,
  signal?: AbortSignal,
): Promise<MatchView> {
  const form = new FormData();
  form.append('job_description', request.jobDescription);
  if (request.resumeFile) {
    form.append('resume_pdf', request.resumeFile);
  } else if (request.resumeText) {
    form.append('resume_text', request.resumeText);
  }

  let response: Response;
  try {
    response = await fetch('/api/v1/match', { method: 'POST', body: form, signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new AdapterError({ kind: 'cancelled', message: 'Analysis cancelled.' });
    }
    throw networkError();
  }

  const payload = await readJson(response);
  if (!response.ok) {
    throw new AdapterError(adaptErrorResponse(response.status, payload));
  }
  return adaptMatchResponse(payload);
}

export async function fetchExamples(signal?: AbortSignal): Promise<ExampleView[]> {
  let response: Response;
  try {
    response = await fetch('/api/v1/examples', { signal });
  } catch {
    throw networkError();
  }

  const payload = await readJson(response);
  if (!response.ok) {
    throw new AdapterError(adaptErrorResponse(response.status, payload));
  }
  return adaptExamplesResponse(payload);
}
