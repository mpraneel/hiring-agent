import { useCallback, useEffect, useRef, useState } from 'react';

import ErrorBanner from './components/ErrorBanner';
import InputView, { emptyInput, hasValidInput, type InputState } from './components/InputView';
import LoadingView from './components/LoadingView';
import ResultsView from './components/ResultsView';
import { AdapterError } from './services/adapter';
import { fetchExamples, requestMatch } from './services/api';
import type { AppError, ExampleView, MatchView } from './services/types';

type Screen = 'input' | 'loading' | 'results';

function toAppError(error: unknown): AppError {
  if (error instanceof AdapterError) return error.appError;
  return {
    kind: 'network',
    message: 'Something unexpected went wrong. Try again.',
  };
}

export default function App() {
  const [screen, setScreen] = useState<Screen>('input');
  const [input, setInput] = useState<InputState>(emptyInput);
  const [result, setResult] = useState<MatchView | null>(null);
  const [error, setError] = useState<AppError | null>(null);
  const [examples, setExamples] = useState<ExampleView[]>([]);
  const [examplesError, setExamplesError] = useState<AppError | null>(null);
  const [loadingExamples, setLoadingExamples] = useState(true);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchExamples(controller.signal)
      .then(setExamples)
      .catch((cause) => {
        if (controller.signal.aborted) return;
        setExamplesError(toAppError(cause));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingExamples(false);
      });
    return () => controller.abort();
  }, []);

  const submit = useCallback(async () => {
    if (!hasValidInput(input)) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setError(null);
    setScreen('loading');

    try {
      const view = await requestMatch(
        {
          jobDescription: input.jobDescription,
          resumeFile: input.mode === 'upload' ? input.resumeFile : null,
          resumeText: input.mode === 'paste' ? input.resumeText : null,
        },
        controller.signal,
      );
      setResult(view);
      setScreen('results');
    } catch (cause) {
      const appError = toAppError(cause);
      // A cancel is a deliberate act, not a failure worth a banner.
      setError(appError.kind === 'cancelled' ? null : appError);
      setScreen('input');
    } finally {
      abortRef.current = null;
    }
  }, [input]);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    setScreen('input');
  }, []);

  const useExample = useCallback((example: ExampleView) => {
    setError(null);
    setInput({
      jobDescription: example.jobDescription,
      resumeText: example.resumeText,
      resumeFile: null,
      mode: 'paste',
    });
  }, []);

  return (
    <div className="mx-auto min-h-screen max-w-5xl px-4 py-8 sm:px-6">
      <header className="mb-8">
        <h1 className="text-2xl font-bold tracking-tight">Hiring Agent</h1>
        <p className="mt-1 max-w-2xl text-sm text-ink-muted">
          Match a resume against a job description. The score is computed from the
          extracted skills, so it is reproducible, and every skill links back to the
          text it came from.
        </p>
      </header>

      <main>
        {error && (
          <div className="mb-6">
            <ErrorBanner
              error={error}
              onRetry={screen === 'input' && hasValidInput(input) ? submit : undefined}
              onDismiss={() => setError(null)}
            />
          </div>
        )}

        {screen === 'input' && (
          <InputView
            value={input}
            onChange={setInput}
            onSubmit={submit}
            examples={examples}
            examplesError={examplesError}
            loadingExamples={loadingExamples}
            onUseExample={useExample}
          />
        )}

        {screen === 'loading' && <LoadingView onCancel={cancel} />}

        {screen === 'results' && result && (
          <ResultsView
            result={result}
            onEditInputs={() => setScreen('input')}
            onStartOver={() => {
              setInput(emptyInput);
              setResult(null);
              setScreen('input');
            }}
          />
        )}
      </main>

      <footer className="mt-12 border-t border-line pt-4 text-xs text-ink-muted">
        Keyword presence is not the same as depth of experience. Treat the score as a
        screening aid, not a decision.
      </footer>
    </div>
  );
}
