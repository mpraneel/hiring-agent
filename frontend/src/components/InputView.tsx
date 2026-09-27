import { useCallback, useId, useRef, useState } from 'react';

import type { AppError, ExampleView } from '../services/types';

const MAX_PDF_BYTES = 5 * 1024 * 1024;

export interface InputState {
  jobDescription: string;
  resumeText: string;
  resumeFile: File | null;
  mode: 'upload' | 'paste';
}

export const emptyInput: InputState = {
  jobDescription: '',
  resumeText: '',
  resumeFile: null,
  mode: 'upload',
};

interface Props {
  value: InputState;
  onChange: (next: InputState) => void;
  onSubmit: () => void;
  examples: ExampleView[];
  examplesError: AppError | null;
  loadingExamples: boolean;
  onUseExample: (example: ExampleView) => void;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Client-side check so an obviously bad file never costs a round trip. */
export function validateFile(file: File): string | null {
  const isPdf =
    file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
  if (!isPdf) return 'That file is not a PDF. Upload a PDF, or paste the resume text instead.';
  if (file.size > MAX_PDF_BYTES) {
    return `That file is ${formatBytes(file.size)}. The limit is 5 MB.`;
  }
  if (file.size === 0) return 'That file is empty.';
  return null;
}

export function hasValidInput(state: InputState): boolean {
  const resumeReady =
    state.mode === 'upload' ? state.resumeFile !== null : state.resumeText.trim().length > 0;
  return resumeReady && state.jobDescription.trim().length > 0;
}

export default function InputView({
  value,
  onChange,
  onSubmit,
  examples,
  examplesError,
  loadingExamples,
  onUseExample,
}: Props) {
  const [fileError, setFileError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropZoneId = useId();
  const jdId = useId();
  const resumeTextId = useId();

  const acceptFile = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      const problem = validateFile(file);
      if (problem) {
        setFileError(problem);
        onChange({ ...value, resumeFile: null });
        return;
      }
      setFileError(null);
      onChange({ ...value, resumeFile: file });
    },
    [onChange, value],
  );

  const openFilePicker = () => fileInputRef.current?.click();

  const ready = hasValidInput(value);
  const jdCount = value.jobDescription.length;

  return (
    <div className="space-y-6">
      <section className="card" aria-labelledby="try-heading">
        <h2 id="try-heading" className="text-base font-semibold">
          New here? Start with an example
        </h2>
        <p className="mt-1 text-sm text-ink-muted">
          Loads a real job description and resume so you can see the whole flow before
          uploading anything of your own.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          {loadingExamples && (
            <span className="text-sm text-ink-muted">Loading examples…</span>
          )}
          {examplesError && (
            <p role="status" className="text-sm text-ink-muted">
              {examplesError.message}
            </p>
          )}
          {examples.map((example) => (
            <button
              key={example.id}
              type="button"
              className="btn-primary"
              onClick={() => onUseExample(example)}
            >
              Try an example: {example.label}
            </button>
          ))}
        </div>
      </section>

      <section className="card" aria-labelledby="resume-heading">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 id="resume-heading" className="text-base font-semibold">
            Resume
          </h2>
          <div className="flex gap-1 rounded-lg border border-line p-1" role="group" aria-label="Resume input method">
            <button
              type="button"
              aria-pressed={value.mode === 'upload'}
              className={`rounded px-3 py-1 text-sm ${
                value.mode === 'upload' ? 'bg-accent text-white' : 'text-ink-muted'
              }`}
              onClick={() => onChange({ ...value, mode: 'upload' })}
            >
              Upload PDF
            </button>
            <button
              type="button"
              aria-pressed={value.mode === 'paste'}
              className={`rounded px-3 py-1 text-sm ${
                value.mode === 'paste' ? 'bg-accent text-white' : 'text-ink-muted'
              }`}
              onClick={() => onChange({ ...value, mode: 'paste' })}
            >
              Paste text instead
            </button>
          </div>
        </div>

        {value.mode === 'upload' ? (
          <div className="mt-4">
            {/* A button, not a div: that makes the drop zone reachable by Tab and
                operable with Enter or Space without extra key handlers. */}
            <button
              type="button"
              id={dropZoneId}
              onClick={openFilePicker}
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                acceptFile(event.dataTransfer.files[0]);
              }}
              aria-describedby={`${dropZoneId}-hint`}
              className={`flex w-full flex-col items-center justify-center gap-2 rounded-lg
                border-2 border-dashed px-4 py-8 text-center transition-colors ${
                  dragging ? 'border-accent bg-accent/5' : 'border-line hover:border-accent'
                }`}
            >
              <span aria-hidden="true" className="text-2xl">
                📄
              </span>
              <span className="text-sm font-medium">
                {value.resumeFile
                  ? value.resumeFile.name
                  : 'Drop a PDF here, or click to browse'}
              </span>
              <span id={`${dropZoneId}-hint`} className="text-xs text-ink-muted">
                {value.resumeFile
                  ? `${formatBytes(value.resumeFile.size)} selected`
                  : 'PDF only, up to 5 MB'}
              </span>
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="application/pdf,.pdf"
              className="sr-only"
              aria-label="Resume PDF"
              onChange={(event) => acceptFile(event.target.files?.[0])}
            />
            {value.resumeFile && (
              <button
                type="button"
                className="mt-2 text-sm text-accent underline"
                onClick={() => {
                  setFileError(null);
                  onChange({ ...value, resumeFile: null });
                }}
              >
                Remove {value.resumeFile.name}
              </button>
            )}
            {fileError && (
              <p role="alert" className="mt-2 text-sm text-ink">
                <span aria-hidden="true">⚠ </span>
                {fileError}
              </p>
            )}
          </div>
        ) : (
          <div className="mt-4">
            <label htmlFor={resumeTextId} className="text-sm text-ink-muted">
              Paste the resume text
            </label>
            <textarea
              id={resumeTextId}
              value={value.resumeText}
              onChange={(event) => onChange({ ...value, resumeText: event.target.value })}
              rows={10}
              placeholder={'Jane Doe\n\nSkills\nPython, Docker, React\n\nExperience\n…'}
              className="mt-1 w-full rounded-lg border border-line bg-surface p-3 font-mono text-sm"
            />
            <p className="text-xs text-ink-muted">
              {value.resumeText.length.toLocaleString()} characters
            </p>
          </div>
        )}
      </section>

      <section className="card" aria-labelledby="jd-heading">
        <h2 id="jd-heading" className="text-base font-semibold">
          Job description
        </h2>
        <label htmlFor={jdId} className="mt-1 block text-sm text-ink-muted">
          Paste the full posting. Section headings such as Requirements and Nice to have
          help the parser tell hard requirements from preferences.
        </label>
        <textarea
          id={jdId}
          value={value.jobDescription}
          onChange={(event) => onChange({ ...value, jobDescription: event.target.value })}
          rows={10}
          placeholder={'Senior Backend Engineer\n\nRequirements\n- Strong Python\n\nNice to have\n- Kubernetes'}
          className="mt-2 w-full rounded-lg border border-line bg-surface p-3 font-mono text-sm"
        />
        <p className="text-xs text-ink-muted">
          {jdCount.toLocaleString()} characters
        </p>
      </section>

      <div className="flex flex-wrap items-center gap-3">
        <button type="button" className="btn-primary" disabled={!ready} onClick={onSubmit}>
          Analyse match
        </button>
        {!ready && (
          <p className="text-sm text-ink-muted">
            Add a resume and a job description to continue.
          </p>
        )}
      </div>
    </div>
  );
}
