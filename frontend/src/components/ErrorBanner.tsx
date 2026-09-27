import type { AppError } from '../services/types';

const TITLES: Record<AppError['kind'], string> = {
  network: 'Could not reach the server',
  validation: 'Check the inputs',
  server: 'The server hit a problem',
  malformed: 'Unexpected reply from the server',
  cancelled: 'Analysis cancelled',
};

interface Props {
  error: AppError;
  onRetry?: () => void;
  onDismiss?: () => void;
}

/**
 * Each failure kind gets its own wording. A request id is shown only for server
 * errors, where it is the thing that makes a bug report actionable.
 */
export default function ErrorBanner({ error, onRetry, onDismiss }: Props) {
  return (
    <div role="alert" className="card border-amber-600/50 bg-amber-600/5">
      <h2 className="flex items-center gap-2 text-base font-semibold">
        <span aria-hidden="true">⚠</span>
        {TITLES[error.kind]}
      </h2>
      <p className="mt-1 text-sm">{error.message}</p>
      {error.requestId && error.kind === 'server' && (
        <p className="mt-2 text-xs text-ink-muted">
          Quote request {error.requestId} if you report this.
        </p>
      )}
      <div className="mt-3 flex flex-wrap gap-2">
        {onRetry && (
          <button type="button" className="btn-secondary" onClick={onRetry}>
            Try again
          </button>
        )}
        {onDismiss && (
          <button type="button" className="btn-secondary" onClick={onDismiss}>
            Dismiss
          </button>
        )}
      </div>
    </div>
  );
}
