/**
 * The UI model.
 *
 * Components only ever see these types. They are deliberately shaped for
 * rendering rather than mirroring the wire format: chips are grouped the way the
 * results view displays them, and every span carries offsets that are known to
 * be usable.
 */

export type ExtractionPath = 'llm' | 'deterministic' | 'fallback' | 'hybrid';
export type LlmStatus = 'ok' | 'unavailable' | 'disabled';
export type Priority = 'must' | 'nice';

/** Which document a highlight belongs to. */
export type DocumentKind = 'jd' | 'resume';

export interface EvidenceSpan {
  /** Canonical skill name. */
  skill: string;
  /** The line of text the skill was found on. */
  evidence: string;
  /** Character offsets into the owning document's text, when locatable. */
  start: number | null;
  end: number | null;
}

export interface SkillChip {
  skill: string;
  priority: Priority;
  matched: boolean;
  /** Where this skill is evidenced in the job description, if anywhere. */
  jdSpan: EvidenceSpan | null;
  /** Where this skill is evidenced in the resume, if anywhere. */
  resumeSpan: EvidenceSpan | null;
}

export interface SourceDocument {
  kind: DocumentKind;
  heading: string;
  text: string;
}

export interface MatchBreakdown {
  mustMatched: number;
  mustTotal: number;
  niceMatched: number;
  niceTotal: number;
}

export interface MatchView {
  score: number;
  scorePercent: number;
  /** A short human label for the score band. */
  scoreLabel: string;
  breakdown: MatchBreakdown;
  matchedMust: SkillChip[];
  missingMust: SkillChip[];
  matchedNice: SkillChip[];
  missingNice: SkillChip[];
  rationale: string | null;
  suggestions: string[];
  extractionPath: ExtractionPath;
  llmStatus: LlmStatus;
  requestId: string;
  jd: SourceDocument;
  resume: SourceDocument;
  jobTitle: string | null;
}

export interface ExampleView {
  id: string;
  label: string;
  description: string;
  jobDescription: string;
  resumeText: string;
}

/** Every failure the UI distinguishes. */
export type AppErrorKind = 'network' | 'validation' | 'server' | 'malformed' | 'cancelled';

export interface AppError {
  kind: AppErrorKind;
  /** A complete sentence, safe to show a user. */
  message: string;
  /** Present for server errors, so a user can quote it in a bug report. */
  requestId?: string;
}
