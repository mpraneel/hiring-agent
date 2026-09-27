/**
 * Zod schemas describing the raw API responses.
 *
 * These types are private to the adapter. Components import the UI model from
 * `types.ts` instead, so a change in the wire format cannot ripple into the view
 * layer, and a malformed response becomes a typed error rather than a crash
 * halfway through rendering.
 */
import { z } from 'zod';

export const spanSchema = z.object({
  skill: z.string(),
  priority: z.enum(['must', 'nice']).nullish(),
  evidence: z.string().default(''),
  start: z.number().int().nullish(),
  end: z.number().int().nullish(),
});

export const documentsSchema = z.object({
  jd: z.object({
    title: z.string().nullish(),
    source_text: z.string().default(''),
    requirements: z.array(spanSchema).default([]),
  }),
  resume: z.object({
    name: z.string().nullish(),
    source_text: z.string().default(''),
    skills: z.array(spanSchema).default([]),
  }),
});

export const breakdownSchema = z.object({
  must_matched: z.array(z.string()).default([]),
  must_missing: z.array(z.string()).default([]),
  must_total: z.number().int().default(0),
  nice_matched: z.array(z.string()).default([]),
  nice_missing: z.array(z.string()).default([]),
  nice_total: z.number().int().default(0),
});

export const matchResponseSchema = z.object({
  match_score: z.number(),
  baseline_score: z.number(),
  matched_skills: z.array(z.string()).default([]),
  missing_skills: z.array(z.string()).default([]),
  nice_matches: z.array(z.string()).default([]),
  llm_rationale: z.string().nullish(),
  suggestions: z.array(z.string()).default([]),
  extraction_path: z.enum(['llm', 'deterministic', 'fallback', 'hybrid']),
  llm_status: z.enum(['ok', 'unavailable', 'disabled']),
  request_id: z.string(),
  breakdown: breakdownSchema,
  documents: documentsSchema,
});

export const exampleSchema = z.object({
  id: z.string(),
  label: z.string(),
  description: z.string().default(''),
  job_description: z.string(),
  resume_text: z.string(),
});

export const examplesResponseSchema = z.object({
  examples: z.array(exampleSchema).default([]),
});

export const errorDetailSchema = z.object({
  error: z.string().optional(),
  message: z.string().optional(),
  request_id: z.string().optional(),
});

export const errorResponseSchema = z.object({
  detail: z.union([errorDetailSchema, z.string()]).optional(),
  error: z.string().optional(),
  request_id: z.string().optional(),
  message: z.string().optional(),
});
