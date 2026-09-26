# Evaluation set

This directory measures how well the system extracts skills and requirements.
It is the only thing that turns "the parser feels better now" into a number.

Gold labels are written by a human. The two cases shipped here have **empty**
gold lists on purpose, so nothing in `eval/results/` is meaningful until they
are labeled.

## Layout

```
eval/
  data/
    <case_id>/
      jd.txt        job description, as plain text
      resume.txt    resume, as plain text
      resume.pdf    optional, only if you also want to exercise PDF extraction
      gold.json     human written labels
  results/
    deterministic.json
    llm.json
    hybrid.json
  run_eval.py
```

A case id is a directory name. Use something readable, for example
`case_003_data_engineer`.

## gold.json

```json
{
  "notes": "free text, ignored by the scorer",
  "jd_must_have": ["python", "postgresql", "docker"],
  "jd_nice_to_have": ["kubernetes", "kafka"],
  "resume_skills": ["python", "fastapi", "postgresql", "docker"]
}
```

Three fields, each a list of **canonical** skill names. `notes` is optional and
ignored. Any other key is an error, so a typo fails loudly instead of being
silently dropped.

Labels are compared case-insensitively after trimming, so `"Python"` and
`"python"` are the same label. Write them lowercase anyway for consistency.

## How to label a case

1. **Use canonical names from the ontology.** Open
   `core/normalize/skill_ontology.json` and use the top level key, not a
   variant. Write `javascript`, not `JS`. Write `kubernetes`, not `k8s`.
2. **A skill the ontology does not know still gets labeled**, spelled the way
   the document spells it, lowercased. Extraction is allowed to pass unknown
   skills through unnormalized, so the label has to match that. Missing
   ontology coverage is a real finding worth recording in `notes`.
3. **Label what the document says, not what you infer.** If the JD never
   mentions testing, `testing` is not a must-have even when it is obviously
   expected of the role.
4. **`jd_must_have` and `jd_nice_to_have` must be disjoint.** If a JD mentions
   a skill in both sections, label it must-have only. That matches the rule the
   parser follows, so the two cannot disagree on a technicality.
5. **Label only skills, not requirements in general.** "5+ years of
   experience" and "bachelor's degree" are not skills and belong in neither
   list. Years of experience is not something this system scores.
6. **`resume_skills` is everything the resume evidences**, whether or not the
   JD asks for it. Include skills named in experience bullets, not just the
   skills section. The scorer only ever asks whether a JD skill is present, so
   extra resume skills cannot inflate a score.
7. **When you are genuinely unsure, leave it out and say so in `notes`.** A
   borderline label that a reasonable person would disagree with costs more
   than the coverage it adds.

### Sizing

Aim for 15 to 25 cases. Fewer than about 15 and a single case swings the
numbers; more than about 25 and labeling drifts as you get tired, which is its
own bias. Vary the roles: backend, frontend, data, and at least a couple of
deliberately poor matches so precision has something to fail on.

Do not use real resumes. Use synthetic ones, and keep names and contact
details obviously fake, because these files are committed and two of them are
served publicly by the examples endpoint.

## Running

```bash
python eval/run_eval.py --mode deterministic
python eval/run_eval.py --mode llm          # needs LLM_API_KEY
python eval/run_eval.py --mode hybrid       # needs LLM_API_KEY
```

Each run prints a markdown table and writes `eval/results/<mode>.json`.

Useful flags:

| Flag | Effect |
| --- | --- |
| `--min-f1 0.55` | exit non-zero if micro F1 drops below the threshold, used by CI |
| `--no-write` | print the report without touching `eval/results/` |
| `--data-dir` | point at a different case directory |

## What the numbers mean

Per field, and micro-averaged across all three fields:

- **Precision** is the share of extracted skills that are correct. Low
  precision means the parser is inventing skills, which is how a candidate gets
  credit for something they never mentioned.
- **Recall** is the share of gold skills that were found. Low recall means the
  parser is missing skills, which is how a good candidate gets rejected.
- **F1** is their harmonic mean.

**Must/nice classification accuracy** is scored only over skills that appear in
both the gold labels and the prediction, since a skill that was never extracted
cannot be misclassified. It answers a separate question from F1: given that we
found a requirement, did we get its priority right? Priority drives the 2x
weighting in the score, so this can be the difference between a pass and a
reject even when F1 looks fine.

Unlabeled cases are counted and reported, but excluded from every metric.
