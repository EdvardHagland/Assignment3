# Annotation Plan

## Goal

Create a small, reliable labeled dataset of defense-sector risk disclosures to support supervised classification.

## Recommended workflow for 5 annotators

### Phase 1: Pilot

- Sample `120` chunks
- All `5` annotators label the same `120`
- Compare disagreements
- Revise the codebook

This phase is for:

- tightening category definitions
- spotting confusing chunk boundaries
- improving coder agreement before the main round

### Phase 2: Main annotation

- Sample `600` additional chunks
- Each chunk is labeled by `2` coders
- Track coder IDs in the output

This yields:

- `120` fully shared pilot rows
- `600` double-coded main rows
- a solid labeled training set after adjudication

### Phase 3: Adjudication

- Resolve disagreements on the pilot and main overlap
- Assign a final label for each row
- Preserve original coder labels for transparency

## Suggested workload

Approximate annotation acts:

- pilot: `120 x 5 = 600`
- main: `600 x 2 = 1200`
- total: `1800` annotation acts

Across `5` annotators, that averages:

- `360` annotation acts each

This is substantial but realistic for a group project.

## Reliability reporting

For the final notebook/report, you can state:

- the pilot sample was fully overlapping
- the codebook was revised after the pilot
- the main set used double-coding
- disagreements were resolved by adjudication or consensus

If you calculate an agreement metric, report it. If not, still document the overlap and adjudication process clearly.

## Label storage

Use the template in [labels_template.csv](/C:/Users/edvar/OneDrive/Skrivebord/Assignment%203/annotation/labels_template.csv).

Recommended columns:

- `chunk_id`
- `ticker`
- `filing_date`
- `coder_id`
- `primary_label`
- `secondary_label`
- `confidence`
- `notes`

## Practical note

Keep labels broad. If the team tries to use too many narrow categories, overlap quality will drop and adjudication will become painful.
