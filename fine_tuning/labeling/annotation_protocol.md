# Annotation Protocol

## Goal

Create a reliable supervised dataset from the SEC risk corpus without allowing single-coded or persistently ambiguous rows into the final training set.

## Team setup

This protocol assumes `6` annotators.

## Corpus versus training data

Keep these layers separate:

- `data/final/sec_defense_risk_dataset.csv` remains the full cleaned corpus
- the annotation pool is a sampled subset drawn from that corpus
- the final supervised dataset is built only after agreement checks on the labeled subset

This means annotated rows do not get removed from the original corpus. The important split comes later, when the consensus-labeled rows are divided into train, validation, and test sets.

## Rules

- every item must be labeled by at least `2` annotators
- no item enters the training set after only one label
- disagreements are recycled into another labeling round
- rows with persistent disagreement are excluded from the final training set
- rows repeatedly coded as `OTHER_UNCLEAR` should be reviewed and may be excluded from the final training set
- preserve every raw label for transparency and auditability

## Workflow

### Phase 1: Pilot calibration

- sample `90-120` items from the final corpus
- all `6` annotators label the same pilot sample
- compare disagreement patterns
- revise the codebook before the main round

Use this phase to catch:

- unclear label boundaries
- bad chunking cases
- labels that are too narrow to code reliably

### Phase 2: Main double-coding round

- sample the main annotation pool from `data/final/sec_defense_risk_dataset.csv`
- assign each item to exactly `2` annotators
- distribute batches so no annotator is overloaded and no item is single-coded
- show annotators only the text and minimal item metadata needed for workflow control
- keep the annotator action lightweight: click one label and move immediately to the next row
- allow `OTHER_UNCLEAR` only as a sparse review label, not as a default fallback

Recommended storage format:

- one row per `annotation_id` per annotator
- one output file or batch per annotator if that is easier operationally
- merge all raw labels before checking agreement

### Phase 3: Recycle disagreements

Any item with a disagreement on `primary_label` goes back into the queue.

For recycled items:

- assign the item to `2` different annotators where possible
- keep the earlier labels; do not overwrite them
- mark the new pass with a new `round_id`

This gives us a clearer signal about whether the disagreement came from coder noise, a weak codebook rule, or a genuinely ambiguous chunk.

### Phase 4: Build the training set

Include a row in the final training set only if it reaches a clear final label after review.

A practical rule is:

- keep rows with immediate agreement in round 1
- recycle rows with disagreement
- after recycling, keep rows only when a clear consensus emerges
- drop rows that remain unstable, split, or too ambiguous

In practice, that means a row can be dropped because:

- labels stay split after multiple rounds
- the text is too vague to code reliably
- the row exposes a chunking problem rather than a labeling problem

After consensus is established, split the retained labeled rows into train, validation, and test partitions. Do not fine-tune and evaluate on the exact same labeled rows.

## Reporting language

For the paper or notebook, we should be able to say:

- the pilot was fully overlapping across all `6` annotators
- the main annotation pool was double-coded
- disagreements were recycled into a second round
- only consensus-ready rows were retained for supervised training
- persistently disputed rows were removed from the final training set
