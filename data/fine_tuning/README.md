# Fine-Tuning Data

This directory is for annotation and training-data artifacts derived from the canonical corpus.

Planned files:

- `annotation_pool.csv`: sampled rows sent out for labeling
- `annotation.sqlite3`: local database used by the annotation webapp
- `exports/raw_labels.csv`: one row per annotator per item
- `exports/conservative_consensus.csv`: rows that already have enough agreement to be used downstream
- `exports/recycle_needed.csv`: rows that need another round or adjudication

The source corpus for all of these files is `data/final/sec_defense_risk_dataset.csv`.
