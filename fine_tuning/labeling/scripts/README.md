# Fine-Tuning Scripts

Current scripts:

- `build_annotation_pool.py`
  - samples a balanced annotation pool from the canonical corpus
- `init_annotation_db.py`
  - creates the local SQLite database and loads the annotation pool into it
- `export_label_artifacts.py`
  - exports raw labels, conservative consensus rows, and recycle-needed rows
- `common.py`
  - shared helpers for config, paths, and CSV I/O

Run these from the repository root so the relative paths resolve cleanly.
