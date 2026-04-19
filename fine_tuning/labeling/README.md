# Labeling

This folder contains the entire annotation system in one place.

Contents:

- `annotation_protocol.md`
- `codebook.md`
- `labels_template.csv`
- `config/`
- `scripts/`
- `webapp/`
- `run_annotation_app.py`

Fast path:

```powershell
python fine_tuning/labeling/scripts/build_annotation_pool.py
python fine_tuning/labeling/scripts/init_annotation_db.py --reset
python fine_tuning/labeling/run_annotation_app.py
```

Backend visibility:

- app admin page: `http://127.0.0.1:5000/admin`
- local SQLite file: `data/fine_tuning/annotation.sqlite3`

To change categories, edit:

- `fine_tuning/labeling/config/label_options.json`

If you change labels before a real annotation round starts, just rebuild the database with `--reset`.

The labeling screen is intentionally minimal: annotators see only the text, click one label, and the next row loads immediately.

Current label set: `15` categories, including a sparse `OTHER_UNCLEAR` review label.

For small-team annotation over ngrok, the app now runs with Flask threaded mode and SQLite WAL mode, which is sufficient for a light workflow like six annotators clicking labels one at a time on a local laptop.
