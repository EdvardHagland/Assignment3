from __future__ import annotations

import csv
import json
import random
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS annotators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS label_options (
    slug TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    description TEXT NOT NULL,
    sort_order INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    annotation_id TEXT NOT NULL UNIQUE,
    filing_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    company_name TEXT NOT NULL,
    cik TEXT NOT NULL,
    filing_date TEXT NOT NULL,
    filing_year INTEGER NOT NULL,
    period_bucket TEXT NOT NULL,
    comparison_window TEXT NOT NULL,
    form TEXT NOT NULL,
    accession_number TEXT NOT NULL,
    primary_document TEXT NOT NULL,
    source_url TEXT NOT NULL,
    risk_section_char_count INTEGER NOT NULL,
    annotation_index INTEGER NOT NULL,
    merge_type TEXT NOT NULL,
    start_paragraph_index INTEGER NOT NULL,
    end_paragraph_index INTEGER NOT NULL,
    start_paragraph_id TEXT NOT NULL,
    end_paragraph_id TEXT NOT NULL,
    source_paragraph_ids TEXT NOT NULL,
    source_paragraph_count INTEGER NOT NULL,
    list_context_text TEXT,
    list_item_index INTEGER,
    list_item_count INTEGER,
    text_char_count INTEGER NOT NULL,
    text TEXT NOT NULL,
    pool_phase TEXT NOT NULL,
    required_labels INTEGER NOT NULL,
    sampling_stratum TEXT NOT NULL,
    selection_seed INTEGER NOT NULL,
    selection_rank INTEGER NOT NULL,
    pool_rank INTEGER NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    annotator_id INTEGER NOT NULL REFERENCES annotators(id) ON DELETE CASCADE,
    round_id INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'assigned',
    assigned_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TEXT,
    submitted_at TEXT,
    UNIQUE(item_id, annotator_id, round_id)
);

CREATE TABLE IF NOT EXISTS labels (
    assignment_id INTEGER PRIMARY KEY REFERENCES assignments(id) ON DELETE CASCADE,
    primary_label TEXT NOT NULL REFERENCES label_options(slug),
    submitted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_items_phase ON items(pool_phase, comparison_window);
CREATE INDEX IF NOT EXISTS idx_items_company ON items(company_name, filing_year);
CREATE INDEX IF NOT EXISTS idx_items_stratum ON items(sampling_stratum);
CREATE INDEX IF NOT EXISTS idx_assignments_item ON assignments(item_id);
CREATE INDEX IF NOT EXISTS idx_assignments_annotator ON assignments(annotator_id, status);
"""


PHASE_ORDER = {"pilot": 0, "main": 1, "recycle": 2}


def connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA)
    connection.commit()


def load_label_options(connection: sqlite3.Connection, config_path: Path) -> None:
    label_options = json.loads(config_path.read_text(encoding="utf-8-sig"))
    connection.executemany(
        """
        INSERT INTO label_options (slug, display_name, description, sort_order)
        VALUES (:slug, :display_name, :description, :sort_order)
        ON CONFLICT(slug) DO UPDATE SET
            display_name = excluded.display_name,
            description = excluded.description,
            sort_order = excluded.sort_order
        """,
        label_options,
    )
    connection.commit()


def _load_corpus_rows(corpus_path: Path) -> list[dict[str, str]]:
    with corpus_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("comparison_window", "").startswith(("pre_", "post_"))]


def _window_bucket(value: str) -> str:
    if value.startswith("pre_"):
        return "pre"
    if value.startswith("post_"):
        return "post"
    return value


def _balanced_sample(rows: list[dict[str, str]], total: int, rng: random.Random) -> list[dict[str, str]]:
    if total <= 0 or not rows:
        return []

    by_window: dict[str, list[dict[str, str]]] = {"pre": [], "post": []}
    for row in rows:
        by_window[_window_bucket(row["comparison_window"])].append(row)

    target_pre = total // 2
    target_post = total - target_pre

    chosen: list[dict[str, str]] = []
    for window_name, target in (("pre", target_pre), ("post", target_post)):
        window_rows = by_window[window_name]
        if not window_rows:
            continue
        if len(window_rows) <= target:
            chosen.extend(window_rows)
        else:
            chosen.extend(rng.sample(window_rows, target))
    return chosen


def seed_annotation_pool(
    connection: sqlite3.Connection,
    corpus_path: Path,
    *,
    pilot_items: int = 60,
    main_items: int = 540,
    seed: int = 42,
) -> dict[str, int]:
    existing = connection.execute("SELECT COUNT(*) AS count FROM items").fetchone()
    if existing and int(existing["count"]) > 0:
        return {"pilot_items": 0, "main_items": 0, "total_items": int(existing["count"])}

    rows = _load_corpus_rows(corpus_path)
    rng = random.Random(seed)
    pilot_rows = _balanced_sample(rows, pilot_items, rng)
    pilot_ids = {row["annotation_id"] for row in pilot_rows}
    remaining_rows = [row for row in rows if row["annotation_id"] not in pilot_ids]
    main_rows = _balanced_sample(remaining_rows, main_items, random.Random(seed + 1))

    def _insert_rows(selected_rows: list[dict[str, str]], pool_phase: str, required_labels: int) -> int:
        inserted = 0
        for row in selected_rows:
            connection.execute(
                """
                INSERT INTO items (
                    annotation_id,
                    filing_id,
                    ticker,
                    company_name,
                    filing_date,
                    filing_year,
                    comparison_window,
                    pool_phase,
                    required_labels,
                    sampling_stratum,
                    text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["annotation_id"],
                    row["filing_id"],
                    row["ticker"],
                    row["company_name"],
                    row["filing_date"],
                    int(row["filing_year"]),
                    row["comparison_window"],
                    pool_phase,
                    required_labels,
                    row["comparison_window"],
                    row["text"],
                ),
            )
            inserted += 1
        return inserted

    inserted_pilot = _insert_rows(pilot_rows, "pilot", 6)
    inserted_main = _insert_rows(main_rows, "main", 2)
    connection.commit()
    return {"pilot_items": inserted_pilot, "main_items": inserted_main, "total_items": inserted_pilot + inserted_main}


def get_or_create_annotator(connection: sqlite3.Connection, email: str) -> sqlite3.Row:
    normalized = email.strip().lower()
    connection.execute("INSERT OR IGNORE INTO annotators (email) VALUES (?)", (normalized,))
    connection.commit()
    row = connection.execute("SELECT * FROM annotators WHERE email = ?", (normalized,)).fetchone()
    if row is None:
        raise RuntimeError(f"Unable to create annotator for {normalized}")
    return row


def fetch_label_options(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return connection.execute("SELECT * FROM label_options ORDER BY sort_order, slug").fetchall()


def fetch_progress(connection: sqlite3.Connection, annotator_id: int) -> dict[str, Any]:
    annotator_counts = connection.execute(
        "SELECT COUNT(*) AS submitted_count FROM assignments WHERE annotator_id = ? AND status = 'submitted'",
        (annotator_id,),
    ).fetchone()
    pool_counts = connection.execute(
        """
        SELECT items.pool_phase, COUNT(DISTINCT items.id) AS total_items,
               SUM(CASE WHEN submitted.labels_submitted >= items.required_labels THEN 1 ELSE 0 END) AS completed_items
        FROM items
        LEFT JOIN (
            SELECT item_id, COUNT(*) AS labels_submitted
            FROM assignments
            WHERE status = 'submitted'
            GROUP BY item_id
        ) AS submitted ON submitted.item_id = items.id
        WHERE items.is_active = 1
        GROUP BY items.pool_phase
        ORDER BY CASE items.pool_phase WHEN 'pilot' THEN 0 WHEN 'main' THEN 1 WHEN 'recycle' THEN 2 ELSE 3 END
        """
    ).fetchall()
    return {
        "submitted_count": int(annotator_counts["submitted_count"] if annotator_counts else 0),
        "pool_counts": pool_counts,
    }


def fetch_existing_assignment(connection: sqlite3.Connection, annotator_id: int) -> Optional[sqlite3.Row]:
    return connection.execute(
        """
        SELECT assignments.id AS assignment_id, assignments.round_id, items.*
        FROM assignments
        JOIN items ON items.id = assignments.item_id
        WHERE assignments.annotator_id = ?
          AND assignments.status = 'assigned'
          AND items.is_active = 1
        ORDER BY CASE items.pool_phase WHEN 'pilot' THEN 0 WHEN 'main' THEN 1 WHEN 'recycle' THEN 2 ELSE 3 END,
                 assignments.assigned_at,
                 items.id
        LIMIT 1
        """,
        (annotator_id,),
    ).fetchone()


def assign_next_item(connection: sqlite3.Connection, annotator_id: int) -> Optional[sqlite3.Row]:
    existing = fetch_existing_assignment(connection, annotator_id)
    if existing is not None:
        return existing

    candidate = connection.execute(
        """
        WITH submitted_counts AS (
            SELECT item_id, COUNT(*) AS labels_submitted
            FROM assignments
            WHERE status = 'submitted'
            GROUP BY item_id
        )
        SELECT items.*
        FROM items
        LEFT JOIN submitted_counts ON submitted_counts.item_id = items.id
        WHERE items.is_active = 1
          AND COALESCE(submitted_counts.labels_submitted, 0) < items.required_labels
          AND NOT EXISTS (
                SELECT 1
                FROM assignments prior
                WHERE prior.item_id = items.id AND prior.annotator_id = ?
          )
        ORDER BY CASE items.pool_phase WHEN 'pilot' THEN 0 WHEN 'main' THEN 1 WHEN 'recycle' THEN 2 ELSE 3 END,
                 COALESCE(submitted_counts.labels_submitted, 0),
                 items.company_name,
                 items.filing_date,
                 items.id
        LIMIT 1
        """,
        (annotator_id,),
    ).fetchone()
    if candidate is None:
        return None

    cursor = connection.execute(
        "INSERT INTO assignments (item_id, annotator_id, round_id, status, started_at) VALUES (?, ?, 1, 'assigned', CURRENT_TIMESTAMP)",
        (candidate["id"], annotator_id),
    )
    connection.commit()
    return connection.execute(
        "SELECT assignments.id AS assignment_id, assignments.round_id, items.* FROM assignments JOIN items ON items.id = assignments.item_id WHERE assignments.id = ?",
        (cursor.lastrowid,),
    ).fetchone()


def submit_label(connection: sqlite3.Connection, assignment_id: int, annotator_id: int, primary_label: str) -> None:
    assignment = connection.execute(
        "SELECT * FROM assignments WHERE id = ? AND annotator_id = ?",
        (assignment_id, annotator_id),
    ).fetchone()
    if assignment is None:
        raise ValueError("Assignment not found for annotator")
    if assignment["status"] == "submitted":
        raise ValueError("Assignment already submitted")

    connection.execute(
        """
        INSERT INTO labels (assignment_id, primary_label)
        VALUES (?, ?)
        ON CONFLICT(assignment_id) DO UPDATE SET
            primary_label = excluded.primary_label,
            submitted_at = CURRENT_TIMESTAMP
        """,
        (assignment_id, primary_label),
    )
    connection.execute("UPDATE assignments SET status = 'submitted', submitted_at = CURRENT_TIMESTAMP WHERE id = ?", (assignment_id,))
    connection.commit()


def fetch_admin_summary(connection: sqlite3.Connection) -> dict[str, Iterable[sqlite3.Row]]:
    item_progress = connection.execute(
        """
        SELECT items.pool_phase, items.comparison_window, COUNT(*) AS total_items,
               SUM(CASE WHEN progress.labels_submitted >= items.required_labels THEN 1 ELSE 0 END) AS fully_labeled,
               AVG(COALESCE(progress.labels_submitted, 0)) AS average_labels
        FROM items
        LEFT JOIN (
            SELECT item_id, COUNT(*) AS labels_submitted
            FROM assignments
            WHERE status = 'submitted'
            GROUP BY item_id
        ) AS progress ON progress.item_id = items.id
        GROUP BY items.pool_phase, items.comparison_window
        ORDER BY items.pool_phase, items.comparison_window
        """
    ).fetchall()

    disagreement_rows = connection.execute(
        """
        SELECT items.id AS item_id, items.annotation_id, items.pool_phase, items.company_name, items.filing_year,
               items.required_labels,
               GROUP_CONCAT(labels.primary_label, ' | ') AS labels_seen,
               COUNT(DISTINCT labels.primary_label) AS distinct_labels,
               COUNT(labels.assignment_id) AS label_count
        FROM items
        JOIN assignments ON assignments.item_id = items.id AND assignments.status = 'submitted'
        JOIN labels ON labels.assignment_id = assignments.id
        GROUP BY items.id
        HAVING COUNT(labels.assignment_id) >= 2 AND COUNT(DISTINCT labels.primary_label) > 1
        ORDER BY items.pool_phase, items.company_name, items.annotation_id
        LIMIT 50
        """
    ).fetchall()

    annotator_rows = connection.execute(
        """
        SELECT annotators.email,
               COUNT(CASE WHEN assignments.status = 'submitted' THEN 1 END) AS submitted,
               COUNT(CASE WHEN assignments.status = 'assigned' THEN 1 END) AS in_progress
        FROM annotators
        LEFT JOIN assignments ON assignments.annotator_id = annotators.id
        GROUP BY annotators.id
        ORDER BY annotators.email
        """
    ).fetchall()
    total_items = connection.execute("SELECT COUNT(*) AS total_items FROM items WHERE is_active = 1").fetchone()
    total_submitted = connection.execute(
        "SELECT COUNT(*) AS submitted_items FROM assignments WHERE status = 'submitted'"
    ).fetchone()
    return {
        "item_progress": item_progress,
        "disagreements": disagreement_rows,
        "annotators": annotator_rows,
        "totals": {
            "total_items": int(total_items["total_items"] if total_items else 0),
            "submitted_items": int(total_submitted["submitted_items"] if total_submitted else 0),
        },
    }


def recycle_item(connection: sqlite3.Connection, item_id: int) -> None:
    item = connection.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    if item is None:
        raise ValueError(f"Item {item_id} not found")

    submitted_count = connection.execute(
        "SELECT COUNT(*) AS submitted_count FROM assignments WHERE item_id = ? AND status = 'submitted'",
        (item_id,),
    ).fetchone()
    submitted_total = int(submitted_count["submitted_count"] if submitted_count else 0)
    next_required = max(int(item["required_labels"]), submitted_total + 1)
    connection.execute(
        "UPDATE items SET pool_phase = 'recycle', required_labels = ? WHERE id = ?",
        (next_required, item_id),
    )
    connection.commit()
