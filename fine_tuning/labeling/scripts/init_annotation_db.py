from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import ensure_parent, load_csv_rows, load_json, resolve_path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize the local annotation SQLite database from the pool CSV.")
    parser.add_argument("--config", default="fine_tuning/labeling/config/annotation_workflow.json", help="Workflow config JSON.")
    parser.add_argument("--db", default=None, help="Override the SQLite database path.")
    parser.add_argument("--pool", default=None, help="Override the annotation pool CSV path.")
    parser.add_argument("--reset", action="store_true", help="Delete any existing database file before creating it.")
    parser.add_argument("--annotator", action="append", default=[], help="Optional annotator email to pre-register. Repeatable.")
    return parser.parse_args()


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


def load_label_options(connection: sqlite3.Connection, config_path: str | Path) -> None:
    label_options = load_json(config_path)
    connection.execute("DELETE FROM label_options")
    connection.executemany(
        """
        INSERT INTO label_options (slug, display_name, description, sort_order)
        VALUES (:slug, :display_name, :description, :sort_order)
        """,
        label_options,
    )
    connection.commit()


def reset_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        DELETE FROM labels;
        DELETE FROM assignments;
        DELETE FROM annotators;
        DELETE FROM items;
        """
    )
    connection.commit()


def preload_items(connection: sqlite3.Connection, pool_rows: list[dict[str, str]]) -> None:
    connection.executemany(
        """
        INSERT INTO items (
            annotation_id, filing_id, ticker, company_name, cik, filing_date, filing_year,
            period_bucket, comparison_window, form, accession_number, primary_document, source_url,
            risk_section_char_count, annotation_index, merge_type, start_paragraph_index,
            end_paragraph_index, start_paragraph_id, end_paragraph_id, source_paragraph_ids,
            source_paragraph_count, list_context_text, list_item_index, list_item_count, text_char_count,
            text, pool_phase, required_labels, sampling_stratum, selection_seed, selection_rank, pool_rank
        ) VALUES (
            :annotation_id, :filing_id, :ticker, :company_name, :cik, :filing_date, :filing_year,
            :period_bucket, :comparison_window, :form, :accession_number, :primary_document, :source_url,
            :risk_section_char_count, :annotation_index, :merge_type, :start_paragraph_index,
            :end_paragraph_index, :start_paragraph_id, :end_paragraph_id, :source_paragraph_ids,
            :source_paragraph_count, :list_context_text, :list_item_index, :list_item_count, :text_char_count,
            :text, :pool_phase, :required_labels, :sampling_stratum, :selection_seed, :selection_rank, :pool_rank
        )
        """,
        pool_rows,
    )
    connection.commit()


def seed_annotators(connection: sqlite3.Connection, annotator_emails: list[str]) -> None:
    for email in annotator_emails:
        normalized = email.strip().lower()
        if not normalized:
            continue
        connection.execute("INSERT OR IGNORE INTO annotators (email) VALUES (?)", (normalized,))
    connection.commit()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    db_path = resolve_path(args.db or config["database_path"])
    pool_path = resolve_path(args.pool or config["annotation_pool_output"])

    if db_path.exists() and not args.reset:
        raise FileExistsError(f"Database already exists: {db_path}. Re-run with --reset to rebuild it.")

    ensure_parent(db_path)
    if db_path.exists() and args.reset:
        db_path.unlink()

    pool_rows = load_csv_rows(pool_path)

    with connect(db_path) as connection:
        initialize_database(connection)
        load_label_options(connection, config["label_options"])
        preload_items(connection, pool_rows)
        seed_annotators(connection, args.annotator)

    print(f"Initialized {db_path} with {len(pool_rows)} annotation items")
    print(f"Loaded {len(load_json(config['label_options']))} label options")


if __name__ == "__main__":
    main()
