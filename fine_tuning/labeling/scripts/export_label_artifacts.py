from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import ensure_parent, load_json, resolve_path, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export raw labels, conservative consensus, and recycle-needed rows.")
    parser.add_argument("--config", default="fine_tuning/labeling/config/annotation_workflow.json", help="Workflow config JSON.")
    parser.add_argument("--db", default=None, help="Override the SQLite database path.")
    parser.add_argument("--out-dir", default=None, help="Override the export directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite any existing export files.")
    return parser.parse_args()


def connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def fetch_raw_labels(connection: sqlite3.Connection) -> list[dict[str, object]]:
    query = """
        SELECT
            items.annotation_id,
            items.pool_phase,
            items.required_labels,
            items.sampling_stratum,
            items.selection_rank,
            items.pool_rank,
            items.company_name,
            items.ticker,
            items.filing_year,
            items.filing_date,
            items.comparison_window,
            items.merge_type,
            items.list_item_index,
            items.list_item_count,
            items.text_char_count,
            items.text AS item_text,
            annotators.email AS annotator_email,
            assignments.round_id,
            assignments.status AS assignment_status,
            labels.primary_label,
            labels.submitted_at
        FROM assignments
        JOIN labels ON labels.assignment_id = assignments.id
        JOIN items ON items.id = assignments.item_id
        JOIN annotators ON annotators.id = assignments.annotator_id
        WHERE assignments.status = 'submitted'
        ORDER BY items.pool_phase, items.company_name, items.filing_year, items.pool_rank, annotators.email
    """
    return [dict(row) for row in connection.execute(query).fetchall()]


RAW_COLUMNS = [
    "annotation_id",
    "pool_phase",
    "required_labels",
    "sampling_stratum",
    "selection_rank",
    "pool_rank",
    "company_name",
    "ticker",
    "filing_year",
    "filing_date",
    "comparison_window",
    "merge_type",
    "list_item_index",
    "list_item_count",
    "text_char_count",
    "item_text",
    "annotator_email",
    "round_id",
    "assignment_status",
    "primary_label",
    "submitted_at",
]


def summarize_item(row: sqlite3.Row, labels: list[sqlite3.Row], consensus_fraction: float) -> dict[str, object]:
    label_counts = Counter(label["primary_label"] for label in labels)
    total = sum(label_counts.values())
    top_label, top_count = label_counts.most_common(1)[0]
    second_count = label_counts.most_common(2)[1][1] if len(label_counts) > 1 else 0
    unanimous = len(label_counts) == 1
    supermajority = top_count / total >= consensus_fraction and top_count > second_count
    consensus = unanimous or supermajority

    labels_seen = " | ".join(f"{label} ({count})" for label, count in label_counts.most_common())
    return {
        "annotation_id": row["annotation_id"],
        "pool_phase": row["pool_phase"],
        "required_labels": row["required_labels"],
        "submitted_labels": total,
        "distinct_labels": len(label_counts),
        "top_label": top_label,
        "top_label_count": top_count,
        "top_label_share": round(top_count / total, 4),
        "consensus_label": top_label if consensus and total >= int(row["required_labels"]) else "",
        "consensus_type": "unanimous" if unanimous else ("supermajority" if consensus else ""),
        "labels_seen": labels_seen,
        "needs_recycle": not (consensus and total >= int(row["required_labels"])),
        "company_name": row["company_name"],
        "ticker": row["ticker"],
        "filing_year": row["filing_year"],
        "filing_date": row["filing_date"],
        "comparison_window": row["comparison_window"],
        "merge_type": row["merge_type"],
        "list_item_index": row["list_item_index"],
        "list_item_count": row["list_item_count"],
        "text_char_count": row["text_char_count"],
        "item_text": row["text"],
    }


def export_artifacts(connection: sqlite3.Connection, out_dir: Path, overwrite: bool, consensus_fraction: float) -> None:
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"Export directory already exists: {out_dir}. Re-run with --overwrite to rebuild exports.")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = fetch_raw_labels(connection)
    raw_path = out_dir / "raw_labels.csv"
    write_csv(raw_path, raw_rows, RAW_COLUMNS)

    item_query = """
        SELECT
            items.*,
            items.id AS item_id
        FROM items
        ORDER BY items.pool_phase, items.company_name, items.filing_year, items.pool_rank
    """
    item_rows = connection.execute(item_query).fetchall()

    labels_by_item: dict[int, list[sqlite3.Row]] = defaultdict(list)
    label_query = """
        SELECT assignments.item_id, labels.*
        FROM assignments
        JOIN labels ON labels.assignment_id = assignments.id
        WHERE assignments.status = 'submitted'
    """
    for row in connection.execute(label_query):
        labels_by_item[int(row["item_id"])].append(row)

    consensus_rows: list[dict[str, object]] = []
    recycle_rows: list[dict[str, object]] = []
    for item in item_rows:
        item_labels = labels_by_item.get(int(item["item_id"]), [])
        if not item_labels:
            recycle_rows.append(
                {
                    "annotation_id": item["annotation_id"],
                    "pool_phase": item["pool_phase"],
                    "required_labels": item["required_labels"],
                    "submitted_labels": 0,
                    "distinct_labels": 0,
                    "top_label": "",
                    "top_label_count": 0,
                    "top_label_share": 0,
                    "consensus_label": "",
                    "consensus_type": "",
                    "labels_seen": "",
                    "needs_recycle": True,
                    "recycle_reason": "no_submissions",
                    "company_name": item["company_name"],
                    "ticker": item["ticker"],
                    "filing_year": item["filing_year"],
                    "filing_date": item["filing_date"],
                    "comparison_window": item["comparison_window"],
                    "merge_type": item["merge_type"],
                    "list_item_index": item["list_item_index"],
                    "list_item_count": item["list_item_count"],
                    "text_char_count": item["text_char_count"],
                    "item_text": item["text"],
                }
            )
            continue

        summary = summarize_item(item, item_labels, consensus_fraction)
        if summary["needs_recycle"]:
            recycle_rows.append({**summary, "recycle_reason": "insufficient_labels_or_disagreement"})
        else:
            consensus_rows.append(summary)

    consensus_columns = [
        "annotation_id",
        "pool_phase",
        "required_labels",
        "submitted_labels",
        "distinct_labels",
        "top_label",
        "top_label_count",
        "top_label_share",
        "consensus_label",
        "consensus_type",
        "labels_seen",
        "company_name",
        "ticker",
        "filing_year",
        "filing_date",
        "comparison_window",
        "merge_type",
        "list_item_index",
        "list_item_count",
        "text_char_count",
        "item_text",
    ]
    recycle_columns = consensus_columns + ["recycle_reason", "needs_recycle"]

    write_csv(out_dir / "conservative_consensus.csv", consensus_rows, consensus_columns)
    write_csv(out_dir / "recycle_needed.csv", recycle_rows, recycle_columns)

    print(f"Exported raw labels: {raw_path}")
    print(f"Exported consensus rows: {out_dir / 'conservative_consensus.csv'}")
    print(f"Exported recycle queue: {out_dir / 'recycle_needed.csv'}")
    print(f"Raw label rows: {len(raw_rows)}")
    print(f"Consensus rows: {len(consensus_rows)}")
    print(f"Recycle-needed rows: {len(recycle_rows)}")


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    db_path = resolve_path(args.db or config["database_path"])
    out_dir = resolve_path(args.out_dir or config["exports_dir"])
    consensus_fraction = float(config["sample"]["consensus_threshold_fraction"])

    with connect(db_path) as connection:
        export_artifacts(connection, out_dir, args.overwrite, consensus_fraction)


if __name__ == "__main__":
    main()
