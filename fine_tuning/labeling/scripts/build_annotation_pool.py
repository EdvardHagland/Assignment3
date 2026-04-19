from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import allocate_counts, group_rows, load_csv_rows, load_json, resolve_path, stable_hash, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced annotation pool from the canonical SEC corpus.")
    parser.add_argument("--config", default="fine_tuning/labeling/config/annotation_workflow.json", help="Workflow config JSON.")
    parser.add_argument("--output", default=None, help="Override annotation pool output CSV.")
    parser.add_argument("--seed", type=int, default=None, help="Override the deterministic sampling seed.")
    return parser.parse_args()


def select_rows(rows: list[dict[str, str]], target: int, seed: int) -> list[dict[str, str]]:
    if target <= 0 or not rows:
        return []

    by_merge_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_merge_type[row["merge_type"]].append(row)

    allocation = allocate_counts({merge_type: len(bucket) for merge_type, bucket in by_merge_type.items()}, target)
    selected: list[dict[str, str]] = []

    for merge_type in sorted(by_merge_type):
        bucket = sorted(
            by_merge_type[merge_type],
            key=lambda row: (
                stable_hash(row["annotation_id"], seed),
                row["annotation_index"],
                row["annotation_id"],
            ),
        )
        selected.extend(bucket[: allocation[merge_type]])

    return selected


def enrich_pool(rows: list[dict[str, str]], pool_phase: str, required_labels: int, seed: int) -> list[dict[str, str]]:
    enriched: list[dict[str, str]] = []
    for selection_rank, row in enumerate(rows, start=1):
        item = dict(row)
        item["pool_phase"] = pool_phase
        item["required_labels"] = str(required_labels)
        item["selection_seed"] = str(seed)
        item["selection_rank"] = str(selection_rank)
        item["sampling_stratum"] = "::".join(
            [row["comparison_window"], row["company_name"], row["filing_year"]]
        )
        enriched.append(item)
    return enriched


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    seed = args.seed if args.seed is not None else int(config["sample"]["seed"])
    source_path = resolve_path(config["source_dataset"])
    output_path = resolve_path(args.output or config["annotation_pool_output"])

    source_rows = load_csv_rows(source_path)
    filtered_rows = [
        row
        for row in source_rows
        if row["comparison_window"] in set(config["comparison_windows"])
        and int(row["text_char_count"]) >= int(config["sample"]["min_text_chars"])
    ]

    grouped = group_rows(filtered_rows, ["comparison_window", "company_name", "filing_year"])
    selected_rows: list[dict[str, str]] = []
    phase_counts = defaultdict(int)

    for bucket_key in sorted(grouped):
        bucket_rows = grouped[bucket_key]
        pilot_target = min(int(config["sample"]["pilot_rows_per_company_year"]), len(bucket_rows))
        pilot_rows = select_rows(bucket_rows, pilot_target, seed)
        pilot_ids = {row["annotation_id"] for row in pilot_rows}
        remaining_rows = [row for row in bucket_rows if row["annotation_id"] not in pilot_ids]

        main_target = min(int(config["sample"]["main_rows_per_company_year"]), len(remaining_rows))
        main_rows = select_rows(remaining_rows, main_target, seed + 1)

        selected_rows.extend(enrich_pool(pilot_rows, "pilot", int(config["sample"]["pilot_required_labels"]), seed))
        selected_rows.extend(enrich_pool(main_rows, "main", int(config["sample"]["main_required_labels"]), seed))
        phase_counts["pilot"] += len(pilot_rows)
        phase_counts["main"] += len(main_rows)

    selected_rows.sort(
        key=lambda row: (
            0 if row["pool_phase"] == "pilot" else 1,
            row["company_name"],
            row["filing_year"],
            row["filing_date"],
            row["merge_type"],
            int(row["annotation_index"]),
            row["annotation_id"],
        )
    )

    for index, row in enumerate(selected_rows, start=1):
        row["pool_rank"] = str(index)

    fieldnames = list(source_rows[0].keys()) + [
        "pool_phase",
        "required_labels",
        "selection_seed",
        "selection_rank",
        "pool_rank",
        "sampling_stratum",
    ]
    write_csv(output_path, selected_rows, fieldnames)

    print(f"Wrote {len(selected_rows)} annotation pool rows to {output_path}")
    print(f"Pilot rows: {phase_counts['pilot']}; main rows: {phase_counts['main']}")


if __name__ == "__main__":
    main()
