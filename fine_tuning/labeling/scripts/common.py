from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root() / candidate


def load_json(path: str | Path) -> Any:
    return json.loads(resolve_path(path).read_text(encoding="utf-8-sig"))


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with resolve_path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def ensure_parent(path: str | Path) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> Path:
    resolved = ensure_parent(path)
    with resolved.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return resolved


def stable_hash(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def allocate_counts(group_sizes: dict[str, int], target: int) -> dict[str, int]:
    keys = list(group_sizes)
    if target <= 0 or not keys:
        return {key: 0 for key in keys}

    total_available = sum(group_sizes.values())
    if target >= total_available:
        return dict(group_sizes)

    allocation = {key: 0 for key in keys}

    if target < len(keys):
        for key in sorted(keys, key=lambda item: (-group_sizes[item], item))[:target]:
            allocation[key] = 1
        return allocation

    for key in keys:
        allocation[key] = 1

    remaining = target - len(keys)
    if remaining <= 0:
        return allocation

    capacities = {key: max(group_sizes[key] - allocation[key], 0) for key in keys}
    capacity_total = sum(capacities.values())
    if capacity_total <= 0:
        return allocation

    ideal = {key: remaining * capacities[key] / capacity_total for key in keys}
    floor_alloc = {key: min(capacities[key], int(ideal[key])) for key in keys}
    for key, count in floor_alloc.items():
        allocation[key] += count

    leftover = remaining - sum(floor_alloc.values())
    if leftover <= 0:
        return allocation

    order = sorted(
        keys,
        key=lambda item: (ideal[item] - floor_alloc[item], capacities[item], item),
        reverse=True,
    )
    for key in order:
        if leftover <= 0:
            break
        if allocation[key] < group_sizes[key]:
            allocation[key] += 1
            leftover -= 1

    return allocation


def group_rows(rows: list[dict[str, str]], keys: list[str]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        bucket_key = tuple(row[key] for key in keys)
        grouped.setdefault(bucket_key, []).append(row)
    return grouped
