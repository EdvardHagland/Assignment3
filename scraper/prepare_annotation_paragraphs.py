#!/usr/bin/env python3
"""
Clean extracted SEC risk paragraphs into annotation-ready rows.

This script keeps the raw paragraph extract untouched and writes a second CSV
with three lightweight transformations:
1. drop generic lead-in boilerplate that is not a specific risk;
2. drop SEC "risk factor summary" blocks near the top of a filing; and
3. merge short heading-like risk statements with the following explanatory
   paragraph so annotators see one self-contained text unit.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SUMMARY_TRIGGER_PATTERNS = (
    "risk factor summary",
    "summary of risk factors",
    "summary of our risks",
    "summary of our risk factors",
    "following is a summary of the risks",
    "this risk factor summary",
)

BULLET_ITEM_PATTERN = re.compile(r"(?:^|[;])\s*(?:[•●]|u)\s+")

SECTION_LABEL_PREFIXES = (
    "industry and economic risks",
    "legal and regulatory risks",
    "risks related to our business",
    "risks related to our industry",
    "risks related to our u.s. government contracts",
    "risks related to our contracts",
    "risks related to our indebtedness",
    "risks related to our common stock",
    "risks related to our operations",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create annotation-ready SEC risk paragraphs."
    )
    parser.add_argument(
        "--input",
        default="data/intermediate/sec_10k_risk_paragraphs.csv",
        help="Path to the raw paragraph CSV from sec_fetch_risk_factors.py",
    )
    parser.add_argument(
        "--sections",
        default="data/intermediate/sec_10k_risk_sections.csv",
        help="Path to the extracted section CSV used to enrich row metadata.",
    )
    parser.add_argument(
        "--output",
        default="data/final/sec_defense_risk_dataset.csv",
        help="Output CSV path for the final cleaned dataset.",
    )
    parser.add_argument(
        "--report",
        default="data/intermediate/sec_10k_risk_cleaning_report.csv",
        help="Output CSV path for the cleaning report.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> List[dict]:
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            break
        except OverflowError:
            max_size = int(max_size / 10)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_section_lookup(path: Path) -> Dict[str, dict]:
    lookup: Dict[str, dict] = {}
    for row in read_rows(path):
        lookup[row["filing_id"]] = {
            "ticker": row.get("ticker", ""),
            "resolved_ticker": row.get("resolved_ticker", row.get("ticker", "")),
            "company_name": row.get("company_name", ""),
            "company_layer": row.get("company_layer", ""),
            "company_notes": row.get("company_notes", row.get("notes", "")),
            "cik": row.get("cik", ""),
            "form": row.get("form", ""),
            "filing_date": row.get("filing_date", ""),
            "filing_year": row.get("filing_year", ""),
            "accession_number": row.get("accession_number", ""),
            "primary_document": row.get("primary_document", ""),
            "source_url": row.get("source_url", ""),
            "risk_section_char_count": row.get("risk_section_char_count", ""),
        }
    return lookup


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def enrich_rows(rows: List[dict], section_lookup: Dict[str, dict]) -> List[dict]:
    enriched: List[dict] = []
    for row in rows:
        combined = dict(section_lookup.get(row["filing_id"], {}))
        combined.update(row)
        filing_year = str(combined.get("filing_year", "")).strip()
        if not filing_year:
            filing_date = combined.get("filing_date", "")
            filing_year = filing_date[:4] if filing_date else ""
            combined["filing_year"] = filing_year

        year_value = int(filing_year) if filing_year.isdigit() else None
        if year_value is None or year_value < 2018 or year_value > 2025:
            continue
        combined["period_bucket"] = "pre_2022" if year_value and year_value < 2022 else "post_2022"
        if 2018 <= year_value <= 2021:
            combined["comparison_window"] = "pre_2018_2021"
        elif 2022 <= year_value <= 2025:
            combined["comparison_window"] = "post_2022_2025"
        enriched.append(combined)
    return enriched


def normalize_text(text: str) -> str:
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u2060]", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def normalized_lower(text: str) -> str:
    return normalize_text(text).lower()


def sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?](?=\s|$)", text))


def bullet_like(text: str) -> bool:
    return bool(re.search(r"(?:^|[\s;])[•●u]\s", text))


def trim_terminal_conjunction(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[;,\s]+$", "", text)
    text = re.sub(r"(?:[;,\s]+)(and|or)$", "", text, flags=re.IGNORECASE)
    return text.strip(" ;,")


def parse_bullet_segments(text: str) -> Tuple[str, List[str]]:
    clean = normalize_text(text)
    matches = list(BULLET_ITEM_PATTERN.finditer(clean))
    if not matches:
        return "", []

    lead = clean[: matches[0].start()].strip(" ;")
    items: List[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(clean)
        item = trim_terminal_conjunction(clean[start:end])
        if item:
            items.append(item)
    return lead, items


def normalize_list_context(text: str) -> str:
    context = trim_terminal_conjunction(text)
    context = re.sub(
        r"(?:\b(?:including|include|such as)\s+the\s+following|"
        r"\b(?:including|include|such as)\s+the\s+following\s*:"
        r"|\bdo\s+the\s+following|\bdo\s+the\s+following\s*:)\s*$",
        "",
        context,
        flags=re.IGNORECASE,
    )
    context = context.rstrip(" :;,-")
    if normalized_lower(context) in SECTION_LABEL_PREFIXES:
        return ""
    return context


def is_bullet_only_row(text: str) -> bool:
    lead, items = parse_bullet_segments(text)
    return bool(items) and not normalize_list_context(lead)


def looks_like_list_context(text: str) -> bool:
    clean = normalize_text(text)
    lower = clean.lower()
    if has_summary_trigger(clean):
        return False
    return lower.endswith(":") or "including the following" in lower or "include the following" in lower


def with_list_defaults(row: dict) -> dict:
    updated = dict(row)
    updated.setdefault("list_context_text", "")
    updated.setdefault("list_item_index", "")
    updated.setdefault("list_item_count", "")
    return updated


def combine_context_and_item(context: str, item: str) -> str:
    item = trim_terminal_conjunction(item)
    context = normalize_list_context(context)
    if not context:
        return item
    joiner = ":" if not context.endswith((".", "!", "?")) else ""
    return f"{context}{joiner} {item}"


def combine_source_metadata(rows: List[dict]) -> dict:
    paragraph_ids: List[str] = []
    for row in rows:
        paragraph_ids.extend(part for part in row["source_paragraph_ids"].split("|") if part)

    unique_ids: List[str] = []
    seen = set()
    for paragraph_id in paragraph_ids:
        if paragraph_id in seen:
            continue
        seen.add(paragraph_id)
        unique_ids.append(paragraph_id)

    start_index = min(int(row["start_paragraph_index"]) for row in rows)
    end_index = max(int(row["end_paragraph_index"]) for row in rows)

    start_row = min(rows, key=lambda row: int(row["start_paragraph_index"]))
    end_row = max(rows, key=lambda row: int(row["end_paragraph_index"]))

    return {
        "start_paragraph_index": start_index,
        "end_paragraph_index": end_index,
        "start_paragraph_id": start_row["start_paragraph_id"],
        "end_paragraph_id": end_row["end_paragraph_id"],
        "source_paragraph_ids": "|".join(unique_ids),
        "source_paragraph_count": len(unique_ids),
    }


def build_bullet_split_row(
    base_row: dict,
    context: str,
    item_text: str,
    source_rows: List[dict],
    item_index: int,
    item_count: int,
) -> dict:
    combined_text = combine_context_and_item(context, item_text)
    metadata = combine_source_metadata(source_rows)
    updated = dict(base_row)
    updated.update(metadata)
    updated["annotation_id"] = ""
    updated["annotation_index"] = 0
    updated["merge_type"] = "bullet_split"
    updated["list_context_text"] = normalize_list_context(context)
    updated["list_item_index"] = item_index
    updated["list_item_count"] = item_count
    updated["text"] = combined_text
    updated["text_char_count"] = len(combined_text)
    return updated


def renumber_annotation_rows(rows: List[dict]) -> List[dict]:
    renumbered: List[dict] = []
    for index, row in enumerate(rows, start=1):
        updated = with_list_defaults(row)
        updated["annotation_index"] = index
        updated["annotation_id"] = f"{updated['filing_id']}_a_{index:03d}"
        renumbered.append(updated)
    return renumbered


def split_bullet_lists(rows: List[dict], stats: Counter) -> List[dict]:
    output: List[dict] = []
    index = 0

    while index < len(rows):
        row = rows[index]
        lead, inline_items = parse_bullet_segments(row["text"])
        list_context = normalize_list_context(lead)
        collected: List[Tuple[str, List[dict]]] = []
        next_index = index + 1

        if inline_items and list_context:
            for item in inline_items:
                collected.append((item, [row]))
        elif looks_like_list_context(row["text"]) and index + 1 < len(rows) and is_bullet_only_row(rows[index + 1]["text"]):
            list_context = normalize_list_context(row["text"])
            next_index = index + 1
        else:
            output.append(with_list_defaults(row))
            index += 1
            continue

        scan_index = next_index
        while scan_index < len(rows):
            trailing_row = rows[scan_index]
            trailing_lead, trailing_items = parse_bullet_segments(trailing_row["text"])
            if not trailing_items or normalize_list_context(trailing_lead):
                break
            for item in trailing_items:
                collected.append((item, [row, trailing_row]))
            scan_index += 1

        if not collected:
            output.append(with_list_defaults(row))
            index += 1
            continue

        item_count = len(collected)
        for item_index, (item_text, source_rows) in enumerate(collected, start=1):
            base_row = source_rows[-1]
            output.append(
                build_bullet_split_row(
                    base_row=base_row,
                    context=list_context,
                    item_text=item_text,
                    source_rows=source_rows,
                    item_index=item_index,
                    item_count=item_count,
                )
            )

        stats["split_bullet_groups"] += 1
        stats["split_bullet_items"] += item_count
        index = scan_index

    return output


def has_summary_trigger(text: str) -> bool:
    lower = normalized_lower(text)
    return any(pattern in lower for pattern in SUMMARY_TRIGGER_PATTERNS)


def is_intro_boilerplate(text: str) -> bool:
    lower = normalized_lower(text)
    strong_patterns = (
        "an investment in our common stock or debt securities involves risks and uncertainties",
        "you should consider and read carefully all of the risks and uncertainties described below",
        "in your evaluation of our company and business",
        "a description of the risks and uncertainties associated with our business is set forth below",
    )
    if any(pattern in lower for pattern in strong_patterns):
        return True

    intro_patterns = (
        "you should carefully consider",
        "the risks described below are not the only",
        "the occurrence of any of the following risks",
        "this annual report also contains forward-looking statements",
        "before investing in our common stock",
        "the market price of our stock could decline",
    )
    hits = sum(1 for pattern in intro_patterns if pattern in lower)
    return hits >= 2


def is_summary_continuation(text: str) -> bool:
    clean = normalize_text(text)
    if has_summary_trigger(clean):
        return True
    if bullet_like(clean):
        return True
    return False


def is_heading_like(text: str) -> bool:
    clean = normalize_text(text)
    lower = clean.lower()
    if not clean or has_summary_trigger(clean):
        return False
    if bullet_like(clean):
        return False
    if len(clean) > 280:
        return False
    if sentence_count(clean) > 2:
        return False
    if clean.endswith(":"):
        return True
    if any(lower.startswith(prefix + " ") for prefix in SECTION_LABEL_PREFIXES):
        return True
    return sentence_count(clean) in {1, 2}


def should_merge_with_next(current: dict, following: dict) -> bool:
    current_text = normalize_text(current["text"])
    following_text = normalize_text(following["text"])
    if not is_heading_like(current_text):
        return False
    if bullet_like(following_text) or has_summary_trigger(following_text):
        return False
    if len(following_text) < 250:
        return False
    if sentence_count(following_text) < 2:
        return False
    return True


def should_merge_continuation(current: dict, following: dict) -> bool:
    current_text = normalize_text(current["text"])
    following_text = normalize_text(following["text"])
    if not current_text or not following_text:
        return False
    if has_summary_trigger(current_text) or has_summary_trigger(following_text):
        return False
    if bullet_like(current_text) or bullet_like(following_text):
        return False
    if re.search(r"[.!?:]$", current_text):
        return False
    return bool(following_text[:1].islower() or len(current_text) <= 220)


def group_by_filing(rows: Iterable[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["filing_id"]].append(row)
    for filing_rows in grouped.values():
        filing_rows.sort(key=lambda row: int(row["paragraph_index"]))
    return grouped


def drop_summary_block(rows: List[dict]) -> Tuple[List[dict], int]:
    kept: List[dict] = []
    dropped = 0
    index = 0

    while index < len(rows):
        row = rows[index]
        para_index = int(row["paragraph_index"])
        text = normalize_text(row["text"])
        if para_index <= 10 and has_summary_trigger(text):
            dropped += 1
            index += 1
            while index < len(rows) and int(rows[index]["paragraph_index"]) <= 40:
                if not is_summary_continuation(rows[index]["text"]):
                    break
                dropped += 1
                index += 1
            continue
        kept.append(row)
        index += 1

    return kept, dropped


def clean_filing_rows(rows: List[dict]) -> Tuple[List[dict], Counter]:
    stats = Counter()
    filtered: List[dict] = []

    for row in rows:
        text = normalize_text(row["text"])
        para_index = int(row["paragraph_index"])
        if para_index <= 2 and is_intro_boilerplate(text):
            stats["dropped_intro"] += 1
            continue
        row = dict(row)
        row["text"] = text
        row["text_char_count"] = str(len(text))
        filtered.append(row)

    filtered, summary_drops = drop_summary_block(filtered)
    stats["dropped_summary"] += summary_drops

    merged_rows: List[dict] = []
    index = 0
    annotation_index = 1

    while index < len(filtered):
        current = filtered[index]
        if index + 1 < len(filtered) and should_merge_continuation(current, filtered[index + 1]):
            following = filtered[index + 1]
            merged_text = f"{current['text']} {following['text']}"
            merged_rows.append(
                {
                    "annotation_id": f"{current['filing_id']}_a_{annotation_index:03d}",
                    "filing_id": current["filing_id"],
                    "ticker": current["ticker"],
                    "resolved_ticker": current.get("resolved_ticker", current["ticker"]),
                    "company_name": current["company_name"],
                    "company_layer": current.get("company_layer", ""),
                    "company_notes": current.get("company_notes", ""),
                    "cik": current.get("cik", ""),
                    "filing_date": current["filing_date"],
                    "filing_year": current.get("filing_year", ""),
                    "period_bucket": current.get("period_bucket", ""),
                    "comparison_window": current.get("comparison_window", ""),
                    "form": current["form"],
                    "accession_number": current.get("accession_number", ""),
                    "primary_document": current.get("primary_document", ""),
                    "source_url": current["source_url"],
                    "risk_section_char_count": current.get("risk_section_char_count", ""),
                    "annotation_index": annotation_index,
                    "merge_type": "broken_pair",
                    "start_paragraph_index": current["paragraph_index"],
                    "end_paragraph_index": following["paragraph_index"],
                    "start_paragraph_id": current["paragraph_id"],
                    "end_paragraph_id": following["paragraph_id"],
                    "source_paragraph_ids": f"{current['paragraph_id']}|{following['paragraph_id']}",
                    "source_paragraph_count": 2,
                    "text_char_count": len(merged_text),
                    "text": merged_text,
                }
            )
            stats["merged_broken_pairs"] += 1
            annotation_index += 1
            index += 2
            continue

        if index + 1 < len(filtered) and should_merge_with_next(current, filtered[index + 1]):
            following = filtered[index + 1]
            merged_text = f"{current['text']} {following['text']}"
            merged_rows.append(
                {
                    "annotation_id": f"{current['filing_id']}_a_{annotation_index:03d}",
                    "filing_id": current["filing_id"],
                    "ticker": current["ticker"],
                    "resolved_ticker": current.get("resolved_ticker", current["ticker"]),
                    "company_name": current["company_name"],
                    "company_layer": current.get("company_layer", ""),
                    "company_notes": current.get("company_notes", ""),
                    "cik": current.get("cik", ""),
                    "filing_date": current["filing_date"],
                    "filing_year": current.get("filing_year", ""),
                    "period_bucket": current.get("period_bucket", ""),
                    "comparison_window": current.get("comparison_window", ""),
                    "form": current["form"],
                    "accession_number": current.get("accession_number", ""),
                    "primary_document": current.get("primary_document", ""),
                    "source_url": current["source_url"],
                    "risk_section_char_count": current.get("risk_section_char_count", ""),
                    "annotation_index": annotation_index,
                    "merge_type": "heading_pair",
                    "start_paragraph_index": current["paragraph_index"],
                    "end_paragraph_index": following["paragraph_index"],
                    "start_paragraph_id": current["paragraph_id"],
                    "end_paragraph_id": following["paragraph_id"],
                    "source_paragraph_ids": f"{current['paragraph_id']}|{following['paragraph_id']}",
                    "source_paragraph_count": 2,
                    "text_char_count": len(merged_text),
                    "text": merged_text,
                }
            )
            stats["merged_heading_pairs"] += 1
            annotation_index += 1
            index += 2
            continue

        merged_rows.append(
            {
                "annotation_id": f"{current['filing_id']}_a_{annotation_index:03d}",
                "filing_id": current["filing_id"],
                "ticker": current["ticker"],
                "resolved_ticker": current.get("resolved_ticker", current["ticker"]),
                "company_name": current["company_name"],
                "company_layer": current.get("company_layer", ""),
                "company_notes": current.get("company_notes", ""),
                "cik": current.get("cik", ""),
                "filing_date": current["filing_date"],
                "filing_year": current.get("filing_year", ""),
                "period_bucket": current.get("period_bucket", ""),
                "comparison_window": current.get("comparison_window", ""),
                "form": current["form"],
                "accession_number": current.get("accession_number", ""),
                "primary_document": current.get("primary_document", ""),
                "source_url": current["source_url"],
                "risk_section_char_count": current.get("risk_section_char_count", ""),
                "annotation_index": annotation_index,
                "merge_type": "single",
                "start_paragraph_index": current["paragraph_index"],
                "end_paragraph_index": current["paragraph_index"],
                "start_paragraph_id": current["paragraph_id"],
                "end_paragraph_id": current["paragraph_id"],
                "source_paragraph_ids": current["paragraph_id"],
                "source_paragraph_count": 1,
                "text_char_count": len(current["text"]),
                "text": current["text"],
            }
        )
        annotation_index += 1
        index += 1

    merged_rows = split_bullet_lists(merged_rows, stats)
    merged_rows = renumber_annotation_rows(merged_rows)

    return merged_rows, stats


def summarize_report(cleaned_rows: List[dict], filing_stats: Dict[str, Counter]) -> List[dict]:
    by_company = Counter(row["company_name"] for row in cleaned_rows)
    by_company_layer = Counter(row.get("company_layer", "") for row in cleaned_rows)
    by_year = Counter(row["filing_date"][:4] for row in cleaned_rows)
    by_comparison_window = Counter(row["comparison_window"] for row in cleaned_rows)
    total_stats = Counter()
    for stats in filing_stats.values():
        total_stats.update(stats)

    report_rows: List[dict] = [
        {"metric": "cleaned_rows", "group": "all", "value": len(cleaned_rows)},
        {"metric": "dropped_intro", "group": "all", "value": total_stats["dropped_intro"]},
        {"metric": "dropped_summary", "group": "all", "value": total_stats["dropped_summary"]},
        {"metric": "merged_broken_pairs", "group": "all", "value": total_stats["merged_broken_pairs"]},
        {"metric": "merged_heading_pairs", "group": "all", "value": total_stats["merged_heading_pairs"]},
        {"metric": "split_bullet_groups", "group": "all", "value": total_stats["split_bullet_groups"]},
        {"metric": "split_bullet_items", "group": "all", "value": total_stats["split_bullet_items"]},
    ]

    for company_name, count in sorted(by_company.items()):
        report_rows.append(
            {"metric": "cleaned_rows_by_company", "group": company_name, "value": count}
        )
    for company_layer, count in sorted(by_company_layer.items()):
        report_rows.append(
            {"metric": "cleaned_rows_by_company_layer", "group": company_layer, "value": count}
        )
    for year, count in sorted(by_year.items()):
        report_rows.append(
            {"metric": "cleaned_rows_by_year", "group": year, "value": count}
        )
    for window, count in sorted(by_comparison_window.items()):
        report_rows.append(
            {"metric": "cleaned_rows_by_comparison_window", "group": window, "value": count}
        )

    return report_rows


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    sections_path = Path(args.sections)
    output_path = Path(args.output)
    report_path = Path(args.report)

    raw_rows = read_rows(input_path)
    section_lookup = build_section_lookup(sections_path)
    raw_rows = enrich_rows(raw_rows, section_lookup)
    filing_groups = group_by_filing(raw_rows)

    cleaned_rows: List[dict] = []
    filing_stats: Dict[str, Counter] = {}
    for filing_id, filing_rows in filing_groups.items():
        cleaned_group, stats = clean_filing_rows(filing_rows)
        cleaned_rows.extend(cleaned_group)
        filing_stats[filing_id] = stats

    cleaned_rows.sort(
        key=lambda row: (
            row["company_name"],
            row["filing_date"],
            int(row["annotation_index"]),
        )
    )

    write_csv(
        output_path,
        cleaned_rows,
        [
            "annotation_id",
            "filing_id",
            "ticker",
            "resolved_ticker",
            "company_name",
            "company_layer",
            "company_notes",
            "cik",
            "filing_date",
            "filing_year",
            "period_bucket",
            "comparison_window",
            "form",
            "accession_number",
            "primary_document",
            "source_url",
            "risk_section_char_count",
            "annotation_index",
            "merge_type",
            "start_paragraph_index",
            "end_paragraph_index",
            "start_paragraph_id",
            "end_paragraph_id",
            "source_paragraph_ids",
            "source_paragraph_count",
            "list_context_text",
            "list_item_index",
            "list_item_count",
            "text_char_count",
            "text",
        ],
    )

    report_rows = summarize_report(cleaned_rows, filing_stats)
    write_csv(report_path, report_rows, ["metric", "group", "value"])

    print(f"[done] Wrote {len(cleaned_rows)} annotation-ready rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
