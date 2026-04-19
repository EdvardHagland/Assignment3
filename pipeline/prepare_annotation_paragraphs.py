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
        default="data/sec_defense_risk_corpus/processed/sec_10k_risk_paragraphs.csv",
        help="Path to the raw paragraph CSV from sec_fetch_risk_factors.py",
    )
    parser.add_argument(
        "--sections",
        default="data/sec_defense_risk_corpus/processed/sec_10k_risk_sections.csv",
        help="Path to the extracted section CSV used to enrich row metadata.",
    )
    parser.add_argument(
        "--output",
        default="data/sec_defense_risk_corpus/processed/sec_10k_risk_annotation_units.csv",
        help="Output CSV path for cleaned annotation rows.",
    )
    parser.add_argument(
        "--report",
        default="data/sec_defense_risk_corpus/processed/sec_10k_risk_cleaning_report.csv",
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
            "company_name": row.get("company_name", ""),
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
        combined["period_bucket"] = "pre_2022" if filing_year and int(filing_year) < 2022 else "post_2022"
        enriched.append(combined)
    return enriched


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalized_lower(text: str) -> str:
    return normalize_text(text).lower()


def sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?](?=\s|$)", text))


def bullet_like(text: str) -> bool:
    return bool(re.search(r"(?:^|[\s;])[•●u]\s", text))


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
        row["char_count"] = str(len(text))
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
                    "company_name": current["company_name"],
                    "cik": current.get("cik", ""),
                    "filing_date": current["filing_date"],
                    "filing_year": current.get("filing_year", ""),
                    "period_bucket": current.get("period_bucket", ""),
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
                    "company_name": current["company_name"],
                    "cik": current.get("cik", ""),
                    "filing_date": current["filing_date"],
                    "filing_year": current.get("filing_year", ""),
                    "period_bucket": current.get("period_bucket", ""),
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
                "company_name": current["company_name"],
                "cik": current.get("cik", ""),
                "filing_date": current["filing_date"],
                "filing_year": current.get("filing_year", ""),
                "period_bucket": current.get("period_bucket", ""),
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

    return merged_rows, stats


def summarize_report(cleaned_rows: List[dict], filing_stats: Dict[str, Counter]) -> List[dict]:
    by_company = Counter(row["company_name"] for row in cleaned_rows)
    by_year = Counter(row["filing_date"][:4] for row in cleaned_rows)
    total_stats = Counter()
    for stats in filing_stats.values():
        total_stats.update(stats)

    report_rows: List[dict] = [
        {"metric": "cleaned_rows", "group": "all", "value": len(cleaned_rows)},
        {"metric": "dropped_intro", "group": "all", "value": total_stats["dropped_intro"]},
        {"metric": "dropped_summary", "group": "all", "value": total_stats["dropped_summary"]},
        {"metric": "merged_broken_pairs", "group": "all", "value": total_stats["merged_broken_pairs"]},
        {"metric": "merged_heading_pairs", "group": "all", "value": total_stats["merged_heading_pairs"]},
    ]

    for company_name, count in sorted(by_company.items()):
        report_rows.append(
            {"metric": "cleaned_rows_by_company", "group": company_name, "value": count}
        )
    for year, count in sorted(by_year.items()):
        report_rows.append(
            {"metric": "cleaned_rows_by_year", "group": year, "value": count}
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
            "company_name",
            "cik",
            "filing_date",
            "filing_year",
            "period_bucket",
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
