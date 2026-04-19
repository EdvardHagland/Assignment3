#!/usr/bin/env python3
"""
Fetch SEC 10-K filings for a starter defense-firm universe and extract
Item 1A risk-factor text into annotation-ready chunks.

The script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SEC filings and extract 10-K Item 1A risk factors."
    )
    parser.add_argument(
        "--companies",
        default="config/defense_companies.csv",
        help="CSV with ticker and company_name columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/sec_defense_risk_corpus",
        help="Directory for extracted section and paragraph CSVs.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2022,
        help="Minimum filing year to include.",
    )
    parser.add_argument(
        "--form",
        default="10-K",
        help="SEC form type to pull. Recommended default: 10-K.",
    )
    parser.add_argument(
        "--limit-per-company",
        type=int,
        default=None,
        help="Optional cap on filings per company for testing.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Pause between SEC requests.",
    )
    parser.add_argument(
        "--user-agent",
        default=os.environ.get("SEC_USER_AGENT"),
        help="SEC-compliant User-Agent. Can also be set via SEC_USER_AGENT.",
    )
    return parser.parse_args()


def require_user_agent(value: Optional[str]) -> str:
    if value:
        return value
    raise SystemExit(
        "Missing SEC user agent. Pass --user-agent or set SEC_USER_AGENT "
        "to something like 'Your Name your.email@example.com'."
    )


def http_get(url: str, user_agent: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": user_agent,
        },
    )
    try:
        with urlopen(request) as response:
            raw_bytes = response.read()
            content_encoding = (response.info().get("Content-Encoding") or "").lower()
            if content_encoding == "gzip":
                raw_bytes = gzip.decompress(raw_bytes)
            return raw_bytes.decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise RuntimeError(f"HTTP error for {url}: {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc.reason}") from exc


def get_json(url: str, user_agent: str) -> dict:
    return json.loads(http_get(url, user_agent))


def load_companies(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def get_ticker_map(user_agent: str) -> Dict[str, dict]:
    payload = get_json(SEC_TICKER_CIK_URL, user_agent)
    ticker_map: Dict[str, dict] = {}
    for entry in payload.values():
        ticker = entry["ticker"].upper()
        ticker_map[ticker] = entry
    return ticker_map


def normalize_cik(raw_cik: int) -> str:
    return str(raw_cik).zfill(10)


def combine_filing_tables(submissions: dict, user_agent: str, sleep_seconds: float) -> List[dict]:
    tables = [columnar_to_rows(submissions["filings"]["recent"])]
    for file_info in submissions["filings"].get("files", []):
        file_name = file_info.get("name")
        if not file_name:
            continue
        url = f"https://data.sec.gov/submissions/{file_name}"
        extra = get_json(url, user_agent)
        tables.append(columnar_to_rows(extra))
        time.sleep(sleep_seconds)

    merged: List[dict] = []
    seen = set()
    for table in tables:
        for row in table:
            key = (row.get("accessionNumber"), row.get("form"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    return merged


def columnar_to_rows(table: dict) -> List[dict]:
    keys = list(table.keys())
    if not keys:
        return []
    length = len(table[keys[0]])
    rows = []
    for index in range(length):
        row = {key: table[key][index] for key in keys}
        rows.append(row)
    return rows


def filing_url(cik: str, accession_number: str, primary_document: str) -> str:
    cik_int = str(int(cik))
    accession_no_dashes = accession_number.replace("-", "")
    return f"{SEC_ARCHIVES_BASE}/{cik_int}/{accession_no_dashes}/{primary_document}"


def html_to_text(raw_html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "\n", raw_html)
    text = re.sub(r"(?is)<!--.*?-->", "\n", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(div|p|tr|table|li|ul|ol|h1|h2|h3|h4|h5|h6|section)>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_line_for_match(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def next_nonempty_line(lines: List[str], start_index: int) -> Optional[int]:
    for index in range(start_index, len(lines)):
        if lines[index].strip():
            return index
    return None


def is_10k_item_1a_heading(lines: List[str], index: int) -> Tuple[bool, int]:
    line = normalize_line_for_match(lines[index])
    if re.fullmatch(r"item\s+1a[\.\-:\s]*risk factors\.?", line, flags=re.IGNORECASE):
        return True, index + 1
    if re.fullmatch(r"item\s+1a[\.\-:\s]*", line, flags=re.IGNORECASE):
        next_index = next_nonempty_line(lines, index + 1)
        if next_index is not None:
            next_line = normalize_line_for_match(lines[next_index])
            if re.fullmatch(r"risk factors\.?", next_line, flags=re.IGNORECASE):
                return True, next_index + 1
    return False, index


def is_10k_item_end_heading(line: str) -> bool:
    text = normalize_line_for_match(line)
    return bool(
        re.fullmatch(r"item\s+1b[\.\-:\s].*", text, flags=re.IGNORECASE)
        or re.fullmatch(r"item\s+2[\.\-:\s].*", text, flags=re.IGNORECASE)
        or re.fullmatch(r"item\s+2", text, flags=re.IGNORECASE)
    )


def choose_longest_line_window(candidates: List[Tuple[int, int]], lines: List[str]) -> Optional[str]:
    viable = []
    for start, end in candidates:
        segment = "\n".join(lines[start:end]).strip()
        if 1000 <= len(segment) <= 500000:
            viable.append(segment)
    if not viable:
        return None
    return max(viable, key=len)


def extract_10k_risk_section(text: str) -> Optional[str]:
    lines = text.splitlines()
    starts: List[int] = []
    ends: List[int] = []

    for index in range(len(lines)):
        is_start, content_start = is_10k_item_1a_heading(lines, index)
        if is_start:
            starts.append(content_start)
        if is_10k_item_end_heading(lines[index]):
            ends.append(index)

    candidate_windows: List[Tuple[int, int]] = []
    for start in sorted(set(starts)):
        end = next((value for value in sorted(set(ends)) if value > start), None)
        if end is not None:
            candidate_windows.append((start, end))

    return choose_longest_line_window(candidate_windows, lines)


def extract_risk_section(text: str, form: str) -> Optional[str]:
    form = form.upper()
    if form == "10-K":
        return extract_10k_risk_section(text)
    return None


def is_noise_line(line: str) -> bool:
    text = normalize_line_for_match(line)
    if not text:
        return False
    if re.fullmatch(r"\d{1,3}", text):
        return True
    if text.lower() in {"table of contents", "part i", "part ii", "risk factors"}:
        return True
    if re.fullmatch(r"item\s+1a[\.\-:\s]*risk factors\.?", text, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"item\s+1a[\.\-:\s]*", text, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"item\s+1b[\.\-:\s].*", text, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"item\s+2[\.\-:\s].*", text, flags=re.IGNORECASE):
        return True
    if text in {"•", "-", "—"}:
        return True
    return False


def clean_section_text(section_text: str) -> List[str]:
    text = section_text
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    raw_lines = text.splitlines()
    cleaned_lines: List[str] = []
    previous_blank = False

    for raw_line in raw_lines:
        line = re.sub(r"\s+", " ", raw_line.strip())
        if not line:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue
        previous_blank = False
        if is_noise_line(line):
            continue
        cleaned_lines.append(line)

    while cleaned_lines and cleaned_lines[0] == "":
        cleaned_lines.pop(0)
    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()

    return cleaned_lines


def should_join_lines(previous: str, current: str) -> bool:
    prev = previous.rstrip()
    curr = current.lstrip()
    if not prev or not curr:
        return False
    if prev.endswith(":"):
        return True
    if not re.search(r"[.!?]$", prev):
        return True
    if curr[:1].islower():
        return True
    return False


def split_into_paragraphs(section_text: str) -> List[dict]:
    lines = clean_section_text(section_text)
    paragraphs: List[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        paragraph = re.sub(r"\s+", " ", current).strip()
        if len(paragraph) >= 80:
            paragraphs.append(paragraph)
        current = ""

    for line in lines:
        if line == "":
            flush()
            continue
        if not current:
            current = line
            continue
        if should_join_lines(current, line):
            separator = "; " if current.endswith(":") else " "
            current = current + separator + line
        else:
            flush()
            current = line

    flush()

    return [
        {
            "paragraph_index": index,
            "text": paragraph,
            "char_count": len(paragraph),
        }
        for index, paragraph in enumerate(paragraphs, start=1)
    ]


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    user_agent = require_user_agent(args.user_agent)

    companies_path = Path(args.companies)
    output_dir = Path(args.output_dir)

    companies = load_companies(companies_path)
    ticker_map = get_ticker_map(user_agent)

    section_rows: List[dict] = []
    paragraph_rows: List[dict] = []

    for company in companies:
        ticker = company["ticker"].upper()
        company_name = company["company_name"]
        sec_entry = ticker_map.get(ticker)
        if not sec_entry:
            print(f"[warn] Could not resolve ticker {ticker}", file=sys.stderr)
            continue

        cik = normalize_cik(sec_entry["cik_str"])
        submissions = get_json(SEC_SUBMISSIONS_URL.format(cik=cik), user_agent)
        filing_rows = combine_filing_tables(submissions, user_agent, args.sleep_seconds)

        matched = []
        for filing in filing_rows:
            if filing.get("form") != args.form:
                continue
            filing_date = filing.get("filingDate", "")
            if not filing_date or int(filing_date[:4]) < args.start_year:
                continue
            matched.append(filing)

        matched.sort(key=lambda row: row.get("filingDate", ""))
        if args.limit_per_company is not None:
            matched = matched[: args.limit_per_company]

        print(f"[info] {ticker}: {len(matched)} {args.form} filings", file=sys.stderr)

        for filing in matched:
            accession_number = filing["accessionNumber"]
            primary_document = filing["primaryDocument"]
            filing_date = filing["filingDate"]
            filing_year = int(filing_date[:4])
            source_url = filing_url(cik, accession_number, primary_document)
            raw_html = http_get(source_url, user_agent)
            filing_text = html_to_text(raw_html)
            risk_section = extract_risk_section(filing_text, args.form)
            time.sleep(args.sleep_seconds)

            if not risk_section:
                print(
                    f"[warn] No risk section extracted for {ticker} {accession_number}",
                    file=sys.stderr,
                )
                continue

            filing_id = f"{ticker}_{accession_number.replace('-', '')}"
            section_rows.append(
                {
                    "filing_id": filing_id,
                    "ticker": ticker,
                    "company_name": company_name,
                    "cik": cik,
                    "form": filing["form"],
                    "filing_date": filing_date,
                    "filing_year": filing_year,
                    "accession_number": accession_number,
                    "primary_document": primary_document,
                    "source_url": source_url,
                    "risk_section_char_count": len(risk_section),
                    "risk_section_text": risk_section,
                }
            )

            paragraphs = split_into_paragraphs(risk_section)
            for paragraph in paragraphs:
                paragraph_rows.append(
                    {
                        "paragraph_id": f"{filing_id}_p_{paragraph['paragraph_index']:03d}",
                        "filing_id": filing_id,
                        "ticker": ticker,
                        "company_name": company_name,
                        "cik": cik,
                        "filing_date": filing_date,
                        "filing_year": filing_year,
                        "form": filing["form"],
                        "accession_number": accession_number,
                        "primary_document": primary_document,
                        "source_url": source_url,
                        "risk_section_char_count": len(risk_section),
                        "paragraph_index": paragraph["paragraph_index"],
                        "char_count": paragraph["char_count"],
                        "text": paragraph["text"],
                    }
                )

    write_csv(
        output_dir / "processed" / "sec_10k_risk_sections.csv",
        section_rows,
        [
            "filing_id",
            "ticker",
            "company_name",
            "cik",
            "form",
            "filing_date",
            "filing_year",
            "accession_number",
            "primary_document",
            "source_url",
            "risk_section_char_count",
            "risk_section_text",
        ],
    )
    write_csv(
        output_dir / "processed" / "sec_10k_risk_paragraphs.csv",
        paragraph_rows,
        [
            "paragraph_id",
            "filing_id",
            "ticker",
            "company_name",
            "cik",
            "filing_date",
            "filing_year",
            "form",
            "accession_number",
            "primary_document",
            "source_url",
            "risk_section_char_count",
            "paragraph_index",
            "char_count",
            "text",
        ],
    )

    print(
        f"[done] Wrote {len(section_rows)} sections and {len(paragraph_rows)} paragraphs to {output_dir / 'processed'}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
