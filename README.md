# Assignment 3

## Project

**Title:** Corporate Anxiety After 2022: Risk Narratives in Defense-Sector Filings

**Core question:** Which kinds of strategic risk dominate defense-sector risk disclosures after 2022, and how do those patterns vary across firms and over time?

## Why this corpus works

This project uses the SEC's structured filing system rather than scraping arbitrary company websites. The core corpus is:

- U.S.-listed defense firms
- annual `10-K` filings
- only `Item 1A. Risk Factors`

That gives us a formally defined risk corpus instead of "everything companies say." We are not trying to infer risk from the whole filing. We are analyzing the legally designated risk-disclosure section.

## Proposed method

The main method should be **manual coding plus supervised classification**, not clustering as the main engine.

Workflow:

1. Download defense-firm `10-K` filings from the SEC.
2. Extract `Item 1A. Risk Factors`.
3. Split each section into paragraph-level units.
4. Label a shared annotation sample using the team codebook.
5. Measure overlap and resolve disagreements.
6. Train a lightweight classifier on the final labels.
7. Apply the classifier to the full corpus.
8. Compare label frequencies across firms and years.
9. Close-read representative chunks for interpretation.

## Repository Structure

- `pipeline/`
  - SEC scraper and corpus-cleaning scripts
- `analysis/`
  - proposal, annotation materials, and future notebooks
- `config/`
  - firm universe
- `data/sec_defense_risk_corpus/processed/`
  - canonical section, paragraph, and annotation-unit CSVs

## Initial firm universe

The starter firm list is in [config/defense_companies.csv](/C:/Users/edvar/OneDrive/Skrivebord/Assignment%203/config/defense_companies.csv).

It includes large primes and a few smaller defense-focused firms:

- Lockheed Martin
- RTX
- Northrop Grumman
- General Dynamics
- L3Harris
- Huntington Ingalls
- Leidos
- Booz Allen Hamilton
- Kratos
- AeroVironment

This list is intentionally editable. If you want a stricter "pure defense manufacturing" corpus, trim out the more services-heavy firms.

## Annotation Design

The annotation materials live in `analysis/annotation/`.

Suggested collaborative workflow:

1. Label a shared pilot sample with all five coders.
2. Refine the codebook where disagreement is high.
3. Double-code the main sample.
4. Adjudicate conflicts.
5. Train on the adjudicated labels.

See:

- [codebook.md](/C:/Users/edvar/OneDrive/Skrivebord/Assignment%203/analysis/annotation/codebook.md)
- [annotation_plan.md](/C:/Users/edvar/OneDrive/Skrivebord/Assignment%203/analysis/annotation/annotation_plan.md)
- [labels_template.csv](/C:/Users/edvar/OneDrive/Skrivebord/Assignment%203/analysis/annotation/labels_template.csv)

## Pipeline

The SEC extraction pipeline is in [sec_fetch_risk_factors.py](/C:/Users/edvar/OneDrive/Skrivebord/Assignment%203/pipeline/sec_fetch_risk_factors.py).

The annotation-unit cleaner is in [prepare_annotation_paragraphs.py](/C:/Users/edvar/OneDrive/Skrivebord/Assignment%203/pipeline/prepare_annotation_paragraphs.py).

The pipeline is designed to:

1. resolve tickers to CIKs using SEC data
2. fetch company submissions metadata
3. download `10-K` filings
4. extract `Item 1A. Risk Factors`
5. split the section into paragraphs
6. enrich and clean those paragraphs into annotation-ready units

Both scripts use only the Python standard library.

## Canonical Outputs

The canonical processed corpus lives in `data/sec_defense_risk_corpus/processed/`:

- [sec_10k_risk_sections.csv](/C:/Users/edvar/OneDrive/Skrivebord/Assignment%203/data/sec_defense_risk_corpus/processed/sec_10k_risk_sections.csv)
- [sec_10k_risk_paragraphs.csv](/C:/Users/edvar/OneDrive/Skrivebord/Assignment%203/data/sec_defense_risk_corpus/processed/sec_10k_risk_paragraphs.csv)
- [sec_10k_risk_annotation_units.csv](/C:/Users/edvar/OneDrive/Skrivebord/Assignment%203/data/sec_defense_risk_corpus/processed/sec_10k_risk_annotation_units.csv)
- [sec_10k_risk_cleaning_report.csv](/C:/Users/edvar/OneDrive/Skrivebord/Assignment%203/data/sec_defense_risk_corpus/processed/sec_10k_risk_cleaning_report.csv)

The cleaned dataset now includes filing-level metadata such as:

- `ticker`
- `company_name`
- `cik`
- `filing_date`
- `filing_year`
- `period_bucket`
- `accession_number`
- `primary_document`
- `source_url`
- `merge_type`
- source paragraph identifiers and indices

## Quick start

Set a SEC-compliant user agent and run:

```powershell
$env:SEC_USER_AGENT="Your Name your.email@example.com"
python pipeline/sec_fetch_risk_factors.py --start-year 2018
python pipeline/prepare_annotation_paragraphs.py
```

The pipeline will write output under `data/sec_defense_risk_corpus/processed/`.

## Next recommended steps

1. Review the current annotation-unit schema and decide whether bullet lists should be split into separate risk claims.
2. Trim the firm list if the corpus feels too broad.
3. Pilot-label `100-150` annotation units with all five annotators.
4. Revise the codebook before the main annotation round.
