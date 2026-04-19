# Assignment 3 Scaffold

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
3. Split each section into risk-factor chunks.
4. Label a shared annotation sample using the team codebook.
5. Measure overlap and resolve disagreements.
6. Train a lightweight classifier on the final labels.
7. Apply the classifier to the full corpus.
8. Compare label frequencies across firms and years.
9. Close-read representative chunks for interpretation.

## Unit of analysis

The unit should be a **risk-factor chunk**, not a whole filing.

This keeps the documents:

- focused on risk
- numerous enough for the assignment
- large enough to be interpretable by human coders

## Expected corpus shape

If we start with `10` defense firms and pull `10-K` filings from `2022` onward, we should get a manageable filing set. After splitting the `Item 1A` sections into chunks, the corpus should comfortably pass the assignment's minimum document threshold.

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

## Annotation design

The annotation materials live in `annotation/`.

Suggested collaborative workflow:

1. Label a shared pilot sample with all five coders.
2. Refine the codebook where disagreement is high.
3. Double-code the main sample.
4. Adjudicate conflicts.
5. Train on the adjudicated labels.

See:

- [annotation/codebook.md](/C:/Users/edvar/OneDrive/Skrivebord/Assignment%203/annotation/codebook.md)
- [annotation/annotation_plan.md](/C:/Users/edvar/OneDrive/Skrivebord/Assignment%203/annotation/annotation_plan.md)
- [annotation/labels_template.csv](/C:/Users/edvar/OneDrive/Skrivebord/Assignment%203/annotation/labels_template.csv)

## SEC ingestion

The starter SEC pipeline is in [scripts/sec_fetch_risk_factors.py](/C:/Users/edvar/OneDrive/Skrivebord/Assignment%203/scripts/sec_fetch_risk_factors.py).

It is designed to:

1. resolve tickers to CIKs using SEC data
2. fetch company submissions metadata
3. download `10-K` filings
4. extract `Item 1A. Risk Factors`
5. split the section into chunks
6. write section and chunk CSVs for annotation

The script uses only the Python standard library.

## Quick start

Set a SEC-compliant user agent and run:

```powershell
$env:SEC_USER_AGENT="Your Name your.email@example.com"
python scripts/sec_fetch_risk_factors.py --start-year 2022
```

The script will write output under `data/`.

## Next recommended steps

1. Run the SEC extraction script and inspect the chunk quality.
2. Trim the firm list if the corpus feels too broad.
3. Pilot-label `100-150` chunks with all five annotators.
4. Revise the codebook before the main annotation round.
