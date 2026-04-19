# Assignment 3

## Project

**Title:** Corporate Anxiety After 2022: Risk Narratives in Defense-Sector Filings

**Core question:** Which kinds of strategic risk dominate defense-sector risk disclosures after 2022, and how do those patterns vary across firms and over time?

**Main comparison window:** `2018-2021` versus `2022-2025`

## Repository structure

- `scraper/`
  - SEC scraping and cleaning scripts
- `data/intermediate/`
  - extracted sections, raw paragraphs, and cleaning reports
- `data/final/`
  - the one canonical analysis dataset
- `data/fine_tuning/`
  - annotation and training-data artifacts built from the final corpus
- `fine_tuning/`
  - labeling workflow overview
- `fine_tuning/labeling/`
  - the annotation app, config, protocol, codebook, and labeling scripts in one place
- `analysis/`
  - proposal, notebooks, figures, and interpretation
- `config/`
  - firm universe and project configuration

## Corpus logic

This project uses the SEC's structured EDGAR system rather than arbitrary investor-relations pages. The corpus is limited to:

- U.S.-listed defense firms across a prime and supplier layer
- annual `10-K` filings
- only `Item 1A. Risk Factors`

That keeps the corpus tied to a clearly defined disclosure regime instead of mixing together unrelated corporate text. The company universe now carries `company_layer` metadata so we can compare prime contractors against upstream suppliers without maintaining separate corpora.

## Cleaning logic

The cleaner moves from filing text to annotation-ready risk claims in several passes:

1. extract `Item 1A` using filing structure rather than keyword guesses
2. remove filing noise such as repeated headings, page-number debris, and table-of-contents artifacts
3. merge broken line wraps and short heading-body pairs
4. drop generic risk-summary material when it duplicates fuller discussion later in the section
5. split bullet lists into separate rows when each bullet functions as its own risk claim

The bullet-list split matters because many firms introduce a lead sentence and then enumerate distinct risks underneath it. In the final dataset, each bullet becomes its own row while the shared lead context is carried forward into the cleaned text.

## Fine-tuning and labeling logic

We are treating the training set as a proper collaborative annotation exercise.

Important distinction:

- the full SEC corpus stays intact as the main analysis dataset
- the annotation pool is a sampled subset drawn from that corpus
- the supervised train, validation, and test splits should be created inside the labeled subset later

So no, we do not need to delete labeled rows from the original corpus. We just need to track which rows were sampled for annotation and which labeled rows eventually become part of the supervised dataset.

Rules:

1. no item is single-coded
2. every item is labeled by at least `2` annotators
3. disagreements go back into the queue for another labeling round
4. items with persistent disagreement are dropped from the final training set

The goal is not to keep every row at all costs. The goal is to keep only rows that produce a reliable supervised dataset.

## Current canonical dataset

Main dataset:

- `data/final/sec_defense_risk_dataset.csv`

Supporting reproducibility files:

- `data/intermediate/sec_10k_risk_sections.csv`
- `data/intermediate/sec_10k_risk_paragraphs.csv`
- `data/intermediate/sec_10k_risk_cleaning_report.csv`

Future training-data products will live in:

- `data/fine_tuning/`

The local annotation database also lives there, because the app is meant to run from your laptop while ngrok exposes it outward.

## Quick start

Set a SEC-compliant user agent and run:

```powershell
$env:SEC_USER_AGENT="Your Name your.email@example.com"
python scraper/sec_fetch_risk_factors.py --start-year 2018
python scraper/prepare_annotation_paragraphs.py
```

Then build and serve the annotation workflow:

```powershell
python fine_tuning/labeling/scripts/build_annotation_pool.py
python fine_tuning/labeling/scripts/init_annotation_db.py --reset
python fine_tuning/labeling/run_annotation_app.py
```

If you want to expose it over ngrok, point ngrok at the local Flask port after the app is running.

For backend visibility:

- app dashboard: `http://127.0.0.1:5000/admin`
- local database file: `data/fine_tuning/annotation.sqlite3`
- label definitions: `fine_tuning/labeling/config/label_options.json`

The annotator interface is intentionally minimal: users register with an email, see only the text, click one label, and the app advances immediately to the next item.

The current annotation scheme uses `15` defense-specific risk categories. The only catch-all class is `OTHER_UNCLEAR`, which should be used sparingly and treated as a recycle signal rather than a final analytical category.

## Next step

The structure is now organized around scraping, one canonical corpus, and one explicit fine-tuning workflow. The next job is to build the fine-tuning dataset from the final corpus without breaking the double-label protocol.
