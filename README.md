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
- `analysis/`
  - Colab launch guide, exploratory report pipeline, diagnostics, figures, and interpretation
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

## Current canonical dataset

Main dataset:

- `data/final/sec_defense_risk_dataset.csv`

Supporting reproducibility files:

- `data/intermediate/processed/sec_10k_risk_sections.csv`
- `data/intermediate/processed/sec_10k_risk_paragraphs.csv`
- `data/intermediate/processed/sec_10k_risk_coverage_report.csv`
- `data/intermediate/sec_10k_risk_cleaning_report.csv`

## Quick start

Set a SEC-compliant user agent and run:

```powershell
$env:SEC_USER_AGENT="Your Name your.email@example.com"
python scraper/sec_fetch_risk_factors.py --start-year 2018
python scraper/prepare_annotation_paragraphs.py
```

Then move into the maintained analysis workflow:

- `analysis/COLAB_README.md`
- `analysis/README.md`
- `analysis/exploratory_clustering/render_period_shift_report.py`
- `analysis/exploratory_clustering/render_period_shift_llm_report.py`

## Next step

The current project is centered on one canonical corpus and one maintained analysis path: separate pre/post cluster discovery plus evidence-constrained LLM interpretation. The next job is to tighten the methodological critique and final presentation around that pipeline.

For analysis preparation, start here:

- `analysis/COLAB_README.md`
- `analysis/README.md`
- `analysis/exploratory_clustering/render_period_shift_report.py`
- `analysis/exploratory_clustering/period_shift_template.html.j2`
- `analysis/exploratory_clustering/render_period_shift_llm_report.py`
- `analysis/exploratory_clustering/period_shift_llm_template.html.j2`
- `analysis/exploratory_clustering/render_exploratory_report.py`
- `analysis/exploratory_clustering/report_template.html.j2`
- `analysis/exploratory_clustering/render_cluster_diagnostics.py`
- `analysis/exploratory_clustering/`


