# Assignment 3

**Title:** Corporate Anxiety After 2022: Risk Narratives in Defense-Sector Filings

**Core question:** Which kinds of strategic risk dominate defense-sector risk disclosures after 2022, and how do those patterns vary across firms and over time?

**Main comparison window:** `2018-2021` versus `2022-2025`

## Repository Map

```text
Assignment3/
├── README.md
├── config/
│   └── defense_companies.csv
│       Firm universe with ticker, company name, prime/supplier layer, and notes.
├── scraper/
│   ├── README.md
│   ├── sec_fetch_risk_factors.py
│   │   Resolves configured firms through SEC data, downloads 10-K filings,
│   │   extracts Item 1A, and writes section and paragraph files.
│   └── prepare_annotation_paragraphs.py
│       Cleans extracted paragraphs into final annotation-ready rows.
├── data/
│   ├── final/
│   │   ├── sec_defense_risk_dataset.csv
│   │   │   Canonical dataset used by the analysis workflow.
│   │   └── data_dictionary.md
│   │       Column definitions for the final dataset.
└── analysis/
    ├── README.md
    ├── COLAB_README.ipynb
    │   Importable Google Colab run guide.
    ├── requirements-colab.txt
    │   Python dependencies for the Colab analysis workflow.
    └── exploratory_clustering/
        ├── README.md
        ├── render_period_shift_report.py
        │   Full-corpus or sampled pre/post clustering and matching report.
        ├── period_shift_template.html.j2
        │   Template for the period-shift clustering report.
        ├── render_period_shift_llm_report.py
        │   Gemini-assisted qualitative report built from saved artifacts.
        └── period_shift_llm_template.html.j2
            Template for the Gemini-assisted report.
```

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

## Current Canonical Dataset

Main dataset:

- `data/final/sec_defense_risk_dataset.csv`

The scraper and cleaner regenerate intermediate CSVs under `data/intermediate/`.
Those generated files are ignored so the repository stays focused on the source
code, company configuration, final dataset, and run guide.

## Quick start

For the easiest Colab path, import:

- `analysis/COLAB_README.ipynb`

The notebook lets the runner either load the included GitHub dataset or regenerate the dataset from SEC filings. The Gemini-assisted qualitative report requires Colab Secrets for `GEMINI_API_KEY` and `GEMINI_MODEL`.

For local dataset regeneration, run:

```powershell
$env:SEC_USER_AGENT="Your Name your.email@example.com"
python scraper/sec_fetch_risk_factors.py --start-year 2018
python scraper/prepare_annotation_paragraphs.py
```

Then move into the maintained analysis workflow:

- `analysis/COLAB_README.ipynb`
- `analysis/README.md`
- `analysis/exploratory_clustering/render_period_shift_report.py`
- `analysis/exploratory_clustering/render_period_shift_llm_report.py`

