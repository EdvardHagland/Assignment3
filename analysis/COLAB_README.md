# Google Colab Launch Guide

This file is a copy-paste guide for launching the project in Google Colab.

The normal path should be:

1. clone the repo
2. install analysis dependencies
3. load the finished dataset already stored in GitHub
4. run the exploratory clustering report script
5. download the compiled HTML report

The dataset-generation block is included too, but it is optional. In most cases we should skip it and use the finished CSV already in the repository.

## Recommended Colab runtime

- `Python 3`
- a `T4 GPU` is already enough for a serious exploratory pass
- if you land on an `H100`, just keep the same workflow and increase the sample size if you want

## Block 1: Clone the repository

```python
!git clone https://github.com/EdvardHagland/Assignment3.git
%cd /content/Assignment3
```

## Block 2: Install dependencies

```python
!pip -q install pandas numpy scikit-learn matplotlib seaborn plotly jinja2 kaleido reportlab sentence-transformers umap-learn hdbscan bertopic
```

## Block 3: Optional dataset generation

Skip this block if we just want to analyze the finished dataset already in the repo.

If we do run it, we need a SEC-compliant user agent.

```python
import os

os.environ["SEC_USER_AGENT"] = "Your Name your.email@example.com"
```

```python
!python scraper/sec_fetch_risk_factors.py --start-year 2018
!python scraper/prepare_annotation_paragraphs.py
```

This will rebuild:

- `data/intermediate/processed/sec_10k_risk_sections.csv`
- `data/intermediate/processed/sec_10k_risk_paragraphs.csv`
- `data/intermediate/sec_10k_risk_cleaning_report.csv`
- `data/final/sec_defense_risk_dataset.csv`

## Block 4: Load the finished dataset

```python
import pandas as pd

DATA_PATH = "data/final/sec_defense_risk_dataset.csv"
df = pd.read_csv(DATA_PATH)

print(df.shape)
df.head(3)
```

## Block 5: Basic corpus checks

```python
display(df["comparison_window"].value_counts(dropna=False))
display(df["period_bucket"].value_counts(dropna=False))
display(df["company_layer"].value_counts(dropna=False))
display(df.groupby("ticker").size().sort_values(ascending=False).head(15))
```

```python
df[["annotation_id", "ticker", "company_layer", "filing_year", "period_bucket", "text"]].sample(5, random_state=42)
```

## Block 6: Render the HTML report and PDF

This is now the main exploratory path.

The current default model is `BAAI/bge-m3`.

```python
!python analysis/exploratory_clustering/render_exploratory_report.py \
    --model-name BAAI/bge-m3 \
    --sample-per-period 3000 \
    --scatter-display-max-points 3500 \
    --cluster-min-size 80 \
    --output-html analysis/exploratory_clustering/output/exploratory_clustering_report.html \
    --output-pdf analysis/exploratory_clustering/output/exploratory_clustering_report.pdf
```

This will write:

- `analysis/exploratory_clustering/output/exploratory_clustering_report.html`
- `analysis/exploratory_clustering/output/exploratory_clustering_report.pdf`
- `analysis/exploratory_clustering/output/sampled_cluster_rows.csv`
- `analysis/exploratory_clustering/output/cluster_summary.csv`
- `analysis/exploratory_clustering/output/representative_examples.csv`
- `analysis/exploratory_clustering/output/report_metadata.json`

The HTML stays interactive. The PDF is the safer static sharing version.

## Block 7: Download the compiled outputs

```python
from google.colab import files
files.download("analysis/exploratory_clustering/output/exploratory_clustering_report.html")
files.download("analysis/exploratory_clustering/output/exploratory_clustering_report.pdf")
```

## Block 8: Repair an already-generated HTML file without rerunning the model

If you already have a generated HTML report in the runtime and the plots are blank in the browser, you can repair the standalone file without touching embeddings or clustering:

```python
!python analysis/exploratory_clustering/fix_standalone_html.py \
    analysis/exploratory_clustering/output/exploratory_clustering_report.html
```

## Optional manual workflow

If you want to inspect the mechanics manually rather than using the one-command report script, the disposable files under `analysis/exploratory_clustering/DELETE_ME_*` are still there as a scratch area.

## What we want from this first pass

The first exploratory pass is not meant to produce the final argument. It is meant to help us answer:

- do pre- and post-2022 texts distribute differently across clusters?
- which clusters look genuinely interpretable?
- do some clusters map onto likely hand-label categories?
- which clusters should we inspect closely with representative examples?
- are prime and supplier firms landing in systematically different parts of the cluster space?

## Suggested next steps after the first Colab run

1. Save the most interpretable clusters and representative excerpts.
2. Compare whether primes and suppliers land in different parts of the cluster space.
3. Decide whether BERTopic is worth keeping as a supporting analysis.
4. Use what we learn here to refine the manual codebook and later supervised classification.
5. If the HTML report looks good, use it directly as a shareable exploratory artifact for the group.
