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

## Block 1: Clone or refresh the repository

This version is safe to rerun in the same Colab session. If `/content/Assignment3`
already exists, it refreshes the checkout instead of failing on a second clone.

```python
!if [ ! -d /content/Assignment3/.git ]; then git clone https://github.com/EdvardHagland/Assignment3.git /content/Assignment3; fi
%cd /content/Assignment3
!git fetch origin
!git checkout main
!git pull --ff-only origin main
```

## Block 2: Install dependencies

```python
!pip -q install -r analysis/requirements-colab.txt
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

## Block 6: Render the period-shift HTML report and PDF

This is now the recommended exploratory path.

The current default model is `BAAI/bge-m3`.

```python
!python analysis/exploratory_clustering/render_period_shift_report.py \
    --model-name BAAI/bge-m3 \
    --sample-per-period 3000 \
    --scatter-display-max-points 3500 \
    --cluster-min-size 80 \
    --output-html analysis/exploratory_clustering/output/period_shift_report.html \
    --output-pdf analysis/exploratory_clustering/output/period_shift_report.pdf
```

This will write:

- `analysis/exploratory_clustering/output/period_shift_report.html`
- `analysis/exploratory_clustering/output/period_shift_report.pdf`
- `analysis/exploratory_clustering/output/sampled_cluster_rows.csv`
- `analysis/exploratory_clustering/output/sampled_embeddings.npz`
- `analysis/exploratory_clustering/output/period_cluster_summary.csv`
- `analysis/exploratory_clustering/output/cluster_matches.csv`
- `analysis/exploratory_clustering/output/pairwise_cluster_similarities.csv`
- `analysis/exploratory_clustering/output/representative_examples.csv`
- `analysis/exploratory_clustering/output/period_shift_metadata.json`

The HTML stays interactive. The PDF is the safer static sharing version.

## Block 7: Download the compiled outputs

```python
from google.colab import files
import os

files.download("analysis/exploratory_clustering/output/period_shift_report.html")
if os.path.exists("analysis/exploratory_clustering/output/period_shift_report.pdf"):
    files.download("analysis/exploratory_clustering/output/period_shift_report.pdf")
else:
    print("PDF was not generated. Check analysis/exploratory_clustering/output/period_shift_metadata.json for the export error.")
```

## What we want from this first pass

The first exploratory pass is not meant to produce the final argument. It is meant to help us answer:

- what themes organize the `pre_2022` corpus?
- what themes organize the `post_2022` corpus?
- which post clusters look genuinely new versus persistent?
- where do broad pre themes split into more specific post subthemes?
- are the largest post clusters broad sector themes or concentrated company artifacts?

## Suggested next steps after the first Colab run

1. Save the most interpretable new and split/refined post clusters.
2. Read the representative examples for those clusters closely.
3. Use `cluster_matches.csv` and `pairwise_cluster_similarities.csv` to sanity-check ambiguous mappings.
4. Pair the thematic-structure findings with filing-level contrastive term or phrase analysis.
5. Treat the legacy `render_exploratory_report.py` and `render_cluster_diagnostics.py` path as a comparison baseline, not the default workflow.

## Block 8: Render the final integrated Gemini-assisted report

This second-stage pass assumes your Colab runtime already has Colab Secrets named `GEMINI_API_KEY` and `GEMINI_MODEL`, and that you have granted notebook access to them.

It does not send one giant prompt. It sends one structured request per interesting post cluster, saves those outputs, then asks Gemini for one abstract on top of the cluster-level analyses.

The final HTML is rendered after those Gemini calls finish, so the abstract and cluster writeups are integrated into the same report HTML together with the clustering figures.

It now reuses the exact sampled embeddings from the same discovery run for evidence selection. A second local embedding pass is only allowed if you explicitly opt into `--allow-reembed` for older artifacts that were created before `sampled_embeddings.npz` was added.

The selection is strict by default:

- only post clusters already flagged as broad themes
- only the match types you explicitly list in `--interesting-match-types`
- no fallback to weaker clusters if nothing passes the filter

```python
import os
from google.colab import userdata

os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
os.environ["GEMINI_MODEL"] = userdata.get("GEMINI_MODEL")
```

```python
!python analysis/exploratory_clustering/render_period_shift_llm_report.py \
    --sampled-rows analysis/exploratory_clustering/output/sampled_cluster_rows.csv \
    --sampled-embeddings analysis/exploratory_clustering/output/sampled_embeddings.npz \
    --dataset data/final/sec_defense_risk_dataset.csv \
    --period-cluster-summary analysis/exploratory_clustering/output/period_cluster_summary.csv \
    --pairwise-similarities analysis/exploratory_clustering/output/pairwise_cluster_similarities.csv \
    --cluster-matches analysis/exploratory_clustering/output/cluster_matches.csv \
    --representative-examples analysis/exploratory_clustering/output/representative_examples.csv \
    --metadata analysis/exploratory_clustering/output/period_shift_metadata.json \
    --output-html analysis/exploratory_clustering/output/period_shift_llm_report.html \
    --interesting-match-types new_post_only,split/refined,merged \
    --max-clusters 6 \
    --central-examples 4 \
    --mid-examples 4 \
    --peripheral-examples 4 \
    --matched-pre-examples 3
```

This will write:

- `analysis/exploratory_clustering/output/period_shift_llm_report.html`
- `analysis/exploratory_clustering/output/llm_cluster_evidence.json`
- `analysis/exploratory_clustering/output/llm_cluster_analysis_progress.json`
- `analysis/exploratory_clustering/output/llm_cluster_analyses.json`
- `analysis/exploratory_clustering/output/llm_abstract.json`
- `analysis/exploratory_clustering/output/llm_report_metadata.json`

## Block 9: Download the LLM report outputs

```python
from google.colab import files
import os

if os.path.exists("analysis/exploratory_clustering/output/period_shift_llm_report.html"):
    files.download("analysis/exploratory_clustering/output/period_shift_llm_report.html")
else:
    print("LLM report HTML was not generated.")

if os.path.exists("analysis/exploratory_clustering/output/llm_cluster_analyses.json"):
    files.download("analysis/exploratory_clustering/output/llm_cluster_analyses.json")

if os.path.exists("analysis/exploratory_clustering/output/llm_abstract.json"):
    files.download("analysis/exploratory_clustering/output/llm_abstract.json")

files.download("analysis/exploratory_clustering/output/period_shift_metadata.json")
```
