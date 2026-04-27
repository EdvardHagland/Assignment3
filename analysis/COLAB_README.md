# Google Colab Launch Guide

This file is a copy-paste guide for launching the project in Google Colab.

## Quick start

Use this path when you want to reproduce the dataset and clustering artifacts.
After installing dependencies, choose one data source: either load the finished
dataset already stored in GitHub, or rerun the scraper and cleaner in Colab. The
Gemini-assisted report is optional because it requires private Colab Secrets.

1. clone the repo
2. install analysis dependencies
3. choose one data source: included GitHub CSV or regenerated SEC corpus
4. run basic corpus checks
5. render the full-corpus period-shift clustering artifacts
6. download the non-LLM period-shift report
7. optionally render the Gemini-assisted narrative report if Gemini secrets are available
8. download `period_shift_llm_report.html` if the optional LLM step was run

Minimal block sequence:

- Block 1: clone or refresh the repository
- Block 2: install dependencies
- Block 3A or 3B: choose and load the dataset
- Block 4: preview the loaded dataset
- Block 5: run basic corpus checks
- Block 6: render the full-corpus period-shift report
- Block 7: download the period-shift report outputs

Optional LLM sequence:

- Block 8: render the final integrated Gemini-assisted report, requires Colab Secrets
- Block 9: download the LLM report outputs

For the fastest path, use Block 3A. Use Block 3B only if you want to prove the
corpus can be regenerated from SEC filings.

## Recommended Colab runtime

- `Python 3`
- a `T4 GPU` is enough for the maintained workflow
- if you land on an `H100`, keep the same full-corpus workflow and expect it to run faster

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

## Block 3: Choose one data source

Run either Block 3A or Block 3B. Both options end by loading the same canonical
CSV into `df`, so the rest of the notebook is identical after this point.

### Block 3A: Use the included GitHub dataset

This is the recommended quick-start path.

```python
import pandas as pd

DATA_PATH = "data/final/sec_defense_risk_dataset.csv"
df = pd.read_csv(DATA_PATH)
```

### Block 3B: Regenerate the dataset from SEC filings

Use this path if you want to rerun the scraper and cleaner. You need a
SEC-compliant user agent before making SEC requests.

```python
import os
import pandas as pd

os.environ["SEC_USER_AGENT"] = "Your Name your.email@example.com"

!python scraper/sec_fetch_risk_factors.py --start-year 2018
!python scraper/prepare_annotation_paragraphs.py

DATA_PATH = "data/final/sec_defense_risk_dataset.csv"
df = pd.read_csv(DATA_PATH)
```

Block 3B rebuilds:

- `data/intermediate/processed/sec_10k_risk_sections.csv`
- `data/intermediate/processed/sec_10k_risk_paragraphs.csv`
- `data/intermediate/sec_10k_risk_cleaning_report.csv`
- `data/final/sec_defense_risk_dataset.csv`

## Block 4: Preview the loaded dataset

```python
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

## Block 6: Render the full-corpus period-shift report

This is the maintained discovery path for the final report. It embeds the corpus,
clusters the pre-2022 and post-2022 subsets separately, then writes the artifacts
that the Gemini-assisted report uses as evidence.

The current default model is `BAAI/bge-m3`.

The recommended final run uses the full corpus. `--sample-per-period 0` tells the
script to use all available rows in each period bucket rather than sampling.
`--scatter-display-max-points` only limits how many points are drawn in the
browser figures; it does not limit clustering.

```python
!python analysis/exploratory_clustering/render_period_shift_report.py \
    --model-name BAAI/bge-m3 \
    --sample-per-period 0 \
    --scatter-display-max-points 6000 \
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

## Block 8: Optional Gemini-assisted report

This second-stage pass requires Colab Secrets named `GEMINI_API_KEY` and
`GEMINI_MODEL`, with notebook access granted to both. A grader should not be
expected to have these secrets already loaded. If the secrets are unavailable,
skip Blocks 8 and 9; Blocks 1 through 7 still reproduce the dataset path and the
non-LLM clustering report.

It does not send one giant prompt. It sends one structured request per interesting post cluster, saves those outputs, then asks Gemini for one abstract on top of the cluster-level analyses.

The final HTML is rendered after those Gemini calls finish, so the abstract and cluster writeups are integrated into the same report HTML together with the clustering figures.

It now reuses the exact sampled embeddings from the same discovery run for evidence selection. A second local embedding pass is only allowed if you explicitly opt into `--allow-reembed` for older artifacts that were created before `sampled_embeddings.npz` was added.

The selection is strict by default:

- only post clusters already flagged as broad themes
- only the match types you explicitly list in `--interesting-match-types`
- no fallback to weaker clusters if nothing passes the filter

The recommended final run uses `--max-clusters 0` and
`--persistent-audit-clusters 0`. That tells the script to include all eligible
changed clusters and all eligible persistent audit clusters rather than trimming
to a top slice.

```python
import os
from google.colab import userdata

gemini_api_key = userdata.get("GEMINI_API_KEY")
gemini_model = userdata.get("GEMINI_MODEL")

if not gemini_api_key or not gemini_model:
    raise RuntimeError(
        "Block 8 requires Colab Secrets named GEMINI_API_KEY and GEMINI_MODEL. "
        "Skip Blocks 8 and 9 if you are running without Gemini credentials."
    )

os.environ["GEMINI_API_KEY"] = gemini_api_key
os.environ["GEMINI_MODEL"] = gemini_model
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
    --max-clusters 0 \
    --persistent-audit-clusters 0 \
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

Run this only if Block 8 completed successfully.

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
