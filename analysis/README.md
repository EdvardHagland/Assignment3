# Analysis

This folder is for the interpretation and notebook side of the project.

Current starter files:

- [COLAB_README.md](/C:/Users/edvar/Assignment3/analysis/COLAB_README.md)
  - copy-paste Google Colab blocks for loading the corpus, optionally regenerating it, and rendering a first exploratory clustering HTML report
- [exploratory_clustering/README.md](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/README.md)
  - the exploratory clustering workflow and supporting diagnostics for pre/post-2022 analysis

Main analysis files:

- [render_exploratory_report.py](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/render_exploratory_report.py)
  - builds embeddings, clusters the balanced sample, and writes a polished HTML report, an optional PDF, and CSV artifacts
- [report_template.html.j2](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/report_template.html.j2)
  - custom report layout and styling for the compiled HTML output
- [render_cluster_diagnostics.py](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/render_cluster_diagnostics.py)
  - CPU-only post-processing that re-ranks saved clusters by relative change, filters concentrated clusters, and computes within-cluster pre/post contrast terms
