# Analysis

This folder is for the interpretation and notebook side of the project.

Current starter files:

- [COLAB_README.ipynb](/C:/Users/edvar/Assignment3/analysis/COLAB_README.ipynb)
  - importable Google Colab run guide for loading or regenerating the corpus, rendering the full-corpus period-shift artifacts, and producing the Gemini-assisted qualitative report when secrets are available
- [requirements-colab.txt](/C:/Users/edvar/Assignment3/analysis/requirements-colab.txt)
  - dependency list for the maintained Colab analysis and report workflow
- [METHODOLOGICAL_CRITIQUE.md](/C:/Users/edvar/Assignment3/analysis/METHODOLOGICAL_CRITIQUE.md)
  - methodological notes on what the embedding + LLM pipeline captures and where it has limitations, including intra-cluster churn, post-period temporal dynamics, concentration, category blending, HDBSCAN sensitivity, and disclosure genre effects
- [exploratory_clustering/README.md](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/README.md)
  - the exploratory clustering workflow for separate pre/post discovery and approximate cluster matching

Main analysis files:

- [render_period_shift_report.py](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/render_period_shift_report.py)
  - embeds a balanced sample or full corpus once, clusters `pre_2022` and `post_2022` separately, matches post clusters back to pre clusters, and writes the main HTML/PDF/CSV artifacts
- [period_shift_template.html.j2](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/period_shift_template.html.j2)
  - custom report layout and styling for the period-shift report
- [render_period_shift_llm_report.py](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/render_period_shift_llm_report.py)
  - second-stage Gemini-assisted qualitative report that packages evidence for interesting changed clusters and writes an executive abstract plus evidence-aware cluster cards
- [period_shift_llm_template.html.j2](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/period_shift_llm_template.html.j2)
  - template used by the Gemini-assisted narrative report
- [render_exploratory_report.py](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/render_exploratory_report.py)
  - legacy single-map clustering report kept for comparison with the older workflow
- [report_template.html.j2](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/report_template.html.j2)
  - template used by the legacy single-map report
- [render_cluster_diagnostics.py](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/render_cluster_diagnostics.py)
  - CPU-only post-processing for legacy global-clustering artifacts
