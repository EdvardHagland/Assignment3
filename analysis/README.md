# Analysis

This folder is for the interpretation and notebook side of the project.

Current starter files:

- [COLAB_README.md](/C:/Users/edvar/Assignment3/analysis/COLAB_README.md)
  - copy-paste Google Colab blocks for loading the corpus, optionally regenerating it, rendering the full-corpus period-shift artifacts, and optionally producing the Gemini-assisted final report when secrets are available
- [requirements-colab.txt](/C:/Users/edvar/Assignment3/analysis/requirements-colab.txt)
  - dependency list for the maintained Colab analysis and report workflow
- [METHODOLOGICAL_CRITIQUE.md](/C:/Users/edvar/Assignment3/analysis/METHODOLOGICAL_CRITIQUE.md)
  - honest self-assessment of what the embedding + LLM pipeline sees and what it structurally cannot see (intra-cluster churn, post-period temporal dynamics, concentration, category blending, HDBSCAN sensitivity, genre-versus-reality)
- [exploratory_clustering/README.md](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/README.md)
  - the exploratory clustering workflow for separate pre/post discovery and approximate cluster matching

Main analysis files:

- [render_period_shift_report.py](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/render_period_shift_report.py)
  - embeds a balanced sample or full corpus once, clusters `pre_2022` and `post_2022` separately, matches post clusters back to pre clusters, and writes the main HTML/PDF/CSV artifacts
- [period_shift_template.html.j2](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/period_shift_template.html.j2)
  - custom report layout and styling for the period-shift report
- [render_period_shift_llm_report.py](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/render_period_shift_llm_report.py)
  - second-stage Gemini-assisted narrative report that packages evidence for interesting changed clusters and writes an executive abstract plus evidence-aware cluster cards
- [period_shift_llm_template.html.j2](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/period_shift_llm_template.html.j2)
  - template used by the Gemini-assisted narrative report
- [render_exploratory_report.py](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/render_exploratory_report.py)
  - legacy single-map clustering report kept for comparison with the older workflow
- [report_template.html.j2](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/report_template.html.j2)
  - template used by the legacy single-map report
- [render_cluster_diagnostics.py](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/render_cluster_diagnostics.py)
  - CPU-only post-processing for legacy global-clustering artifacts
