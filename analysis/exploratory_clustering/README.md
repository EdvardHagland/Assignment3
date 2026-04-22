# Exploratory Clustering

This folder contains the maintained exploratory clustering workflow for pre/post-2022 differences in risk language.

Current purpose:

- prepare for a first exploratory clustering pass
- compare `pre_2022` versus `post_2022`
- inspect representative examples before we lock in stronger analytical claims
- track how the thematic structure changed, not just which rows move in one shared cluster map

Primary report pipeline:

- [render_period_shift_report.py](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/render_period_shift_report.py)
- [period_shift_template.html.j2](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/period_shift_template.html.j2)

This is now the recommended exploratory path. It:

- embeds all sampled rows once with the same model
- clusters `pre_2022` and `post_2022` separately
- summarizes each cluster with terms, representative examples, company breadth, and filing breadth
- matches post clusters back to the pre system approximately using centroid cosine similarity and term overlap

Optional narrative synthesis layer:

- [render_period_shift_llm_report.py](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/render_period_shift_llm_report.py)
- [period_shift_llm_template.html.j2](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/period_shift_llm_template.html.j2)

This second-stage pass:

- reads the saved period-shift artifacts
- selects only the most interesting changed post clusters
- packages centroid-near and moderately peripheral evidence for each cluster
- sends one Gemini request per cluster
- asks Gemini for a critical report abstract on top of those structured cluster analyses

Legacy single-map pipeline:

- [render_exploratory_report.py](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/render_exploratory_report.py)
- [report_template.html.j2](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/report_template.html.j2)

These older files are still available if we want to compare against the one-global-cluster-map approach.

Secondary diagnostics pass:

- [render_cluster_diagnostics.py](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/render_cluster_diagnostics.py)

This diagnostics script is tied to the legacy global-clustering artifacts. It is not the default follow-up for the separate-discovery workflow.
