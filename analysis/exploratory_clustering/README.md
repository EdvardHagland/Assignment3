# Exploratory Clustering

This folder contains the maintained exploratory clustering workflow for pre/post-2022 differences in risk language.

Current purpose:

- compare `pre_2022` versus `post_2022`
- render the full-corpus period-shift clustering report
- save reusable cluster, match, example, and embedding artifacts
- run the Gemini-assisted qualitative synthesis from those artifacts

Primary report pipeline:

- [render_period_shift_report.py](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/render_period_shift_report.py)
- [period_shift_template.html.j2](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/period_shift_template.html.j2)

This is now the recommended exploratory path. It:

- embeds all sampled rows once with the same model
- clusters `pre_2022` and `post_2022` separately
- summarizes each cluster with terms, representative examples, company breadth, and filing breadth
- matches post clusters back to the pre system approximately using centroid cosine similarity and term overlap

Gemini-assisted qualitative synthesis layer:

- [render_period_shift_llm_report.py](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/render_period_shift_llm_report.py)
- [period_shift_llm_template.html.j2](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/period_shift_llm_template.html.j2)

This second-stage pass:

- reads the saved period-shift artifacts
- selects only the most interesting changed post clusters
- packages centroid-near and moderately peripheral evidence for each cluster
- sends one Gemini request per cluster
- asks Gemini for a critical report abstract on top of those structured cluster analyses

Generated outputs are written to `analysis/exploratory_clustering/output/`.
That directory is ignored by Git because the reports and CSV artifacts are
recreated by the Colab notebook or by rerunning the scripts locally.
