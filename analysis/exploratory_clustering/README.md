# Exploratory Clustering

This folder is for early analysis experiments around pre/post-2022 differences in risk language.

The idea is to keep this area lightweight and disposable while we figure out what is actually worth keeping in the final workflow.

Current purpose:

- prepare for a first exploratory clustering pass
- compare `pre_2022` versus `post_2022`
- inspect representative examples before we lock in stronger analytical claims

Primary report pipeline:

- [render_exploratory_report.py](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/render_exploratory_report.py)
- [report_template.html.j2](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/report_template.html.j2)

These are the real files for the first presentation-quality exploratory pass.

The report pipeline can now generate:

- an interactive HTML report for exploration
- a static PDF for safer sharing and presentation

The `DELETE_ME_` files in this folder are intentional scratch files. They are here to make experimentation easy, and they can be removed once we know what belongs in the final analysis workflow.


Secondary diagnostics pass:

- [render_cluster_diagnostics.py](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/render_cluster_diagnostics.py)

This pass is CPU-only and is meant for the moment when the first clustering report feels too boilerplate-heavy. It re-ranks clusters by relative movement, filters out company-dominated clusters, and adds within-cluster pre/post contrast terms from the saved sampled rows.
