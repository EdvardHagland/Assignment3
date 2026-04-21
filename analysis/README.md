# Analysis

This folder is for the interpretation and notebook side of the project.

Current starter files:

- [COLAB_README.md](/C:/Users/edvar/Assignment3/analysis/COLAB_README.md)
  - copy-paste Google Colab blocks for loading the corpus, optionally regenerating it, and rendering a first exploratory clustering HTML report
- [exploratory_clustering/README.md](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/README.md)
  - a lightweight sandbox for early pre/post-2022 clustering work

Main analysis files:

- [render_exploratory_report.py](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/render_exploratory_report.py)
  - builds embeddings, clusters the balanced sample, and writes a polished HTML report, an optional PDF, and CSV artifacts
- [report_template.html.j2](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/report_template.html.j2)
  - custom report layout and styling for the compiled HTML output
- [fix_standalone_html.py](/C:/Users/edvar/Assignment3/analysis/exploratory_clustering/fix_standalone_html.py)
  - repairs an already-generated HTML report so it works as a local standalone file

The `DELETE_ME_` files under `analysis/exploratory_clustering/` are intentional scratch files. They are there to make early experimentation easy, and they can be removed once the final analysis workflow is clearer.
