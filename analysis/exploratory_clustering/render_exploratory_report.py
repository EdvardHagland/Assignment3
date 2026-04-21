#!/usr/bin/env python3
"""
Render a polished exploratory clustering HTML report for the SEC defense corpus.

The report is designed for early analytical exploration rather than final claims.
It uses a balanced pre/post-2022 sample for embedding-based clustering while still
computing descriptive corpus figures from the full final dataset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plotly.offline import get_plotlyjs
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
import hdbscan
import umap


PRE_PERIOD = "pre_2022"
POST_PERIOD = "post_2022"


@dataclass
class ReportArtifacts:
    sampled_df: pd.DataFrame
    cluster_summary_df: pd.DataFrame
    representative_df: pd.DataFrame
    figures: Dict[str, str]
    summary_metrics: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an exploratory clustering HTML report for the SEC defense corpus."
    )
    parser.add_argument(
        "--dataset",
        default="data/final/sec_defense_risk_dataset.csv",
        help="Path to the canonical final dataset.",
    )
    parser.add_argument(
        "--output-html",
        default="analysis/exploratory_clustering/output/exploratory_clustering_report.html",
        help="Path to the rendered HTML report.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="analysis/exploratory_clustering/output",
        help="Directory for sampled CSV and cluster summary exports.",
    )
    parser.add_argument(
        "--template",
        default="analysis/exploratory_clustering/report_template.html.j2",
        help="Path to the HTML Jinja template.",
    )
    parser.add_argument(
        "--model-name",
        default="BAAI/bge-m3",
        help="Sentence-transformers compatible embedding model.",
    )
    parser.add_argument(
        "--sample-per-period",
        type=int,
        default=3000,
        help="Maximum rows to sample per period bucket for embedding and clustering.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling and dimensionality reduction.",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=25,
        help="UMAP neighborhood size.",
    )
    parser.add_argument(
        "--cluster-min-size",
        type=int,
        default=80,
        help="Minimum HDBSCAN cluster size.",
    )
    parser.add_argument(
        "--top-clusters",
        type=int,
        default=10,
        help="How many clusters to highlight in the HTML report.",
    )
    parser.add_argument(
        "--top-terms",
        type=int,
        default=8,
        help="How many top terms to show per cluster.",
    )
    parser.add_argument(
        "--examples-per-cluster",
        type=int,
        default=3,
        help="How many representative examples to show per cluster.",
    )
    return parser.parse_args()


def build_plotly_template() -> str:
    template_name = "defense_report"
    if template_name in pio.templates:
        return template_name

    pio.templates[template_name] = go.layout.Template(
        layout=go.Layout(
            font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", color="#13212c", size=15),
            title=dict(font=dict(family="Source Serif 4, Georgia, serif", size=22, color="#13212c")),
            paper_bgcolor="#f5f1e8",
            plot_bgcolor="#fffdf8",
            colorway=[
                "#0f4c5c",
                "#c56b3c",
                "#709255",
                "#9d4e5f",
                "#3e6c8f",
                "#c1a35f",
                "#5d576b",
                "#1b998b",
            ],
            margin=dict(l=30, r=30, t=70, b=40),
            hoverlabel=dict(
                bgcolor="#fffdf8",
                bordercolor="#d9cdb8",
                font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", color="#13212c"),
            ),
            xaxis=dict(showgrid=True, gridcolor="#ebe2d3", linecolor="#d6c9b7", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#ebe2d3", linecolor="#d6c9b7", zeroline=False),
            legend=dict(bgcolor="rgba(255,253,248,0.9)", bordercolor="#d9cdb8", borderwidth=1),
        )
    )
    return template_name


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["comparison_window"].isin(["pre_2018_2021", "post_2022_2025"])].copy()
    df["text"] = df["text"].fillna("").str.strip()
    df = df[df["text"] != ""].copy()
    return df


def sample_corpus(df: pd.DataFrame, sample_per_period: int, random_state: int) -> pd.DataFrame:
    return (
        df.groupby("period_bucket", group_keys=False)
        .apply(lambda part: part.sample(min(len(part), sample_per_period), random_state=random_state))
        .reset_index(drop=True)
    )


def embed_texts(df: pd.DataFrame, model_name: str, batch_size: int) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings)


def cluster_embeddings(
    embeddings: np.ndarray,
    random_state: int,
    umap_neighbors: int,
    cluster_min_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    cluster_umap = umap.UMAP(
        n_neighbors=umap_neighbors,
        n_components=15,
        metric="cosine",
        min_dist=0.0,
        random_state=random_state,
    )
    cluster_space = cluster_umap.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cluster_min_size,
        min_samples=max(10, cluster_min_size // 3),
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(cluster_space)

    viz_umap = umap.UMAP(
        n_neighbors=umap_neighbors,
        n_components=2,
        metric="cosine",
        min_dist=0.08,
        random_state=random_state,
    )
    coords_2d = viz_umap.fit_transform(embeddings)
    return labels, coords_2d


def cluster_label(cluster_id: int) -> str:
    if cluster_id == -1:
        return "Noise"
    return f"C{cluster_id:02d}"


def top_terms_by_cluster(df: pd.DataFrame, top_terms: int) -> Dict[int, str]:
    vectorizer = TfidfVectorizer(
        max_features=20000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.75,
    )
    matrix = vectorizer.fit_transform(df["text"])
    vocab = np.array(vectorizer.get_feature_names_out())

    summaries: Dict[int, str] = {}
    for cluster_id in sorted(df["cluster"].unique()):
        if cluster_id == -1:
            summaries[cluster_id] = "heterogeneous / noise"
            continue
        cluster_idx = np.where(df["cluster"].to_numpy() == cluster_id)[0]
        if len(cluster_idx) == 0:
            summaries[cluster_id] = ""
            continue
        mean_scores = np.asarray(matrix[cluster_idx].mean(axis=0)).ravel()
        top_idx = mean_scores.argsort()[::-1][:top_terms]
        summaries[cluster_id] = ", ".join(vocab[top_idx])
    return summaries


def representative_examples(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    examples_per_cluster: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    labels = df["cluster"].to_numpy()
    for cluster_id in sorted(df["cluster"].unique()):
        if cluster_id == -1:
            continue
        idx = np.where(labels == cluster_id)[0]
        cluster_vectors = embeddings[idx]
        centroid = cluster_vectors.mean(axis=0, keepdims=True)
        distances = cosine_distances(cluster_vectors, centroid).ravel()
        best_local = np.argsort(distances)[:examples_per_cluster]
        for rank, local_idx in enumerate(best_local, start=1):
            row = df.iloc[idx[local_idx]]
            rows.append(
                {
                    "cluster": cluster_id,
                    "cluster_label": cluster_label(cluster_id),
                    "rank": rank,
                    "annotation_id": row["annotation_id"],
                    "ticker": row["ticker"],
                    "company_layer": row["company_layer"],
                    "filing_year": int(row["filing_year"]),
                    "period_bucket": row["period_bucket"],
                    "text": row["text"],
                }
            )
    return pd.DataFrame(rows)


def build_cluster_summary(sampled_df: pd.DataFrame, top_term_map: Dict[int, str]) -> pd.DataFrame:
    cluster_counts = (
        sampled_df.groupby(["cluster", "cluster_label", "period_bucket"])
        .size()
        .reset_index(name="n")
    )
    cluster_counts["period_share"] = (
        cluster_counts.groupby("period_bucket")["n"].transform(lambda x: x / x.sum())
    )
    period_pivot = (
        cluster_counts.pivot(index=["cluster", "cluster_label"], columns="period_bucket", values="period_share")
        .fillna(0)
        .reset_index()
    )
    period_pivot["post_minus_pre"] = period_pivot.get(POST_PERIOD, 0) - period_pivot.get(PRE_PERIOD, 0)

    layer_counts = (
        sampled_df.groupby(["cluster", "cluster_label", "company_layer"])
        .size()
        .reset_index(name="n")
    )
    layer_counts["layer_share"] = (
        layer_counts.groupby("company_layer")["n"].transform(lambda x: x / x.sum())
    )
    layer_pivot = (
        layer_counts.pivot(index=["cluster", "cluster_label"], columns="company_layer", values="layer_share")
        .fillna(0)
        .reset_index()
    )

    totals = sampled_df.groupby(["cluster", "cluster_label"]).size().reset_index(name="cluster_size")
    summary = totals.merge(period_pivot, on=["cluster", "cluster_label"], how="left")
    summary = summary.merge(layer_pivot, on=["cluster", "cluster_label"], how="left")
    summary["top_terms"] = summary["cluster"].map(top_term_map)
    summary = summary.sort_values(["cluster_size", "post_minus_pre"], ascending=[False, False]).reset_index(drop=True)
    return summary


def corpus_overview_figure(df: pd.DataFrame, template_name: str) -> go.Figure:
    year_layer = (
        df.groupby(["filing_year", "company_layer"])
        .size()
        .reset_index(name="n")
        .sort_values("filing_year")
    )
    fig = px.bar(
        year_layer,
        x="filing_year",
        y="n",
        color="company_layer",
        barmode="stack",
        template=template_name,
        title="Corpus volume by filing year and company layer",
        labels={"filing_year": "Filing year", "n": "Annotation units", "company_layer": "Company layer"},
    )
    fig.update_layout(height=440)
    return fig


def sample_mix_figure(sampled_df: pd.DataFrame, template_name: str) -> go.Figure:
    mix = sampled_df.groupby(["period_bucket", "company_layer"]).size().reset_index(name="n")
    fig = px.bar(
        mix,
        x="period_bucket",
        y="n",
        color="company_layer",
        barmode="group",
        template=template_name,
        title="Balanced exploratory sample by period and layer",
        labels={"period_bucket": "Period bucket", "n": "Sample rows", "company_layer": "Company layer"},
    )
    fig.update_layout(height=420)
    return fig


def umap_period_figure(sampled_df: pd.DataFrame, template_name: str) -> go.Figure:
    fig = px.scatter(
        sampled_df,
        x="umap_x",
        y="umap_y",
        color="period_bucket",
        symbol="company_layer",
        template=template_name,
        title="Embedding space by period",
        opacity=0.72,
        hover_data={
            "annotation_id": True,
            "ticker": True,
            "company_layer": True,
            "filing_year": True,
            "cluster_label": True,
            "text_preview": True,
            "umap_x": False,
            "umap_y": False,
        },
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(height=620)
    return fig


def umap_cluster_figure(sampled_df: pd.DataFrame, template_name: str) -> go.Figure:
    fig = px.scatter(
        sampled_df,
        x="umap_x",
        y="umap_y",
        color="cluster_label",
        template=template_name,
        title="Embedding space by emergent cluster",
        opacity=0.72,
        hover_data={
            "annotation_id": True,
            "ticker": True,
            "company_layer": True,
            "filing_year": True,
            "period_bucket": True,
            "text_preview": True,
            "umap_x": False,
            "umap_y": False,
        },
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(height=620)
    return fig


def cluster_period_figure(cluster_summary_df: pd.DataFrame, top_clusters: int, template_name: str) -> go.Figure:
    top_df = (
        cluster_summary_df[cluster_summary_df["cluster"] != -1]
        .assign(abs_delta=lambda x: x["post_minus_pre"].abs())
        .sort_values("abs_delta", ascending=False)
        .head(top_clusters)
        .sort_values("post_minus_pre", ascending=False)
    )

    plot_df = top_df.melt(
        id_vars=["cluster_label"],
        value_vars=[PRE_PERIOD, POST_PERIOD],
        var_name="period_bucket",
        value_name="share",
    )
    fig = px.bar(
        plot_df,
        x="cluster_label",
        y="share",
        color="period_bucket",
        barmode="group",
        template=template_name,
        title="Cluster share comparison: pre-2022 versus post-2022",
        labels={"cluster_label": "Cluster", "share": "Share of sampled rows", "period_bucket": "Period bucket"},
    )
    fig.update_layout(height=460)
    return fig


def cluster_layer_figure(cluster_summary_df: pd.DataFrame, top_clusters: int, template_name: str) -> go.Figure:
    top_df = cluster_summary_df[cluster_summary_df["cluster"] != -1].sort_values("cluster_size", ascending=False).head(top_clusters)
    value_cols = [col for col in ["prime", "supplier"] if col in top_df.columns]
    plot_df = top_df.melt(
        id_vars=["cluster_label"],
        value_vars=value_cols,
        var_name="company_layer",
        value_name="share",
    )
    fig = px.bar(
        plot_df,
        x="cluster_label",
        y="share",
        color="company_layer",
        barmode="group",
        template=template_name,
        title="Cluster share comparison: primes versus suppliers",
        labels={"cluster_label": "Cluster", "share": "Share of sampled rows", "company_layer": "Company layer"},
    )
    fig.update_layout(height=460)
    return fig


def render_plot(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})


def render_report(args: argparse.Namespace, full_df: pd.DataFrame, artifacts: ReportArtifacts) -> str:
    template_path = Path(args.template)
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_path.name)

    top_clusters_df = (
        artifacts.cluster_summary_df[artifacts.cluster_summary_df["cluster"] != -1]
        .assign(abs_delta=lambda x: x["post_minus_pre"].abs())
        .sort_values(["abs_delta", "cluster_size"], ascending=[False, False])
        .head(args.top_clusters)
        .copy()
    )
    cluster_cards: List[dict] = []
    for _, row in top_clusters_df.iterrows():
        reps = artifacts.representative_df[artifacts.representative_df["cluster"] == row["cluster"]]
        examples = [
            {
                "ticker": rep["ticker"],
                "company_layer": rep["company_layer"],
                "filing_year": int(rep["filing_year"]),
                "period_bucket": rep["period_bucket"],
                "text": shorten(rep["text"], width=340, placeholder="..."),
            }
            for _, rep in reps.iterrows()
        ]
        cluster_cards.append(
            {
                "cluster_label": row["cluster_label"],
                "cluster_size": int(row["cluster_size"]),
                "top_terms": row["top_terms"],
                "pre_share": float(row.get(PRE_PERIOD, 0.0)),
                "post_share": float(row.get(POST_PERIOD, 0.0)),
                "delta": float(row["post_minus_pre"]),
                "prime_share": float(row.get("prime", 0.0)),
                "supplier_share": float(row.get("supplier", 0.0)),
                "examples": examples,
            }
        )

    return template.render(
        title="Defense Risk Narratives: Exploratory Clustering Report",
        subtitle="SEC 10-K Item 1A corpus, comparing 2018-2021 against 2022-2025",
        plotly_js=get_plotlyjs(),
        summary=artifacts.summary_metrics,
        figures=artifacts.figures,
        cluster_cards=cluster_cards,
        model_name=args.model_name,
        sample_per_period=args.sample_per_period,
        generated_from=str(Path(args.dataset).as_posix()),
    )


def main() -> None:
    args = parse_args()
    output_html = Path(args.output_html)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    template_name = build_plotly_template()
    full_df = load_dataset(Path(args.dataset))
    sampled_df = sample_corpus(full_df, args.sample_per_period, args.random_state)
    embeddings = embed_texts(sampled_df, args.model_name, args.batch_size)
    cluster_ids, coords_2d = cluster_embeddings(
        embeddings,
        random_state=args.random_state,
        umap_neighbors=args.umap_neighbors,
        cluster_min_size=args.cluster_min_size,
    )

    sampled_df = sampled_df.copy()
    sampled_df["cluster"] = cluster_ids
    sampled_df["cluster_label"] = sampled_df["cluster"].map(cluster_label)
    sampled_df["umap_x"] = coords_2d[:, 0]
    sampled_df["umap_y"] = coords_2d[:, 1]
    sampled_df["text_preview"] = sampled_df["text"].map(lambda text: shorten(text, width=180, placeholder="..."))

    top_term_map = top_terms_by_cluster(sampled_df, args.top_terms)
    representative_df = representative_examples(sampled_df, embeddings, args.examples_per_cluster)
    cluster_summary_df = build_cluster_summary(sampled_df, top_term_map)

    sampled_df.drop(columns=["text_preview"]).to_csv(artifacts_dir / "sampled_cluster_rows.csv", index=False)
    cluster_summary_df.to_csv(artifacts_dir / "cluster_summary.csv", index=False)
    representative_df.to_csv(artifacts_dir / "representative_examples.csv", index=False)

    figures = {
        "corpus_overview": render_plot(corpus_overview_figure(full_df, template_name)),
        "sample_mix": render_plot(sample_mix_figure(sampled_df, template_name)),
        "umap_period": render_plot(umap_period_figure(sampled_df, template_name)),
        "umap_cluster": render_plot(umap_cluster_figure(sampled_df, template_name)),
        "cluster_period": render_plot(cluster_period_figure(cluster_summary_df, args.top_clusters, template_name)),
        "cluster_layer": render_plot(cluster_layer_figure(cluster_summary_df, args.top_clusters, template_name)),
    }

    cluster_count = int((cluster_summary_df["cluster"] != -1).sum())
    noise_share = float((sampled_df["cluster"] == -1).mean())
    summary_metrics = {
        "full_rows": f"{len(full_df):,}",
        "sample_rows": f"{len(sampled_df):,}",
        "companies": f"{full_df['ticker'].nunique():,}",
        "prime_rows": f"{(full_df['company_layer'] == 'prime').sum():,}",
        "supplier_rows": f"{(full_df['company_layer'] == 'supplier').sum():,}",
        "clusters_found": f"{cluster_count}",
        "noise_share": f"{noise_share:.1%}",
        "model_name": args.model_name,
    }

    report_artifacts = ReportArtifacts(
        sampled_df=sampled_df,
        cluster_summary_df=cluster_summary_df,
        representative_df=representative_df,
        figures=figures,
        summary_metrics=summary_metrics,
    )

    html = render_report(args, full_df, report_artifacts)
    output_html.write_text(html, encoding="utf-8")

    metadata = {
        "dataset": args.dataset,
        "output_html": str(output_html),
        "artifacts_dir": str(artifacts_dir),
        "model_name": args.model_name,
        "sample_per_period": args.sample_per_period,
        "cluster_min_size": args.cluster_min_size,
        "umap_neighbors": args.umap_neighbors,
        "summary_metrics": summary_metrics,
    }
    (artifacts_dir / "report_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Rendered HTML report to: {output_html}")
    print(f"Artifacts written to: {artifacts_dir}")


if __name__ == "__main__":
    main()
