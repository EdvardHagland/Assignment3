#!/usr/bin/env python3
"""
Render a polished exploratory clustering HTML report for the SEC defense corpus.

The report is designed for early analytical exploration rather than final claims.
It uses a balanced pre/post-2022 sample for embedding-based clustering while still
computing descriptive corpus figures from the full final dataset.
"""

from __future__ import annotations

import argparse
import html
import json
import re
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
FOCUS_MIN_CLUSTER_SIZE = 150
FOCUS_MIN_TICKER_COUNT = 10
FOCUS_MAX_TOP_TICKER_SHARE = 0.35


@dataclass
class ReportArtifacts:
    sampled_df: pd.DataFrame
    display_df: pd.DataFrame
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
    parser.add_argument(
        "--scatter-display-max-points",
        type=int,
        default=4000,
        help="Maximum number of points to display in browser scatter plots.",
    )
    parser.add_argument(
        "--output-pdf",
        default="",
        help="Optional path for a static PDF export.",
    )
    parser.add_argument(
        "--pdf-title",
        default="Defense Risk Narratives: Exploratory Clustering Report",
        help="Title used in the optional PDF export.",
    )
    return parser.parse_args()


def build_plotly_template() -> str:
    template_name = "defense_report"
    if template_name in pio.templates:
        return template_name

    pio.templates[template_name] = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", color="#1a2a35", size=13),
            title=dict(font=dict(family="Source Serif 4, Georgia, serif", size=20, color="#1a2a35")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,253,249,0.5)",
            colorway=[
                "#0f4c5c",
                "#c56b3c",
                "#5a8a3e",
                "#9d4e5f",
                "#3e6c8f",
                "#c1a35f",
                "#5d576b",
                "#1b998b",
            ],
            margin=dict(l=40, r=24, t=56, b=36),
            hoverlabel=dict(
                bgcolor="#fffdf8",
                bordercolor="#d9cdb8",
                font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color="#1a2a35", size=12),
            ),
            xaxis=dict(showgrid=True, gridcolor="rgba(221,209,190,0.5)", linecolor="#d6c9b7", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(221,209,190,0.5)", linecolor="#d6c9b7", zeroline=False),
            legend=dict(bgcolor="rgba(255,253,248,0.85)", bordercolor="rgba(221,209,190,0.6)", borderwidth=1),
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


def build_display_sample(df: pd.DataFrame, max_points: int, random_state: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.copy()

    group_count = max(df["cluster_label"].nunique(), 1)
    per_group = max(1, max_points // group_count)
    display_df = (
        df.groupby("cluster_label", group_keys=False)
        .apply(lambda part: part.sample(min(len(part), per_group), random_state=random_state))
        .reset_index(drop=True)
    )
    if len(display_df) > max_points:
        display_df = display_df.sample(max_points, random_state=random_state).reset_index(drop=True)
    return display_df


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


def parse_top_ticker_n(top_tickers: str) -> int:
    match = re.search(r"\((\d+)\)", str(top_tickers))
    return int(match.group(1)) if match else 0


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
    ticker_coverage = (
        sampled_df.groupby(["cluster", "cluster_label"])["ticker"]
        .nunique()
        .reset_index(name="ticker_count")
    )
    top_tickers = (
        sampled_df.groupby(["cluster", "cluster_label", "ticker"])
        .size()
        .reset_index(name="n")
        .sort_values(["cluster", "n", "ticker"], ascending=[True, False, True])
        .groupby(["cluster", "cluster_label"])
        .head(3)
    )
    top_ticker_strings = (
        top_tickers.groupby(["cluster", "cluster_label"])
        .apply(lambda x: ", ".join(f"{row.ticker} ({int(row.n)})" for _, row in x.iterrows()), include_groups=False)
        .reset_index(name="top_tickers")
    )
    summary = totals.merge(period_pivot, on=["cluster", "cluster_label"], how="left")
    summary = summary.merge(layer_pivot, on=["cluster", "cluster_label"], how="left")
    summary = summary.merge(ticker_coverage, on=["cluster", "cluster_label"], how="left")
    summary = summary.merge(top_ticker_strings, on=["cluster", "cluster_label"], how="left")
    summary["top_terms"] = summary["cluster"].map(top_term_map)
    summary["top_ticker_n"] = summary["top_tickers"].map(parse_top_ticker_n)
    summary["top_ticker_share"] = np.where(summary["cluster_size"] > 0, summary["top_ticker_n"] / summary["cluster_size"], np.nan)
    summary["abs_delta"] = summary["post_minus_pre"].abs()
    summary["sector_signal_score"] = (
        summary["abs_delta"]
        * np.log1p(summary["cluster_size"])
        * np.log1p(summary["ticker_count"].clip(lower=1))
        * (1 - summary["top_ticker_share"].fillna(1).clip(lower=0, upper=0.95))
    )
    summary["eligible_sector_shift"] = (
        (summary["cluster"] != -1)
        & (summary["cluster_size"] >= FOCUS_MIN_CLUSTER_SIZE)
        & (summary["ticker_count"] >= FOCUS_MIN_TICKER_COUNT)
        & (summary["top_ticker_share"] <= FOCUS_MAX_TOP_TICKER_SHARE)
    )
    summary["exclusion_reason"] = ""
    summary.loc[summary["cluster"] == -1, "exclusion_reason"] = "noise"
    summary.loc[(summary["cluster"] != -1) & (summary["cluster_size"] < FOCUS_MIN_CLUSTER_SIZE), "exclusion_reason"] = "too_small"
    summary.loc[(summary["cluster"] != -1) & (summary["ticker_count"] < FOCUS_MIN_TICKER_COUNT), "exclusion_reason"] = "too_few_tickers"
    summary.loc[(summary["cluster"] != -1) & (summary["top_ticker_share"] > FOCUS_MAX_TOP_TICKER_SHARE), "exclusion_reason"] = "too_concentrated"
    summary.loc[summary["eligible_sector_shift"], "exclusion_reason"] = ""
    summary = summary.sort_values(
        ["eligible_sector_shift", "sector_signal_score", "abs_delta", "cluster_size"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
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


def select_focus_clusters(cluster_summary_df: pd.DataFrame, top_clusters: int) -> pd.DataFrame:
    eligible = cluster_summary_df[
        (cluster_summary_df["cluster"] != -1) & (cluster_summary_df["eligible_sector_shift"])
    ].copy()
    if len(eligible) < top_clusters:
        eligible = cluster_summary_df[cluster_summary_df["cluster"] != -1].copy()

    return (
        eligible.sort_values(["sector_signal_score", "abs_delta", "cluster_size"], ascending=[False, False, False])
        .head(top_clusters)
        .sort_values("post_minus_pre", ascending=False)
        .copy()
    )


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


def umap_period_figure(display_df: pd.DataFrame, template_name: str) -> go.Figure:
    fig = px.scatter(
        display_df,
        x="umap_x",
        y="umap_y",
        color="period_bucket",
        symbol="company_layer",
        render_mode="webgl",
        template=template_name,
        title="Embedding space by period",
        opacity=0.72,
        hover_data={
            "annotation_id": True,
            "ticker": True,
            "company_layer": True,
            "filing_year": True,
            "cluster_label": True,
            "umap_x": False,
            "umap_y": False,
        },
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(height=620)
    return fig


def umap_cluster_figure(display_df: pd.DataFrame, template_name: str) -> go.Figure:
    fig = px.scatter(
        display_df,
        x="umap_x",
        y="umap_y",
        color="cluster_label",
        render_mode="webgl",
        template=template_name,
        title="Embedding space by emergent cluster",
        opacity=0.72,
        hover_data={
            "annotation_id": True,
            "ticker": True,
            "company_layer": True,
            "filing_year": True,
            "period_bucket": True,
            "umap_x": False,
            "umap_y": False,
        },
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(height=620)
    return fig


def cluster_year_heatmap(sampled_df: pd.DataFrame, cluster_summary_df: pd.DataFrame, top_clusters: int, template_name: str) -> go.Figure:
    top_labels = select_focus_clusters(cluster_summary_df, top_clusters)["cluster_label"].tolist()
    heatmap_df = sampled_df[sampled_df["cluster_label"].isin(top_labels)].copy()
    heatmap_df = (
        heatmap_df.groupby(["cluster_label", "filing_year"])
        .size()
        .reset_index(name="n")
    )
    heatmap_df["year_share"] = (
        heatmap_df.groupby("filing_year")["n"].transform(lambda x: x / x.sum())
    )
    ordered_labels = top_labels[::-1]
    fig = px.imshow(
        heatmap_df.pivot(index="cluster_label", columns="filing_year", values="year_share")
        .reindex(index=ordered_labels)
        .fillna(0),
        color_continuous_scale=["#fff8eb", "#d7e7ea", "#7aa6b8", "#0f4c5c"],
        aspect="auto",
        template=template_name,
        title="Top shifted clusters by filing year share",
        labels=dict(x="Filing year", y="Cluster", color="Share"),
    )
    fig.update_layout(height=460, coloraxis_colorbar=dict(title="Share"))
    return fig


def cluster_period_figure(cluster_summary_df: pd.DataFrame, top_clusters: int, template_name: str) -> go.Figure:
    top_df = select_focus_clusters(cluster_summary_df, top_clusters)

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
    top_df = select_focus_clusters(cluster_summary_df, top_clusters)
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


def build_examples_lookup(sampled_df: pd.DataFrame, representative_df: pd.DataFrame, max_examples: int = 3) -> Dict[int, List[dict]]:
    lookup: Dict[int, List[dict]] = {}

    if not representative_df.empty and {"cluster", "text", "ticker"}.issubset(representative_df.columns):
        for cluster_id, part in representative_df.groupby("cluster"):
            examples = []
            for _, rep in part.head(max_examples).iterrows():
                examples.append(
                    {
                        "ticker": rep["ticker"],
                        "company_layer": rep["company_layer"],
                        "filing_year": int(rep["filing_year"]),
                        "period_bucket": rep["period_bucket"],
                        "text": shorten(rep["text"], width=340, placeholder="..."),
                    }
                )
            lookup[int(cluster_id)] = examples

    for cluster_id, part in sampled_df.groupby("cluster"):
        cluster_id = int(cluster_id)
        if cluster_id in lookup:
            continue
        examples = []
        for period in [POST_PERIOD, PRE_PERIOD]:
            period_part = part[part["period_bucket"] == period]
            if period_part.empty:
                continue
            row = period_part.iloc[0]
            examples.append(
                {
                    "ticker": row["ticker"],
                    "company_layer": row["company_layer"],
                    "filing_year": int(row["filing_year"]),
                    "period_bucket": row["period_bucket"],
                    "text": shorten(row["text"], width=340, placeholder="..."),
                }
            )
            if len(examples) >= max_examples:
                break
        lookup[cluster_id] = examples

    return lookup


def build_cluster_cards(
    cluster_rows_df: pd.DataFrame,
    examples_lookup: Dict[int, List[dict]],
) -> List[dict]:
    cluster_cards: List[dict] = []
    for _, row in cluster_rows_df.iterrows():
        cluster_id = int(row["cluster"])
        if cluster_id == -1:
            status = "Noise"
        elif row.get("eligible_sector_shift", False):
            status = "Notable broad shift"
        else:
            status = f"Documented but filtered ({row.get('exclusion_reason', 'filtered')})"
        cluster_cards.append(
            {
                "cluster_label": row["cluster_label"],
                "cluster": cluster_id,
                "cluster_size": int(row["cluster_size"]),
                "ticker_count": int(row.get("ticker_count", 0)),
                "top_tickers": row.get("top_tickers", ""),
                "top_terms": row["top_terms"],
                "pre_share": float(row.get(PRE_PERIOD, 0.0)),
                "post_share": float(row.get(POST_PERIOD, 0.0)),
                "delta": float(row["post_minus_pre"]),
                "prime_share": float(row.get("prime", 0.0)),
                "supplier_share": float(row.get("supplier", 0.0)),
                "top_ticker_share": float(row.get("top_ticker_share", 0.0)),
                "sector_signal_score": float(row.get("sector_signal_score", 0.0)),
                "status": status,
                "examples": examples_lookup.get(cluster_id, []),
            }
        )
    return cluster_cards


def export_figure_png(fig: go.Figure, path: Path, width: int = 1400, height: int = 860, scale: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), format="png", width=width, height=height, scale=scale)


def build_pdf_report(
    output_pdf: Path,
    title: str,
    subtitle: str,
    summary_metrics: Dict[str, str],
    figure_paths: Dict[str, Path],
    cluster_cards: List[dict],
) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError(
            "PDF export requires reportlab. Install it in Colab with `pip install reportlab kaleido`."
        ) from exc

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=42,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#13212c"),
        spaceAfter=10,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#566572"),
        spaceAfter=14,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#13212c"),
        spaceAfter=8,
        spaceBefore=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#13212c"),
    )
    small_style = ParagraphStyle(
        "Small",
        parent=body_style,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#566572"),
    )

    story = [
        Paragraph(title, title_style),
        Paragraph(subtitle, subtitle_style),
    ]

    metric_rows = [
        ["Full corpus rows", summary_metrics["full_rows"], "Exploratory sample", summary_metrics["sample_rows"]],
        ["Companies", summary_metrics["companies"], "Clusters found", summary_metrics["clusters_found"]],
        ["Prime rows", summary_metrics["prime_rows"], "Supplier rows", summary_metrics["supplier_rows"]],
        ["Noise share", summary_metrics["noise_share"], "Embedding model", summary_metrics["model_name"]],
    ]
    metric_table = Table(metric_rows, colWidths=[1.5 * inch, 1.0 * inch, 1.7 * inch, 2.7 * inch])
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f3eb")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d8ccb8")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d8ccb8")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#13212c")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([metric_table, Spacer(1, 0.22 * inch)])

    figure_sequence = [
        ("Corpus structure", "corpus_overview"),
        ("Balanced exploratory sample", "sample_mix"),
        ("UMAP by period", "umap_period"),
        ("UMAP by cluster", "umap_cluster"),
        ("Pre/post cluster comparison", "cluster_period"),
        ("Prime/supplier cluster comparison", "cluster_layer"),
        ("Cluster change by year", "cluster_year_heatmap"),
    ]
    for label, key in figure_sequence:
        if key not in figure_paths:
            continue
        story.append(Paragraph(label, heading_style))
        story.append(Image(str(figure_paths[key]), width=6.8 * inch, height=4.1 * inch))
        story.append(Spacer(1, 0.18 * inch))

    story.append(PageBreak())
    story.append(Paragraph("Highlighted clusters", heading_style))
    story.append(
        Paragraph(
            "These cards summarize the most shifted clusters from the exploratory pass. They are not final labels, but they help identify promising themes for close reading and later supervised annotation.",
            body_style,
        )
    )
    story.append(Spacer(1, 0.16 * inch))

    for cluster in cluster_cards:
        story.append(
            Paragraph(
                f"<b>{html.escape(cluster['cluster_label'])}</b> · {cluster['cluster_size']} sampled rows · {cluster['ticker_count']} tickers",
                body_style,
            )
        )
        story.append(Paragraph(f"<b>Top terms:</b> {html.escape(cluster['top_terms'])}", small_style))
        if cluster["top_tickers"]:
            story.append(Paragraph(f"<b>Top tickers:</b> {html.escape(cluster['top_tickers'])}", small_style))
        story.append(
            Paragraph(
                f"<b>Pre share:</b> {cluster['pre_share'] * 100:.1f}% &nbsp;&nbsp; <b>Post share:</b> {cluster['post_share'] * 100:.1f}% &nbsp;&nbsp; "
                f"<b>Prime share:</b> {cluster['prime_share'] * 100:.1f}% &nbsp;&nbsp; <b>Supplier share:</b> {cluster['supplier_share'] * 100:.1f}%",
                small_style,
            )
        )
        for example in cluster["examples"][:2]:
            story.append(
                Paragraph(
                    f"<b>{html.escape(example['ticker'])}</b> · {html.escape(example['company_layer'])} · {example['filing_year']} · {html.escape(example['period_bucket'])}<br/>{html.escape(example['text'])}",
                    small_style,
                )
            )
        story.append(Spacer(1, 0.18 * inch))

    doc.build(story)


def render_report(
    args: argparse.Namespace,
    full_df: pd.DataFrame,
    artifacts: ReportArtifacts,
    cluster_cards: List[dict],
    all_cluster_cards: List[dict],
) -> str:
    template_path = Path(args.template)
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_path.name)

    return template.render(
        title="Defense Risk Narratives: Exploratory Clustering Report",
        subtitle="SEC 10-K Item 1A corpus, comparing 2018-2021 against 2022-2025",
        plotly_js=get_plotlyjs(),
        summary=artifacts.summary_metrics,
        figures=artifacts.figures,
        cluster_cards=cluster_cards,
        all_cluster_cards=all_cluster_cards,
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
    output_pdf = Path(args.output_pdf) if args.output_pdf else None

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
    display_df = build_display_sample(sampled_df, args.scatter_display_max_points, args.random_state)

    top_term_map = top_terms_by_cluster(sampled_df, args.top_terms)
    representative_df = representative_examples(sampled_df, embeddings, args.examples_per_cluster)
    cluster_summary_df = build_cluster_summary(sampled_df, top_term_map)
    examples_lookup = build_examples_lookup(sampled_df, representative_df, max_examples=args.examples_per_cluster)
    focus_cluster_df = select_focus_clusters(cluster_summary_df, args.top_clusters)
    cluster_cards = build_cluster_cards(focus_cluster_df, examples_lookup)
    all_cluster_df = (
        cluster_summary_df.sort_values(
            ["eligible_sector_shift", "sector_signal_score", "abs_delta", "cluster_size"],
            ascending=[False, False, False, False],
        )
        .copy()
    )
    all_cluster_cards = build_cluster_cards(all_cluster_df, examples_lookup)

    sampled_df.drop(columns=["text_preview"]).to_csv(artifacts_dir / "sampled_cluster_rows.csv", index=False)
    cluster_summary_df.to_csv(artifacts_dir / "cluster_summary.csv", index=False)
    representative_df.to_csv(artifacts_dir / "representative_examples.csv", index=False)

    figure_objects = {
        "corpus_overview": corpus_overview_figure(full_df, template_name),
        "sample_mix": sample_mix_figure(sampled_df, template_name),
        "umap_period": umap_period_figure(display_df, template_name),
        "umap_cluster": umap_cluster_figure(display_df, template_name),
        "cluster_period": cluster_period_figure(cluster_summary_df, args.top_clusters, template_name),
        "cluster_layer": cluster_layer_figure(cluster_summary_df, args.top_clusters, template_name),
        "cluster_year_heatmap": cluster_year_heatmap(sampled_df, cluster_summary_df, args.top_clusters, template_name),
    }
    figures = {
        key: render_plot(fig)
        for key, fig in figure_objects.items()
    }

    cluster_count = int((cluster_summary_df["cluster"] != -1).sum())
    noise_share = float((sampled_df["cluster"] == -1).mean())
    summary_metrics = {
        "full_rows": f"{len(full_df):,}",
        "sample_rows": f"{len(sampled_df):,}",
        "display_rows": f"{len(display_df):,}",
        "companies": f"{full_df['ticker'].nunique():,}",
        "prime_rows": f"{(full_df['company_layer'] == 'prime').sum():,}",
        "supplier_rows": f"{(full_df['company_layer'] == 'supplier').sum():,}",
        "clusters_found": f"{cluster_count}",
        "noise_share": f"{noise_share:.1%}",
        "model_name": args.model_name,
    }

    report_artifacts = ReportArtifacts(
        sampled_df=sampled_df,
        display_df=display_df,
        cluster_summary_df=cluster_summary_df,
        representative_df=representative_df,
        figures=figures,
        summary_metrics=summary_metrics,
    )

    html = render_report(args, full_df, report_artifacts, cluster_cards, all_cluster_cards)
    output_html.write_text(html, encoding="utf-8")

    if output_pdf:
        figure_dir = artifacts_dir / "pdf_figures"
        figure_paths: Dict[str, Path] = {}
        for key, fig in figure_objects.items():
            figure_path = figure_dir / f"{key}.png"
            export_figure_png(fig, figure_path)
            figure_paths[key] = figure_path
        build_pdf_report(
            output_pdf=output_pdf,
            title=args.pdf_title,
            subtitle="Static export derived from the exploratory clustering run",
            summary_metrics=summary_metrics,
            figure_paths=figure_paths,
            cluster_cards=cluster_cards,
        )

    metadata = {
        "dataset": args.dataset,
        "output_html": str(output_html),
        "output_pdf": str(output_pdf) if output_pdf else "",
        "artifacts_dir": str(artifacts_dir),
        "model_name": args.model_name,
        "sample_per_period": args.sample_per_period,
        "scatter_display_max_points": args.scatter_display_max_points,
        "cluster_min_size": args.cluster_min_size,
        "umap_neighbors": args.umap_neighbors,
        "summary_metrics": summary_metrics,
    }
    (artifacts_dir / "report_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Rendered HTML report to: {output_html}")
    if output_pdf:
        print(f"Rendered PDF report to: {output_pdf}")
    print(f"Artifacts written to: {artifacts_dir}")


if __name__ == "__main__":
    main()
