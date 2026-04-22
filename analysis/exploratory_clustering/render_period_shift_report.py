#!/usr/bin/env python3
"""
Render an exploratory report with shared embeddings and separate pre/post
cluster discovery.

The workflow is:
1. sample once from each period
2. embed all sampled rows once with the same model
3. cluster pre-2022 and post-2022 rows separately
4. summarize each cluster with terms, examples, company breadth, and filing breadth
5. match post clusters back to the pre system approximately
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

import hdbscan
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import umap
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plotly.offline import get_plotlyjs
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances


PRE_PERIOD = "pre_2022"
POST_PERIOD = "post_2022"

PERIOD_META = {
    PRE_PERIOD: {"prefix": "PRE", "title": "Pre-2022", "window": "2018-2021"},
    POST_PERIOD: {"prefix": "POST", "title": "Post-2022", "window": "2022-2025"},
}

MATCH_TYPE_LABELS = {
    "new_post_only": "New post-only",
    "split/refined": "Split/refined",
    "merged": "Merged",
    "persistent": "Persistent",
    "approximate_overlap": "Approximate overlap",
}

MATCH_TYPE_PRIORITY = {
    "new_post_only": 0,
    "split/refined": 1,
    "merged": 2,
    "persistent": 3,
    "approximate_overlap": 4,
}

FOCUS_MIN_CLUSTER_SIZE = 150
FOCUS_MIN_TICKER_COUNT = 10
FOCUS_MIN_FILING_COUNT = 10
FOCUS_MAX_TOP_TICKER_SHARE = 0.35


@dataclass
class PeriodDiscovery:
    period: str
    clustered_df: pd.DataFrame
    summary_df: pd.DataFrame
    representative_df: pd.DataFrame
    display_df: pd.DataFrame
    centroids: Dict[int, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a period-shift exploratory report for the SEC defense corpus."
    )
    parser.add_argument(
        "--dataset",
        default="data/final/sec_defense_risk_dataset.csv",
        help="Path to the canonical final dataset.",
    )
    parser.add_argument(
        "--output-html",
        default="analysis/exploratory_clustering/output/period_shift_report.html",
        help="Path to the rendered HTML report.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="analysis/exploratory_clustering/output",
        help="Directory for CSV and JSON exports.",
    )
    parser.add_argument(
        "--template",
        default="analysis/exploratory_clustering/period_shift_template.html.j2",
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
        help="Maximum rows to sample per period bucket.",
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
        help="Minimum HDBSCAN cluster size within each period subset.",
    )
    parser.add_argument(
        "--top-clusters",
        type=int,
        default=10,
        help="How many post clusters to highlight in the report.",
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
        help="How many representative examples to keep per cluster.",
    )
    parser.add_argument(
        "--scatter-display-max-points",
        type=int,
        default=4000,
        help="Maximum number of points to display in browser scatter plots.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.68,
        help="Cosine threshold for a strong pre/post cluster match.",
    )
    parser.add_argument(
        "--new-cluster-threshold",
        type=float,
        default=0.58,
        help="Cosine threshold below which a post cluster is treated as new post-only.",
    )
    parser.add_argument(
        "--focus-min-filing-count",
        type=int,
        default=FOCUS_MIN_FILING_COUNT,
        help="Minimum filing count for a cluster to be treated as a broad theme.",
    )
    parser.add_argument(
        "--output-pdf",
        default="",
        help="Optional path for a static PDF export.",
    )
    parser.add_argument(
        "--pdf-title",
        default="Defense Risk Narratives: Period Shift Report",
        help="Title used in the optional PDF export.",
    )
    return parser.parse_args()


def build_plotly_template() -> str:
    template_name = "defense_period_shift"
    if template_name in pio.templates:
        return template_name

    pio.templates[template_name] = go.layout.Template(
        layout=go.Layout(
            font=dict(
                family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
                color="#1a2a35",
                size=13,
            ),
            title=dict(font=dict(family="Source Serif 4, Georgia, serif", size=20, color="#1a2a35")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,253,249,0.55)",
            colorway=["#0f4c5c", "#c56b3c", "#5a8a3e", "#9d4e5f", "#3e6c8f", "#1b998b"],
            margin=dict(l=40, r=24, t=56, b=36),
            xaxis=dict(showgrid=True, gridcolor="rgba(221,209,190,0.5)", linecolor="#d6c9b7", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(221,209,190,0.5)", linecolor="#d6c9b7", zeroline=False),
            legend=dict(bgcolor="rgba(255,253,248,0.88)", bordercolor="rgba(221,209,190,0.6)", borderwidth=1),
        )
    )
    return template_name


def empty_figure(message: str, template_name: str, height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=15, color="#566572"),
    )
    fig.update_layout(
        template=template_name,
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=24, r=24, t=48, b=24),
    )
    return fig


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["comparison_window"].isin(["pre_2018_2021", "post_2022_2025"])].copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""].copy()
    return df


def sample_corpus(df: pd.DataFrame, sample_per_period: int, random_state: int) -> pd.DataFrame:
    samples = []
    for _, part in df.groupby("period_bucket", sort=True):
        samples.append(part.sample(min(len(part), sample_per_period), random_state=random_state))
    sampled_df = pd.concat(samples, ignore_index=True)
    sampled_df["sampled_index"] = np.arange(len(sampled_df))
    return sampled_df


def build_display_sample(df: pd.DataFrame, max_points: int, random_state: int, group_col: str) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.copy()
    group_count = max(df[group_col].nunique(), 1)
    per_group = max(1, max_points // group_count)
    samples = []
    for _, part in df.groupby(group_col, group_keys=False):
        samples.append(part.sample(min(len(part), per_group), random_state=random_state))
    display_df = pd.concat(samples, ignore_index=True)
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


def safe_umap_neighbors(requested: int, n_rows: int) -> int:
    if n_rows <= 2:
        return 2
    return max(2, min(requested, n_rows - 1))


def project_embeddings(embeddings: np.ndarray, random_state: int, umap_neighbors: int, min_dist: float) -> np.ndarray:
    if len(embeddings) == 0:
        return np.empty((0, 2))
    if len(embeddings) == 1:
        return np.zeros((1, 2))
    if len(embeddings) == 2:
        return np.asarray([[0.0, 0.0], [1.0, 0.0]])

    reducer = umap.UMAP(
        n_neighbors=safe_umap_neighbors(umap_neighbors, len(embeddings)),
        n_components=2,
        metric="cosine",
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def cluster_embeddings(
    embeddings: np.ndarray,
    random_state: int,
    umap_neighbors: int,
    cluster_min_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    coords_2d = project_embeddings(
        embeddings,
        random_state=random_state,
        umap_neighbors=umap_neighbors,
        min_dist=0.08,
    )
    if len(embeddings) < 5:
        return np.full(len(embeddings), -1, dtype=int), coords_2d

    if len(embeddings) <= 18:
        cluster_space = embeddings
    else:
        reducer = umap.UMAP(
            n_neighbors=safe_umap_neighbors(umap_neighbors, len(embeddings)),
            n_components=min(15, max(2, len(embeddings) - 2)),
            metric="cosine",
            min_dist=0.0,
            random_state=random_state,
        )
        cluster_space = reducer.fit_transform(embeddings)

    effective_cluster_min_size = max(2, min(cluster_min_size, len(embeddings)))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=effective_cluster_min_size,
        min_samples=max(2, min(10, effective_cluster_min_size // 3 or 2)),
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(cluster_space)
    return labels, coords_2d


def period_cluster_label(period: str, cluster_id: int) -> str:
    prefix = PERIOD_META[period]["prefix"]
    if cluster_id == -1:
        return f"{prefix}_Noise"
    return f"{prefix}_C{cluster_id:02d}"


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm


def split_top_terms(value: str) -> List[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [term.strip() for term in value.split(",") if term.strip()]


def parse_top_ticker_n(top_tickers: str) -> int:
    match = re.search(r"\((\d+)\)", str(top_tickers))
    return int(match.group(1)) if match else 0


def top_terms_by_cluster(df: pd.DataFrame, cluster_col: str, top_terms: int) -> Dict[int, str]:
    if df.empty:
        return {}

    min_df = 5 if len(df) >= 200 else 2 if len(df) >= 40 else 1
    max_df = 0.75 if len(df) >= 20 else 1.0
    vectorizer = TfidfVectorizer(
        max_features=20000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
    )

    try:
        matrix = vectorizer.fit_transform(df["text"])
    except ValueError:
        return {-1: "heterogeneous / noise"}

    vocab = np.array(vectorizer.get_feature_names_out())
    summaries: Dict[int, str] = {}
    for cluster_id in sorted(df[cluster_col].unique()):
        if cluster_id == -1:
            summaries[cluster_id] = "heterogeneous / noise"
            continue
        cluster_idx = np.where(df[cluster_col].to_numpy() == cluster_id)[0]
        mean_scores = np.asarray(matrix[cluster_idx].mean(axis=0)).ravel()
        top_idx = mean_scores.argsort()[::-1][:top_terms]
        summaries[cluster_id] = ", ".join(vocab[top_idx])
    return summaries


def representative_examples(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    cluster_col: str,
    cluster_label_col: str,
    examples_per_cluster: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    labels = df[cluster_col].to_numpy()
    for cluster_id in sorted(df[cluster_col].unique()):
        if cluster_id == -1:
            continue
        idx = np.where(labels == cluster_id)[0]
        cluster_vectors = embeddings[idx]
        centroid = normalize_vector(cluster_vectors.mean(axis=0))
        distances = cosine_distances(cluster_vectors, centroid.reshape(1, -1)).ravel()
        for rank, local_idx in enumerate(np.argsort(distances)[:examples_per_cluster], start=1):
            row = df.iloc[idx[local_idx]]
            rows.append(
                {
                    "period_bucket": row["period_bucket"],
                    "cluster": int(cluster_id),
                    "cluster_label": row[cluster_label_col],
                    "rank": rank,
                    "annotation_id": row["annotation_id"],
                    "ticker": row["ticker"],
                    "company_layer": row["company_layer"],
                    "filing_id": row["filing_id"],
                    "filing_year": int(row["filing_year"]),
                    "text": row["text"],
                }
            )
    return pd.DataFrame(rows)


def compute_cluster_centroids(df: pd.DataFrame, embeddings: np.ndarray, cluster_col: str) -> Dict[int, np.ndarray]:
    centroids: Dict[int, np.ndarray] = {}
    labels = df[cluster_col].to_numpy()
    for cluster_id in sorted(df[cluster_col].unique()):
        if cluster_id == -1:
            continue
        idx = np.where(labels == cluster_id)[0]
        centroids[int(cluster_id)] = normalize_vector(embeddings[idx].mean(axis=0))
    return centroids


def build_period_cluster_summary(
    clustered_df: pd.DataFrame,
    period: str,
    top_term_map: Dict[int, str],
    focus_min_filing_count: int,
) -> pd.DataFrame:
    counts = (
        clustered_df.groupby(["period_cluster", "period_cluster_label"])
        .size()
        .reset_index(name="cluster_size")
    )
    counts["period_share"] = counts["cluster_size"] / counts["cluster_size"].sum()

    layer_counts = (
        clustered_df.groupby(["period_cluster", "period_cluster_label", "company_layer"])
        .size()
        .reset_index(name="n")
    )
    layer_totals = layer_counts.groupby(["period_cluster", "period_cluster_label"])["n"].transform("sum")
    layer_counts["layer_share"] = np.where(layer_totals > 0, layer_counts["n"] / layer_totals, 0.0)
    layer_pivot = (
        layer_counts.pivot(
            index=["period_cluster", "period_cluster_label"],
            columns="company_layer",
            values="layer_share",
        )
        .fillna(0.0)
        .reset_index()
    )

    ticker_coverage = (
        clustered_df.groupby(["period_cluster", "period_cluster_label"])["ticker"]
        .nunique()
        .reset_index(name="ticker_count")
    )
    filing_coverage = (
        clustered_df.groupby(["period_cluster", "period_cluster_label"])["filing_id"]
        .nunique()
        .reset_index(name="filing_count")
    )
    top_tickers = (
        clustered_df.groupby(["period_cluster", "period_cluster_label", "ticker"])
        .size()
        .reset_index(name="n")
        .sort_values(["period_cluster", "n", "ticker"], ascending=[True, False, True])
        .groupby(["period_cluster", "period_cluster_label"])
        .head(3)
    )
    top_ticker_strings = (
        top_tickers.groupby(["period_cluster", "period_cluster_label"])
        .apply(lambda part: ", ".join(f"{row.ticker} ({int(row.n)})" for _, row in part.iterrows()))
        .reset_index(name="top_tickers")
    )

    summary = counts.merge(layer_pivot, on=["period_cluster", "period_cluster_label"], how="left")
    summary = summary.merge(ticker_coverage, on=["period_cluster", "period_cluster_label"], how="left")
    summary = summary.merge(filing_coverage, on=["period_cluster", "period_cluster_label"], how="left")
    summary = summary.merge(top_ticker_strings, on=["period_cluster", "period_cluster_label"], how="left")
    summary["period_bucket"] = period
    summary["period_title"] = PERIOD_META[period]["title"]
    summary["top_terms"] = summary["period_cluster"].map(top_term_map)
    summary["top_ticker_n"] = summary["top_tickers"].map(parse_top_ticker_n)
    summary["top_ticker_share"] = np.where(
        summary["cluster_size"] > 0,
        summary["top_ticker_n"] / summary["cluster_size"],
        np.nan,
    )
    summary["eligible_period_theme"] = (
        (summary["period_cluster"] != -1)
        & (summary["cluster_size"] >= FOCUS_MIN_CLUSTER_SIZE)
        & (summary["ticker_count"] >= FOCUS_MIN_TICKER_COUNT)
        & (summary["filing_count"] >= focus_min_filing_count)
        & (summary["top_ticker_share"] <= FOCUS_MAX_TOP_TICKER_SHARE)
    )
    summary["exclusion_reason"] = ""
    summary.loc[summary["period_cluster"] == -1, "exclusion_reason"] = "noise"
    summary.loc[
        (summary["period_cluster"] != -1) & (summary["cluster_size"] < FOCUS_MIN_CLUSTER_SIZE),
        "exclusion_reason",
    ] = "too_small"
    summary.loc[
        (summary["period_cluster"] != -1) & (summary["ticker_count"] < FOCUS_MIN_TICKER_COUNT),
        "exclusion_reason",
    ] = "too_few_tickers"
    summary.loc[
        (summary["period_cluster"] != -1) & (summary["filing_count"] < focus_min_filing_count),
        "exclusion_reason",
    ] = "too_few_filings"
    summary.loc[
        (summary["period_cluster"] != -1) & (summary["top_ticker_share"] > FOCUS_MAX_TOP_TICKER_SHARE),
        "exclusion_reason",
    ] = "too_concentrated"
    summary.loc[summary["eligible_period_theme"], "exclusion_reason"] = ""
    return summary.sort_values(
        ["eligible_period_theme", "period_share", "cluster_size"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def discover_period_clusters(
    sampled_df: pd.DataFrame,
    embeddings: np.ndarray,
    period: str,
    args: argparse.Namespace,
) -> PeriodDiscovery:
    mask = sampled_df["period_bucket"].to_numpy() == period
    period_df = sampled_df.loc[mask].copy().reset_index(drop=True)
    period_embeddings = embeddings[mask]
    cluster_ids, coords = cluster_embeddings(
        period_embeddings,
        random_state=args.random_state,
        umap_neighbors=args.umap_neighbors,
        cluster_min_size=args.cluster_min_size,
    )

    period_df["period_cluster"] = cluster_ids
    period_df["period_cluster_label"] = [period_cluster_label(period, cluster_id) for cluster_id in cluster_ids]
    period_df["period_umap_x"] = coords[:, 0]
    period_df["period_umap_y"] = coords[:, 1]

    top_term_map = top_terms_by_cluster(period_df, cluster_col="period_cluster", top_terms=args.top_terms)
    representative_df = representative_examples(
        period_df,
        period_embeddings,
        cluster_col="period_cluster",
        cluster_label_col="period_cluster_label",
        examples_per_cluster=args.examples_per_cluster,
    )
    summary_df = build_period_cluster_summary(
        period_df,
        period=period,
        top_term_map=top_term_map,
        focus_min_filing_count=args.focus_min_filing_count,
    )
    display_df = build_display_sample(
        period_df,
        max_points=max(250, args.scatter_display_max_points // 2),
        random_state=args.random_state,
        group_col="period_cluster_label",
    )
    centroids = compute_cluster_centroids(period_df, period_embeddings, cluster_col="period_cluster")

    return PeriodDiscovery(
        period=period,
        clustered_df=period_df,
        summary_df=summary_df,
        representative_df=representative_df,
        display_df=display_df,
        centroids=centroids,
    )


def shared_terms_for_pair(pre_terms: str, post_terms: str) -> List[str]:
    return sorted(set(split_top_terms(pre_terms)).intersection(split_top_terms(post_terms)))


def build_cluster_matches(
    pre_summary_df: pd.DataFrame,
    post_summary_df: pd.DataFrame,
    pre_centroids: Dict[int, np.ndarray],
    post_centroids: Dict[int, np.ndarray],
    match_threshold: float,
    new_cluster_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pre_clusters = pre_summary_df[pre_summary_df["period_cluster"] != -1].copy()
    post_clusters = post_summary_df[post_summary_df["period_cluster"] != -1].copy()
    if pre_clusters.empty or post_clusters.empty:
        return pd.DataFrame(), pd.DataFrame()

    pairwise_rows: List[dict] = []
    for _, post_row in post_clusters.iterrows():
        post_cluster = int(post_row["period_cluster"])
        for _, pre_row in pre_clusters.iterrows():
            pre_cluster = int(pre_row["period_cluster"])
            similarity = float(np.clip(np.dot(post_centroids[post_cluster], pre_centroids[pre_cluster]), -1.0, 1.0))
            shared_terms = shared_terms_for_pair(pre_row["top_terms"], post_row["top_terms"])
            pairwise_rows.append(
                {
                    "post_cluster": post_cluster,
                    "post_cluster_label": post_row["period_cluster_label"],
                    "post_cluster_size": int(post_row["cluster_size"]),
                    "post_period_share": float(post_row["period_share"]),
                    "post_top_terms": post_row["top_terms"],
                    "pre_cluster": pre_cluster,
                    "pre_cluster_label": pre_row["period_cluster_label"],
                    "pre_cluster_size": int(pre_row["cluster_size"]),
                    "pre_period_share": float(pre_row["period_share"]),
                    "pre_top_terms": pre_row["top_terms"],
                    "cosine_similarity": similarity,
                    "shared_terms": ", ".join(shared_terms),
                    "shared_term_count": len(shared_terms),
                    "match_score": similarity + 0.015 * len(shared_terms),
                }
            )

    pairwise_df = pd.DataFrame(pairwise_rows)
    if pairwise_df.empty:
        return pairwise_df, pd.DataFrame()

    post_sorted = pairwise_df.sort_values(
        ["post_cluster", "match_score", "cosine_similarity", "shared_term_count"],
        ascending=[True, False, False, False],
    )
    best_post_matches = post_sorted.groupby("post_cluster", as_index=False).first()
    best_pre_matches = (
        pairwise_df.sort_values(
            ["pre_cluster", "match_score", "cosine_similarity", "shared_term_count"],
            ascending=[True, False, False, False],
        )
        .groupby("pre_cluster", as_index=False)
        .first()
    )

    best_pre_lookup = {int(row["pre_cluster"]): int(row["post_cluster"]) for _, row in best_pre_matches.iterrows()}
    strong_post = best_post_matches[best_post_matches["cosine_similarity"] >= match_threshold].copy()
    strong_pre = best_pre_matches[best_pre_matches["cosine_similarity"] >= match_threshold].copy()

    linked_pre_map = strong_pre.groupby("post_cluster")["pre_cluster_label"].apply(list).to_dict()
    sibling_post_map = strong_post.groupby("pre_cluster")["post_cluster_label"].apply(list).to_dict()
    candidate_map = (
        post_sorted.groupby("post_cluster")
        .head(3)
        .groupby("post_cluster")
        .apply(lambda part: "; ".join(f"{row.pre_cluster_label} ({row.cosine_similarity:.2f})" for _, row in part.iterrows()))
        .to_dict()
    )

    rows = []
    for _, row in best_post_matches.iterrows():
        post_cluster = int(row["post_cluster"])
        pre_cluster = int(row["pre_cluster"])
        linked_pre = linked_pre_map.get(post_cluster, [])
        sibling_post = sibling_post_map.get(pre_cluster, [])
        reciprocal = best_pre_lookup.get(pre_cluster) == post_cluster
        similarity = float(row["cosine_similarity"])

        if similarity < new_cluster_threshold:
            match_type = "new_post_only"
        elif len(linked_pre) >= 2:
            match_type = "merged"
        elif len(sibling_post) >= 2:
            match_type = "split/refined"
        elif reciprocal and similarity >= match_threshold:
            match_type = "persistent"
        else:
            match_type = "approximate_overlap"

        rows.append(
            {
                "post_cluster": post_cluster,
                "post_cluster_label": row["post_cluster_label"],
                "post_cluster_size": int(row["post_cluster_size"]),
                "post_period_share": float(row["post_period_share"]),
                "post_top_terms": row["post_top_terms"],
                "pre_cluster": pre_cluster,
                "best_pre_cluster_label": row["pre_cluster_label"],
                "best_pre_cluster_size": int(row["pre_cluster_size"]),
                "best_pre_period_share": float(row["pre_period_share"]),
                "best_pre_top_terms": row["pre_top_terms"],
                "best_pre_similarity": similarity,
                "shared_terms": row["shared_terms"],
                "shared_term_count": int(row["shared_term_count"]),
                "reciprocal_best": bool(reciprocal),
                "linked_pre_clusters": ", ".join(linked_pre),
                "sibling_post_clusters": ", ".join(label for label in sibling_post if label != row["post_cluster_label"]),
                "top_pre_candidates": candidate_map.get(post_cluster, ""),
                "match_type": match_type,
                "match_label": MATCH_TYPE_LABELS[match_type],
                "match_priority": MATCH_TYPE_PRIORITY[match_type],
            }
        )

    match_df = pd.DataFrame(rows).sort_values(
        ["match_priority", "post_period_share", "post_cluster_size", "best_pre_similarity"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return pairwise_df, match_df


def build_examples_lookup(sampled_df: pd.DataFrame, representative_df: pd.DataFrame) -> Dict[str, List[dict]]:
    lookup: Dict[str, List[dict]] = {}
    if not representative_df.empty:
        for cluster_label, part in representative_df.groupby("cluster_label"):
            lookup[str(cluster_label)] = [
                {
                    "ticker": row["ticker"],
                    "company_layer": row["company_layer"],
                    "filing_year": int(row["filing_year"]),
                    "period_bucket": row["period_bucket"],
                    "text": shorten(row["text"], width=320, placeholder="..."),
                }
                for _, row in part.iterrows()
            ]

    for cluster_label, part in sampled_df.groupby("period_cluster_label"):
        key = str(cluster_label)
        if key in lookup:
            continue
        lookup[key] = [
            {
                "ticker": row["ticker"],
                "company_layer": row["company_layer"],
                "filing_year": int(row["filing_year"]),
                "period_bucket": row["period_bucket"],
                "text": shorten(row["text"], width=320, placeholder="..."),
            }
            for _, row in part.head(2).iterrows()
        ]
    return lookup


def select_period_focus_clusters(summary_df: pd.DataFrame, top_clusters: int) -> pd.DataFrame:
    eligible = summary_df[(summary_df["period_cluster"] != -1) & (summary_df["eligible_period_theme"])].copy()
    if len(eligible) < top_clusters:
        eligible = summary_df[summary_df["period_cluster"] != -1].copy()
    return eligible.sort_values(["period_share", "cluster_size"], ascending=[False, False]).head(top_clusters).copy()


def select_highlight_post_clusters(post_catalog_df: pd.DataFrame, top_clusters: int) -> pd.DataFrame:
    eligible = post_catalog_df[(post_catalog_df["period_cluster"] != -1) & (post_catalog_df["eligible_period_theme"])].copy()
    if len(eligible) < top_clusters:
        eligible = post_catalog_df[post_catalog_df["period_cluster"] != -1].copy()
    return eligible.sort_values(
        ["match_priority", "period_share", "cluster_size", "best_pre_similarity"],
        ascending=[True, False, False, False],
    ).head(top_clusters).copy()


def build_match_note(row: pd.Series) -> str:
    if row["match_type"] == "new_post_only":
        return "No sufficiently similar pre-2022 centroid passed the novelty threshold."
    if row["match_type"] == "split/refined":
        return f"Closest to {row['best_pre_cluster_label']}; multiple post clusters now refine that earlier theme."
    if row["match_type"] == "merged":
        linked = row.get("linked_pre_clusters", "")
        if linked:
            return f"This post cluster appears to absorb signal from multiple pre clusters: {linked}."
        return "This post cluster appears to merge several earlier themes."
    if row["match_type"] == "persistent":
        return f"Strong reciprocal match to {row['best_pre_cluster_label']}; this theme looks structurally persistent."
    return f"Closest to {row['best_pre_cluster_label']}, but the match is weak or non-reciprocal."


def build_post_cluster_cards(
    post_rows_df: pd.DataFrame,
    post_examples_lookup: Dict[str, List[dict]],
    pre_examples_lookup: Dict[str, List[dict]],
) -> List[dict]:
    cards: List[dict] = []
    for _, row in post_rows_df.iterrows():
        best_pre_label = row.get("best_pre_cluster_label", "")
        cards.append(
            {
                "cluster_label": row["period_cluster_label"],
                "cluster_size": int(row["cluster_size"]),
                "period_share": float(row["period_share"]),
                "ticker_count": int(row["ticker_count"]),
                "filing_count": int(row["filing_count"]),
                "prime_share": float(row.get("prime", 0.0)),
                "supplier_share": float(row.get("supplier", 0.0)),
                "top_terms": row["top_terms"],
                "top_tickers": row.get("top_tickers", ""),
                "match_type": row["match_type"],
                "match_label": row["match_label"],
                "match_note": build_match_note(row),
                "best_pre_cluster_label": best_pre_label,
                "best_pre_similarity": float(row.get("best_pre_similarity", np.nan)),
                "best_pre_top_terms": row.get("best_pre_top_terms", ""),
                "shared_terms": row.get("shared_terms", ""),
                "top_pre_candidates": row.get("top_pre_candidates", ""),
                "examples": post_examples_lookup.get(str(row["period_cluster_label"]), []),
                "matched_examples": pre_examples_lookup.get(str(best_pre_label), []),
            }
        )
    return cards


def corpus_overview_figure(df: pd.DataFrame, template_name: str) -> go.Figure:
    year_layer = df.groupby(["filing_year", "company_layer"]).size().reset_index(name="n").sort_values("filing_year")
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
    fig.update_layout(height=420)
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


def shared_umap_period_figure(display_df: pd.DataFrame, template_name: str) -> go.Figure:
    if display_df.empty:
        return empty_figure("No sampled rows were available for the shared embedding projection.", template_name, height=540)
    fig = px.scatter(
        display_df,
        x="global_umap_x",
        y="global_umap_y",
        color="period_bucket",
        symbol="company_layer",
        render_mode="webgl",
        template=template_name,
        title="Shared embedding space, colored by period",
        hover_data={
            "annotation_id": True,
            "ticker": True,
            "company_layer": True,
            "filing_year": True,
            "period_cluster_label": True,
            "global_umap_x": False,
            "global_umap_y": False,
        },
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(height=620)
    return fig


def period_cluster_space_figure(display_df: pd.DataFrame, period: str, template_name: str) -> go.Figure:
    if display_df.empty:
        return empty_figure(f"No sampled rows were available for {PERIOD_META[period]['title']} clustering.", template_name, height=540)
    fig = px.scatter(
        display_df,
        x="period_umap_x",
        y="period_umap_y",
        color="period_cluster_label",
        render_mode="webgl",
        template=template_name,
        title=f"{PERIOD_META[period]['title']} cluster discovery",
        hover_data={
            "annotation_id": True,
            "ticker": True,
            "company_layer": True,
            "filing_year": True,
            "period_cluster_label": True,
            "period_umap_x": False,
            "period_umap_y": False,
        },
        opacity=0.76,
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(height=620)
    return fig


def period_cluster_share_figure(summary_df: pd.DataFrame, period: str, top_clusters: int, template_name: str) -> go.Figure:
    plot_df = select_period_focus_clusters(summary_df, top_clusters)
    if plot_df.empty:
        return empty_figure(f"No non-noise {PERIOD_META[period]['title'].lower()} clusters to summarize.", template_name)
    plot_df = plot_df.sort_values("period_share", ascending=True)
    fig = px.bar(
        plot_df,
        x="period_share",
        y="period_cluster_label",
        orientation="h",
        color="eligible_period_theme",
        template=template_name,
        title=f"{PERIOD_META[period]['title']} cluster shares inside their own period",
        labels={
            "period_share": "Share of sampled rows within period",
            "period_cluster_label": "Cluster",
            "eligible_period_theme": "Broad theme",
        },
        hover_data={
            "cluster_size": True,
            "ticker_count": True,
            "filing_count": True,
            "top_terms": True,
            "period_share": ":.1%",
        },
        color_discrete_map={True: "#0f4c5c", False: "#c56b3c"},
    )
    fig.update_layout(height=440, showlegend=False)
    return fig


def match_heatmap_figure(
    pairwise_df: pd.DataFrame,
    pre_summary_df: pd.DataFrame,
    post_summary_df: pd.DataFrame,
    top_clusters: int,
    template_name: str,
) -> go.Figure:
    if pairwise_df.empty:
        return empty_figure("No pre/post centroid pairs were available for matching.", template_name)

    top_pre = select_period_focus_clusters(pre_summary_df, top_clusters)
    top_post = select_period_focus_clusters(post_summary_df, top_clusters)
    if top_pre.empty or top_post.empty:
        return empty_figure("Not enough non-noise clusters to draw the similarity heatmap.", template_name)

    pre_labels = top_pre["period_cluster_label"].tolist()
    post_labels = top_post["period_cluster_label"].tolist()
    plot_df = pairwise_df[
        pairwise_df["pre_cluster_label"].isin(pre_labels) & pairwise_df["post_cluster_label"].isin(post_labels)
    ].copy()
    if plot_df.empty:
        return empty_figure("No similarity scores were available for the highlighted clusters.", template_name)

    matrix = (
        plot_df.pivot(index="post_cluster_label", columns="pre_cluster_label", values="cosine_similarity")
        .reindex(index=post_labels[::-1], columns=pre_labels)
        .fillna(0.0)
    )
    fig = px.imshow(
        matrix,
        color_continuous_scale=["#fff8eb", "#d7e7ea", "#7aa6b8", "#0f4c5c"],
        aspect="auto",
        template=template_name,
        title="Approximate pre/post cluster matching via centroid cosine similarity",
        labels=dict(x="Pre cluster", y="Post cluster", color="Cosine"),
        zmin=0.0,
        zmax=1.0,
    )
    fig.update_layout(height=500, coloraxis_colorbar=dict(title="Cosine"))
    return fig


def post_match_status_figure(post_catalog_df: pd.DataFrame, top_clusters: int, template_name: str) -> go.Figure:
    plot_df = select_highlight_post_clusters(post_catalog_df, top_clusters)
    if plot_df.empty:
        return empty_figure("No post-2022 clusters were available for matching.", template_name)
    plot_df = plot_df.sort_values(["match_priority", "period_share"], ascending=[False, True])
    fig = px.bar(
        plot_df,
        x="period_share",
        y="period_cluster_label",
        orientation="h",
        color="match_type",
        template=template_name,
        title="How the highlighted post clusters map back to the pre system",
        labels={"period_share": "Share of sampled post rows", "period_cluster_label": "Post cluster", "match_type": "Match type"},
        hover_data={
            "best_pre_cluster_label": True,
            "best_pre_similarity": ":.2f",
            "shared_terms": True,
            "top_terms": True,
        },
        color_discrete_map={
            "new_post_only": "#0f4c5c",
            "split/refined": "#1b998b",
            "merged": "#c56b3c",
            "persistent": "#5a8a3e",
            "approximate_overlap": "#8a8f98",
        },
    )
    fig.update_layout(height=440)
    return fig


def render_plot(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})


def export_figure_png(fig: go.Figure, path: Path, width: int = 1400, height: int = 860, scale: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), format="png", width=width, height=height, scale=scale)


def build_pdf_report(output_pdf: Path, title: str, summary_metrics: Dict[str, str], figure_paths: Dict[str, Path]) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError("PDF export requires reportlab. Install it with `pip install reportlab kaleido`.") from exc

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output_pdf), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=42, bottomMargin=36)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26)
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=10, leading=14, textColor=colors.HexColor("#566572"))

    story = [Paragraph(title, title_style), Spacer(1, 0.1 * inch)]
    metric_rows = [
        ["Full corpus rows", summary_metrics["full_rows"], "Sample rows", summary_metrics["sample_rows"]],
        ["Pre clusters", summary_metrics["pre_clusters"], "Post clusters", summary_metrics["post_clusters"]],
        ["Matched post", summary_metrics["matched_post_clusters"], "New post-only", summary_metrics["new_post_clusters"]],
        ["Noise share", summary_metrics["noise_share"], "Embedding model", summary_metrics["model_name"]],
    ]
    table = Table(metric_rows, colWidths=[1.5 * inch, 1.0 * inch, 1.7 * inch, 2.3 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f3eb")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d8ccb8")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d8ccb8")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend([table, Spacer(1, 0.2 * inch), Paragraph("Static summary export from the period-shift report.", body_style), Spacer(1, 0.16 * inch)])
    for key in ["corpus_overview", "sample_mix", "shared_umap_period", "pre_cluster_space", "post_cluster_space", "match_heatmap", "post_match_status"]:
        if key in figure_paths:
            story.extend([Image(str(figure_paths[key]), width=6.8 * inch, height=4.1 * inch), Spacer(1, 0.16 * inch)])
    doc.build(story)


def render_report(
    args: argparse.Namespace,
    figures: Dict[str, str],
    summary_metrics: Dict[str, str],
    highlighted_post_cards: List[dict],
    all_post_cards: List[dict],
    match_rows: List[dict],
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
        title="Defense Risk Narratives: Period Shift Report",
        subtitle="Shared embeddings, separate pre/post cluster discovery, and approximate pre/post theme matching",
        plotly_js=get_plotlyjs(),
        summary=summary_metrics,
        figures=figures,
        post_cluster_cards=highlighted_post_cards,
        all_post_cluster_cards=all_post_cards,
        match_rows=match_rows,
        model_name=args.model_name,
        sample_per_period=args.sample_per_period,
        match_threshold=args.match_threshold,
        new_cluster_threshold=args.new_cluster_threshold,
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
    global_coords = project_embeddings(embeddings, args.random_state, args.umap_neighbors, min_dist=0.08)
    sampled_df["global_umap_x"] = global_coords[:, 0]
    sampled_df["global_umap_y"] = global_coords[:, 1]

    pre_discovery = discover_period_clusters(sampled_df, embeddings, PRE_PERIOD, args)
    post_discovery = discover_period_clusters(sampled_df, embeddings, POST_PERIOD, args)

    sampled_clustered_df = pd.concat([pre_discovery.clustered_df, post_discovery.clustered_df], ignore_index=True).sort_values("sampled_index").reset_index(drop=True)
    representative_df = pd.concat([pre_discovery.representative_df, post_discovery.representative_df], ignore_index=True)
    period_cluster_summary_df = pd.concat([pre_discovery.summary_df, post_discovery.summary_df], ignore_index=True)

    pairwise_df, match_df = build_cluster_matches(
        pre_summary_df=pre_discovery.summary_df,
        post_summary_df=post_discovery.summary_df,
        pre_centroids=pre_discovery.centroids,
        post_centroids=post_discovery.centroids,
        match_threshold=args.match_threshold,
        new_cluster_threshold=args.new_cluster_threshold,
    )

    post_catalog_df = post_discovery.summary_df.merge(match_df, left_on="period_cluster_label", right_on="post_cluster_label", how="left")
    post_catalog_df["match_type"] = post_catalog_df["match_type"].fillna("new_post_only")
    post_catalog_df["match_label"] = post_catalog_df["match_type"].map(MATCH_TYPE_LABELS).fillna("New post-only")
    post_catalog_df["match_priority"] = post_catalog_df["match_type"].map(MATCH_TYPE_PRIORITY).fillna(0).astype(int)

    examples_lookup = build_examples_lookup(sampled_clustered_df, representative_df)
    highlighted_post_df = select_highlight_post_clusters(post_catalog_df, args.top_clusters)
    highlighted_post_cards = build_post_cluster_cards(highlighted_post_df, examples_lookup, examples_lookup)
    all_post_cards = build_post_cluster_cards(
        post_catalog_df.sort_values(["match_priority", "period_share", "cluster_size"], ascending=[True, False, False]),
        examples_lookup,
        examples_lookup,
    )

    global_display_df = build_display_sample(sampled_clustered_df, args.scatter_display_max_points, args.random_state, "period_bucket")
    figure_objects = {
        "corpus_overview": corpus_overview_figure(full_df, template_name),
        "sample_mix": sample_mix_figure(sampled_clustered_df, template_name),
        "shared_umap_period": shared_umap_period_figure(global_display_df, template_name),
        "pre_cluster_space": period_cluster_space_figure(pre_discovery.display_df, PRE_PERIOD, template_name),
        "post_cluster_space": period_cluster_space_figure(post_discovery.display_df, POST_PERIOD, template_name),
        "pre_cluster_share": period_cluster_share_figure(pre_discovery.summary_df, PRE_PERIOD, args.top_clusters, template_name),
        "post_cluster_share": period_cluster_share_figure(post_discovery.summary_df, POST_PERIOD, args.top_clusters, template_name),
        "match_heatmap": match_heatmap_figure(pairwise_df, pre_discovery.summary_df, post_discovery.summary_df, args.top_clusters, template_name),
        "post_match_status": post_match_status_figure(post_catalog_df, args.top_clusters, template_name),
    }
    figures = {key: render_plot(fig) for key, fig in figure_objects.items()}

    summary_metrics = {
        "full_rows": f"{len(full_df):,}",
        "sample_rows": f"{len(sampled_clustered_df):,}",
        "display_rows": f"{len(global_display_df):,}",
        "companies": f"{full_df['ticker'].nunique():,}",
        "pre_clusters": f"{int((pre_discovery.summary_df['period_cluster'] != -1).sum())}",
        "post_clusters": f"{int((post_discovery.summary_df['period_cluster'] != -1).sum())}",
        "matched_post_clusters": f"{int((match_df['best_pre_similarity'] >= args.match_threshold).sum()) if not match_df.empty else 0}",
        "new_post_clusters": f"{int((match_df['match_type'] == 'new_post_only').sum()) if not match_df.empty else 0}",
        "noise_share": f"{float((sampled_clustered_df['period_cluster'] == -1).mean()):.1%}",
        "model_name": args.model_name,
    }

    sampled_clustered_df.to_csv(artifacts_dir / "sampled_cluster_rows.csv", index=False)
    period_cluster_summary_df.to_csv(artifacts_dir / "period_cluster_summary.csv", index=False)
    representative_df.to_csv(artifacts_dir / "representative_examples.csv", index=False)
    pairwise_df.to_csv(artifacts_dir / "pairwise_cluster_similarities.csv", index=False)
    match_df.to_csv(artifacts_dir / "cluster_matches.csv", index=False)

    match_rows = (
        post_catalog_df[
            ["period_cluster_label", "match_label", "best_pre_cluster_label", "best_pre_similarity", "shared_terms", "top_terms"]
        ]
        .sort_values(["best_pre_similarity", "period_cluster_label"], ascending=[False, True], na_position="last")
        .to_dict(orient="records")
    )

    html_report = render_report(
        args,
        figures=figures,
        summary_metrics=summary_metrics,
        highlighted_post_cards=highlighted_post_cards,
        all_post_cards=all_post_cards,
        match_rows=match_rows,
    )
    output_html.write_text(html_report, encoding="utf-8")

    if output_pdf:
        figure_dir = artifacts_dir / "pdf_figures"
        figure_paths: Dict[str, Path] = {}
        for key, fig in figure_objects.items():
            figure_path = figure_dir / f"{key}.png"
            export_figure_png(fig, figure_path)
            figure_paths[key] = figure_path
        build_pdf_report(output_pdf, args.pdf_title, summary_metrics, figure_paths)

    metadata = {
        "dataset": args.dataset,
        "output_html": str(output_html),
        "output_pdf": str(output_pdf) if output_pdf else "",
        "artifacts_dir": str(artifacts_dir),
        "model_name": args.model_name,
        "sample_per_period": args.sample_per_period,
        "cluster_min_size": args.cluster_min_size,
        "match_threshold": args.match_threshold,
        "new_cluster_threshold": args.new_cluster_threshold,
        "summary_metrics": summary_metrics,
    }
    (artifacts_dir / "period_shift_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Rendered HTML report to: {output_html}")
    if output_pdf:
        print(f"Rendered PDF report to: {output_pdf}")
    print(f"Artifacts written to: {artifacts_dir}")


if __name__ == "__main__":
    main()
