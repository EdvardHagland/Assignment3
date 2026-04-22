#!/usr/bin/env python3
"""
Render a Gemini-assisted narrative report for the period-shift clustering output.

This script is a second-stage pass. It reads the CSV artifacts produced by
render_period_shift_report.py, packages the most interesting post-2022 cluster
changes with representative evidence, asks Gemini for structured critical
interpretations, and then asks Gemini for a short executive abstract that goes
at the top of the report.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plotly.offline import get_plotlyjs

from render_period_shift_report import (
    EMERGENT_MAX_TOP_TICKER_SHARE,
    EMERGENT_MIN_CLUSTER_SIZE,
    EMERGENT_MIN_FILING_COUNT,
    EMERGENT_MIN_TICKER_COUNT,
    FOCUS_MAX_TOP_TICKER_SHARE,
    FOCUS_MIN_CLUSTER_SIZE,
    FOCUS_MIN_FILING_COUNT,
    FOCUS_MIN_TICKER_COUNT,
    MATCH_TYPE_LABELS,
    MATCH_TYPE_PRIORITY,
    PERIOD_META,
    build_display_sample,
    corpus_overview_figure,
    match_heatmap_figure,
    period_cluster_share_figure,
    period_cluster_space_figure,
    post_match_status_figure,
    sample_mix_figure,
    shared_umap_period_figure,
)


DEFAULT_INTERESTING_MATCH_TYPES = ["new_post_only", "split/refined", "merged"]
PRE_PERIOD = "pre_2022"
POST_PERIOD = "post_2022"
DEFAULT_DATASET_PATH = Path("data/final/sec_defense_risk_dataset.csv")
EMERGENT_MATCH_TYPE = "new_post_only"
CLUSTER_ANALYSIS_PROMPT_VERSION = 5
ABSTRACT_PROMPT_VERSION = 5
CONTENT_SHIFT_JUDGMENT = "same_cluster_shifted_contents"
EMERGENT_JUDGMENT = "genuinely_new_theme"
CONTINUITY_LABELS = {
    "largely_continuous": "Largely continuous",
    "same_cluster_shifted_contents": "Same cluster, shifted contents",
    "clear_structural_change": "Clear structural change",
    "genuinely_new_theme": "Genuinely new theme",
    "weak_or_unclear_change": "Weak or unclear change",
}
COUNT_GAP_ROLE_LABELS = {
    "genuine_novelty": "Genuine novelty",
    "split_or_refinement": "Split or refinement",
    "denser_or_more_explicit_disclosure": "Denser or more explicit disclosure",
    "little_or_no_count_gap_role": "Little or no count-gap role",
    "unclear": "Unclear",
}
CONTINUITY_READING_PRIORITY = {
    "genuinely_new_theme": 0,
    "clear_structural_change": 1,
    "same_cluster_shifted_contents": 2,
    "weak_or_unclear_change": 3,
    "largely_continuous": 4,
}
COUNT_GAP_ROLE_PRIORITY = {
    "genuine_novelty": 0,
    "split_or_refinement": 1,
    "denser_or_more_explicit_disclosure": 2,
    "unclear": 3,
    "little_or_no_count_gap_role": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Gemini-assisted narrative report from period-shift clustering artifacts."
    )
    parser.add_argument(
        "--sampled-rows",
        default="analysis/exploratory_clustering/output/sampled_cluster_rows.csv",
        help="Path to sampled_cluster_rows.csv from render_period_shift_report.py.",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Optional path to the canonical dataset. Falls back to the dataset path stored in period_shift_metadata.json.",
    )
    parser.add_argument(
        "--period-cluster-summary",
        default="analysis/exploratory_clustering/output/period_cluster_summary.csv",
        help="Path to period_cluster_summary.csv from render_period_shift_report.py.",
    )
    parser.add_argument(
        "--pairwise-similarities",
        default="analysis/exploratory_clustering/output/pairwise_cluster_similarities.csv",
        help="Path to pairwise_cluster_similarities.csv from render_period_shift_report.py.",
    )
    parser.add_argument(
        "--cluster-matches",
        default="analysis/exploratory_clustering/output/cluster_matches.csv",
        help="Path to cluster_matches.csv from render_period_shift_report.py.",
    )
    parser.add_argument(
        "--representative-examples",
        default="analysis/exploratory_clustering/output/representative_examples.csv",
        help="Path to representative_examples.csv from render_period_shift_report.py.",
    )
    parser.add_argument(
        "--metadata",
        default="analysis/exploratory_clustering/output/period_shift_metadata.json",
        help="Path to period_shift_metadata.json from render_period_shift_report.py.",
    )
    parser.add_argument(
        "--sampled-embeddings",
        default="",
        help="Optional path to sampled_embeddings.npz from render_period_shift_report.py. Defaults to the same-run path in metadata or the standard output location.",
    )
    parser.add_argument(
        "--output-html",
        default="analysis/exploratory_clustering/output/period_shift_llm_report.html",
        help="Path to the rendered HTML report.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="analysis/exploratory_clustering/output",
        help="Directory for JSON outputs produced by this script.",
    )
    parser.add_argument(
        "--template",
        default="analysis/exploratory_clustering/period_shift_llm_template.html.j2",
        help="Path to the LLM report template.",
    )
    parser.add_argument(
        "--model-name",
        default="",
        help="Gemini model name. Falls back to GEMINI_MODEL.",
    )
    parser.add_argument(
        "--embedding-model-name",
        default="",
        help="Sentence-transformers model used only if --allow-reembed is explicitly enabled for legacy artifacts.",
    )
    parser.add_argument(
        "--interesting-match-types",
        default=",".join(DEFAULT_INTERESTING_MATCH_TYPES),
        help="Comma-separated post-cluster match types to narrate.",
    )
    parser.add_argument(
        "--emergent-min-cluster-size",
        type=int,
        default=EMERGENT_MIN_CLUSTER_SIZE,
        help="Minimum cluster size for a post-only cluster to qualify as an emergent discovery candidate.",
    )
    parser.add_argument(
        "--emergent-min-ticker-count",
        type=int,
        default=EMERGENT_MIN_TICKER_COUNT,
        help="Minimum ticker breadth for a post-only cluster to qualify as an emergent discovery candidate.",
    )
    parser.add_argument(
        "--emergent-min-filing-count",
        type=int,
        default=EMERGENT_MIN_FILING_COUNT,
        help="Minimum filing breadth for a post-only cluster to qualify as an emergent discovery candidate.",
    )
    parser.add_argument(
        "--emergent-max-top-ticker-share",
        type=float,
        default=EMERGENT_MAX_TOP_TICKER_SHARE,
        help="Maximum top-ticker share for a post-only cluster to qualify as an emergent discovery candidate.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=6,
        help="Maximum number of interesting post clusters to send to Gemini. Use 0 or a negative value to include all eligible changed clusters.",
    )
    parser.add_argument(
        "--persistent-audit-clusters",
        type=int,
        default=2,
        help="How many broad persistent clusters to include as audits for same-cluster content drift. Use 0 or a negative value to include all eligible persistent audit clusters.",
    )
    parser.add_argument(
        "--central-examples",
        type=int,
        default=4,
        help="How many centroid-near post examples to package per cluster.",
    )
    parser.add_argument(
        "--mid-examples",
        type=int,
        default=4,
        help="How many mid-distance post examples to package per cluster.",
    )
    parser.add_argument(
        "--peripheral-examples",
        type=int,
        default=4,
        help="How many moderately peripheral post examples to package per cluster.",
    )
    parser.add_argument(
        "--matched-pre-examples",
        type=int,
        default=3,
        help="How many matched pre-cluster examples to package per cluster.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Gemini generation temperature.",
    )
    parser.add_argument(
        "--abstract-temperature",
        type=float,
        default=0.15,
        help="Gemini temperature for the executive abstract call.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Build the evidence package and HTML shell without calling Gemini.",
    )
    parser.add_argument(
        "--llm-max-concurrency",
        type=int,
        default=8,
        help="Maximum number of cluster-level Gemini requests to run concurrently. The final abstract call remains sequential.",
    )
    parser.add_argument(
        "--llm-request-stagger-seconds",
        type=float,
        default=0.2,
        help="Delay between submitting concurrent cluster-level Gemini requests, to avoid bursting them all at once.",
    )
    parser.add_argument(
        "--allow-reembed",
        action="store_true",
        help="Allow a second local embedding pass only when same-run sampled_embeddings.npz is unavailable. Intended as a legacy fallback.",
    )
    return parser.parse_args()


def build_plotly_template() -> str:
    template_name = "defense_period_shift_llm"
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
            colorway=["#0f4c5c", "#c56b3c", "#5a8a3e", "#1b998b", "#3e6c8f"],
            margin=dict(l=40, r=24, t=56, b=36),
            xaxis=dict(showgrid=True, gridcolor="rgba(221,209,190,0.5)", linecolor="#d6c9b7", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(221,209,190,0.5)", linecolor="#d6c9b7", zeroline=False),
            legend=dict(bgcolor="rgba(255,253,248,0.88)", bordercolor="rgba(221,209,190,0.6)", borderwidth=1),
        )
    )
    return template_name


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_cluster_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(analysis)
    normalized.setdefault("continuity_judgment", "weak_or_unclear_change")
    normalized.setdefault("continuity_note", "This run did not include a more specific continuity judgment.")
    normalized.setdefault("count_gap_role", "unclear")
    normalized.setdefault("count_gap_note", "This run did not include a more specific cluster-count role judgment.")
    normalized.setdefault("layer_reading", "")
    normalized.setdefault("concentration_note", "")
    normalized.setdefault("temporal_reading_post", "")
    normalized.setdefault("intra_cluster_churn_note", "")
    normalized.setdefault("category_blending_note", "")
    normalized.setdefault("_prompt_version", CLUSTER_ANALYSIS_PROMPT_VERSION)
    return normalized


def normalize_abstract(abstract: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(abstract)
    normalized.setdefault("boilerplate_envelope_finding", "")
    normalized.setdefault("layer_asymmetries", [])
    normalized.setdefault("intra_cluster_churn_caveat", "")
    normalized.setdefault("_prompt_version", ABSTRACT_PROMPT_VERSION)
    return normalized


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    sampled_df = pd.read_csv(args.sampled_rows)
    summary_df = pd.read_csv(args.period_cluster_summary)
    matches_df = pd.read_csv(args.cluster_matches)
    pairwise_df = pd.read_csv(args.pairwise_similarities)
    representative_df = pd.read_csv(args.representative_examples)
    metadata = read_json(Path(args.metadata)) if Path(args.metadata).exists() else {}
    return {
        "sampled_df": sampled_df,
        "summary_df": summary_df,
        "matches_df": matches_df,
        "pairwise_df": pairwise_df,
        "representative_df": representative_df,
        "metadata": metadata,
    }


def resolve_gemini_model_name(args: argparse.Namespace) -> str:
    model_name = args.model_name.strip() or os.getenv("GEMINI_MODEL", "").strip()
    if not model_name:
        raise ValueError("No Gemini model configured. Pass --model-name or set GEMINI_MODEL.")
    return model_name


def resolve_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return api_key


def resolve_embedding_model_name(args: argparse.Namespace, metadata: dict[str, Any]) -> str:
    if args.embedding_model_name.strip():
        return args.embedding_model_name.strip()
    model_name = str(metadata.get("model_name", "")).strip()
    if model_name:
        return model_name
    raise ValueError("No embedding model available. Pass --embedding-model-name or supply period_shift_metadata.json.")


def resolve_dataset_path(args: argparse.Namespace, metadata: dict[str, Any]) -> Path:
    if args.dataset.strip():
        return Path(args.dataset)
    dataset_value = str(metadata.get("dataset", "")).strip()
    if dataset_value:
        return Path(dataset_value)
    if DEFAULT_DATASET_PATH.exists():
        return DEFAULT_DATASET_PATH
    raise ValueError("No dataset path available. Pass --dataset or provide period_shift_metadata.json with a dataset entry.")


def resolve_sampled_embeddings_path(args: argparse.Namespace, metadata: dict[str, Any]) -> Path | None:
    if args.sampled_embeddings.strip():
        return Path(args.sampled_embeddings)
    metadata_value = str(metadata.get("sampled_embeddings", "")).strip()
    if metadata_value:
        return Path(metadata_value)
    default_path = Path("analysis/exploratory_clustering/output/sampled_embeddings.npz")
    if default_path.exists():
        return default_path
    return None


def parse_match_types(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    lowered = series.fillna("").astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes", "y"})


def ensure_theme_flags(summary_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    summary = summary_df.copy()
    if "eligible_period_theme" in summary.columns:
        summary["eligible_period_theme"] = coerce_bool_series(summary["eligible_period_theme"])
    else:
        summary["eligible_period_theme"] = (
            (summary["period_cluster"] != -1)
            & (summary["cluster_size"] >= FOCUS_MIN_CLUSTER_SIZE)
            & (summary["ticker_count"] >= FOCUS_MIN_TICKER_COUNT)
            & (summary["filing_count"] >= FOCUS_MIN_FILING_COUNT)
            & (summary["top_ticker_share"] <= FOCUS_MAX_TOP_TICKER_SHARE)
        )

    if "eligible_emergent_theme" in summary.columns:
        summary["eligible_emergent_theme"] = coerce_bool_series(summary["eligible_emergent_theme"])
    else:
        summary["eligible_emergent_theme"] = (
            (summary["period_cluster"] != -1)
            & (summary["cluster_size"] >= args.emergent_min_cluster_size)
            & (summary["ticker_count"] >= args.emergent_min_ticker_count)
            & (summary["filing_count"] >= args.emergent_min_filing_count)
            & (summary["top_ticker_share"] <= args.emergent_max_top_ticker_share)
        )
    return summary


def build_post_selection_frame(
    summary_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    interesting_match_types: list[str],
    max_clusters: int,
    args: argparse.Namespace,
) -> pd.DataFrame:
    post_summary = ensure_theme_flags(summary_df, args)
    post_summary = post_summary[post_summary["period_bucket"] == POST_PERIOD].copy()
    post_selection_df = post_summary.merge(
        matches_df,
        left_on="period_cluster_label",
        right_on="post_cluster_label",
        how="left",
    )
    post_selection_df["match_type"] = post_selection_df["match_type"].fillna("new_post_only")
    post_selection_df["match_label"] = post_selection_df["match_label"].fillna("New post-only")
    post_selection_df["match_priority"] = post_selection_df["match_priority"].fillna(0)
    post_selection_df["eligible_period_theme"] = coerce_bool_series(post_selection_df["eligible_period_theme"])
    post_selection_df["eligible_emergent_theme"] = coerce_bool_series(post_selection_df["eligible_emergent_theme"])
    post_selection_df["eligible_for_narration"] = np.where(
        post_selection_df["match_type"] == EMERGENT_MATCH_TYPE,
        post_selection_df["eligible_emergent_theme"],
        post_selection_df["eligible_period_theme"],
    )
    post_selection_df["narration_filter"] = np.where(
        post_selection_df["match_type"] == EMERGENT_MATCH_TYPE,
        "emergent_discovery",
        "broad_structural",
    )
    post_selection_df["main_candidate"] = (
        (post_selection_df["period_cluster"] != -1)
        & (post_selection_df["eligible_for_narration"].fillna(False))
        & (post_selection_df["match_type"].isin(interesting_match_types))
    )
    post_selection_df["main_candidate_rank"] = np.nan

    main_candidates = post_selection_df[post_selection_df["main_candidate"]].copy()
    main_candidates = main_candidates.sort_values(
        ["match_priority", "period_share", "cluster_size", "best_pre_similarity"],
        ascending=[True, False, False, False],
    ).reset_index()
    if not main_candidates.empty:
        post_selection_df.loc[main_candidates["index"], "main_candidate_rank"] = np.arange(1, len(main_candidates) + 1)

    main_limit = len(main_candidates) if max_clusters <= 0 else max_clusters
    post_selection_df["selected_main"] = post_selection_df["main_candidate_rank"].le(main_limit).fillna(False)
    post_selection_df["persistent_audit_candidate"] = (
        (post_selection_df["period_cluster"] != -1)
        & (post_selection_df["eligible_period_theme"].fillna(False))
        & (post_selection_df["match_type"] == "persistent")
        & (~post_selection_df["selected_main"])
    )
    post_selection_df["persistent_audit_rank"] = np.nan
    persistent_audit_candidates = post_selection_df[post_selection_df["persistent_audit_candidate"]].copy()
    persistent_audit_candidates = persistent_audit_candidates.sort_values(
        ["period_share", "cluster_size", "best_pre_similarity"],
        ascending=[False, False, True],
    ).reset_index()
    if not persistent_audit_candidates.empty:
        post_selection_df.loc[persistent_audit_candidates["index"], "persistent_audit_rank"] = np.arange(1, len(persistent_audit_candidates) + 1)

    persistent_audit_limit = (
        len(persistent_audit_candidates)
        if args.persistent_audit_clusters <= 0
        else args.persistent_audit_clusters
    )
    post_selection_df["selected_persistent_audit"] = post_selection_df["persistent_audit_rank"].le(persistent_audit_limit).fillna(False)
    post_selection_df["selected_for_llm"] = post_selection_df["selected_main"] | post_selection_df["selected_persistent_audit"]
    post_selection_df["selection_reason"] = ""
    post_selection_df.loc[post_selection_df["selected_main"], "selection_reason"] = "main_match_filter"
    post_selection_df.loc[post_selection_df["selected_persistent_audit"], "selection_reason"] = "persistent_audit"

    post_selection_df.loc[
        (post_selection_df["period_cluster"] == -1) & (~post_selection_df["selected_for_llm"]),
        "selection_reason",
    ] = "noise_cluster"
    post_selection_df.loc[
        (~post_selection_df["selected_for_llm"])
        & (post_selection_df["match_type"] == EMERGENT_MATCH_TYPE)
        & (~post_selection_df["eligible_emergent_theme"]),
        "selection_reason",
    ] = "failed_emergent_breadth_screen"
    post_selection_df.loc[
        (~post_selection_df["selected_for_llm"])
        & (post_selection_df["match_type"] != EMERGENT_MATCH_TYPE)
        & (~post_selection_df["eligible_period_theme"]),
        "selection_reason",
    ] = "failed_structural_breadth_screen"
    post_selection_df.loc[
        (~post_selection_df["selected_for_llm"])
        & (post_selection_df["main_candidate"])
        & (max_clusters > 0)
        & (post_selection_df["main_candidate_rank"] > max_clusters),
        "selection_reason",
    ] = "trimmed_by_max_clusters"
    post_selection_df.loc[
        (~post_selection_df["selected_for_llm"])
        & (post_selection_df["persistent_audit_candidate"])
        & (args.persistent_audit_clusters > 0)
        & (post_selection_df["persistent_audit_rank"] > args.persistent_audit_clusters),
        "selection_reason",
    ] = "outside_persistent_audit_limit"
    post_selection_df.loc[
        (~post_selection_df["selected_for_llm"])
        & (post_selection_df["selection_reason"] == "")
        & (~post_selection_df["match_type"].isin(interesting_match_types))
        & (post_selection_df["match_type"] != "persistent"),
        "selection_reason",
    ] = "match_type_not_requested"
    post_selection_df.loc[
        (~post_selection_df["selected_for_llm"])
        & (post_selection_df["selection_reason"] == "")
        & (post_selection_df["match_type"] == "persistent")
        & (~post_selection_df["persistent_audit_candidate"]),
        "selection_reason",
    ] = "persistent_not_in_primary_filter"
    post_selection_df.loc[
        (~post_selection_df["selected_for_llm"])
        & (post_selection_df["selection_reason"] == ""),
        "selection_reason",
    ] = "not_selected"
    return post_selection_df


def select_interesting_post_clusters(
    summary_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    interesting_match_types: list[str],
    max_clusters: int,
    args: argparse.Namespace,
) -> pd.DataFrame:
    post_selection_df = build_post_selection_frame(
        summary_df=summary_df,
        matches_df=matches_df,
        interesting_match_types=interesting_match_types,
        max_clusters=max_clusters,
        args=args,
    )
    selected = post_selection_df[post_selection_df["selected_for_llm"]].copy()
    selected = selected.sort_values(
        ["selected_main", "main_candidate_rank", "selected_persistent_audit", "persistent_audit_rank", "period_share"],
        ascending=[False, True, False, True, False],
    )
    return selected.reset_index(drop=True)


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings)


def load_saved_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    embeddings = np.asarray(payload["embeddings"])
    sampled_index = np.asarray(payload["sampled_index"])
    return embeddings, sampled_index


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm


def compute_cluster_distance_frame(
    sampled_df: pd.DataFrame,
    cluster_labels: list[str],
    embedding_model_name: str,
    embedding_payload: tuple[np.ndarray, np.ndarray] | None,
    allow_reembed: bool,
) -> pd.DataFrame:
    focus_df = sampled_df[sampled_df["period_cluster_label"].isin(cluster_labels)].copy()
    if focus_df.empty:
        return focus_df
    focus_df = focus_df.sort_values("sampled_index").reset_index(drop=True)

    if embedding_payload is not None:
        saved_embeddings, saved_sampled_index = embedding_payload
        embedding_lookup = {int(idx): pos for pos, idx in enumerate(saved_sampled_index.tolist())}
        missing_indices = [int(idx) for idx in focus_df["sampled_index"].tolist() if int(idx) not in embedding_lookup]
        if missing_indices:
            raise ValueError(
                "Saved sampled embeddings are missing rows needed for evidence selection: "
                f"{missing_indices[:5]}"
            )
        embedding_positions = focus_df["sampled_index"].astype(int).map(embedding_lookup).to_numpy()
        embeddings = saved_embeddings[embedding_positions]
    else:
        if not allow_reembed:
            raise ValueError(
                "No same-run sampled embeddings were available for evidence selection. "
                "Rerun render_period_shift_report.py with the latest code or pass --allow-reembed "
                "to use a legacy second embedding pass."
            )
        embeddings = embed_texts(focus_df["text"].astype(str).tolist(), embedding_model_name)

    focus_df["embedding_index"] = np.arange(len(focus_df))
    focus_df["distance_to_centroid"] = np.nan
    focus_df["distance_rank"] = np.nan
    focus_df["distance_percentile"] = np.nan

    for cluster_label, part in focus_df.groupby("period_cluster_label"):
        idx = part["embedding_index"].to_numpy()
        cluster_vectors = embeddings[idx]
        centroid = normalize_vector(cluster_vectors.mean(axis=0))
        distances = 1.0 - np.clip(cluster_vectors @ centroid, -1.0, 1.0)
        order = np.argsort(distances)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))
        percentiles = np.where(
            len(order) > 1,
            ranks / (len(order) - 1),
            0.0,
        )
        focus_df.loc[part.index, "distance_to_centroid"] = distances
        focus_df.loc[part.index, "distance_rank"] = ranks
        focus_df.loc[part.index, "distance_percentile"] = percentiles

    return focus_df


def choose_examples_in_band(
    cluster_df: pd.DataFrame,
    excluded_ids: set[str],
    n: int,
    start_percentile: float,
    end_percentile: float,
) -> pd.DataFrame:
    if cluster_df.empty or n <= 0:
        return cluster_df.head(0)

    candidates = cluster_df[~cluster_df["annotation_id"].isin(excluded_ids)].copy()
    if candidates.empty:
        return candidates

    candidates = candidates.sort_values("distance_percentile").reset_index(drop=True)
    positions = np.linspace(start_percentile, end_percentile, num=min(n, len(candidates)))
    chosen_indices: list[int] = []
    for position in positions:
        idx = int(round(position * (len(candidates) - 1)))
        idx = max(0, min(idx, len(candidates) - 1))
        if idx not in chosen_indices:
            chosen_indices.append(idx)

    while len(chosen_indices) < min(n, len(candidates)):
        fallback = len(candidates) - 1 - len(chosen_indices)
        fallback = max(0, min(fallback, len(candidates) - 1))
        if fallback not in chosen_indices:
            chosen_indices.append(fallback)
        else:
            break

    return candidates.iloc[sorted(chosen_indices)].copy()


def choose_mid_examples(cluster_df: pd.DataFrame, excluded_ids: set[str], n: int) -> pd.DataFrame:
    return choose_examples_in_band(cluster_df, excluded_ids, n, start_percentile=0.28, end_percentile=0.58)


def choose_peripheral_examples(cluster_df: pd.DataFrame, excluded_ids: set[str], n: int) -> pd.DataFrame:
    return choose_examples_in_band(cluster_df, excluded_ids, n, start_percentile=0.68, end_percentile=0.90)


def format_example_rows(rows_df: pd.DataFrame, label_prefix: str) -> list[dict[str, Any]]:
    examples = []
    for idx, (_, row) in enumerate(rows_df.iterrows(), start=1):
        examples.append(
            {
                "example_id": f"{label_prefix}_{idx}",
                "annotation_id": row["annotation_id"],
                "ticker": row["ticker"],
                "company_layer": row["company_layer"],
                "filing_year": int(row["filing_year"]),
                "period_bucket": row["period_bucket"],
                "distance_percentile": float(row.get("distance_percentile", 0.0)),
                "text": str(row["text"]).strip(),
                "text_preview": str(row["text"]).strip()[:420],
            }
        )
    return examples


def build_cluster_evidence_package(
    cluster_row: pd.Series,
    sampled_df: pd.DataFrame,
    representative_df: pd.DataFrame,
    cluster_distance_df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    post_label = str(cluster_row["period_cluster_label"])
    pre_label = str(cluster_row.get("best_pre_cluster_label", ""))

    post_cluster_rows = cluster_distance_df[cluster_distance_df["period_cluster_label"] == post_label].copy()
    pre_cluster_rows = cluster_distance_df[cluster_distance_df["period_cluster_label"] == pre_label].copy()

    central_post_rows = representative_df[representative_df["cluster_label"] == post_label].sort_values("rank").head(args.central_examples).copy()
    central_post_ids = set(central_post_rows["annotation_id"].astype(str))
    mid_post_rows = choose_mid_examples(post_cluster_rows, central_post_ids, args.mid_examples)
    excluded_after_mid = central_post_ids.union(set(mid_post_rows["annotation_id"].astype(str)))
    peripheral_post_rows = choose_peripheral_examples(post_cluster_rows, excluded_after_mid, args.peripheral_examples)

    pre_rows = representative_df[representative_df["cluster_label"] == pre_label].sort_values("rank").head(args.matched_pre_examples).copy()
    if pre_rows.empty and not pre_cluster_rows.empty:
        pre_rows = pre_cluster_rows.sort_values("distance_percentile").head(args.matched_pre_examples).copy()

    central_post_examples = format_example_rows(central_post_rows, f"{post_label}_core")
    mid_post_examples = format_example_rows(mid_post_rows, f"{post_label}_mid")
    peripheral_post_examples = format_example_rows(peripheral_post_rows, f"{post_label}_edge")
    matched_pre_examples = format_example_rows(pre_rows, f"{pre_label or 'pre'}_match")

    post_year_distribution = build_post_year_distribution(
        central_post_examples + mid_post_examples + peripheral_post_examples
    )

    top_ticker_share_value = float(cluster_row.get("top_ticker_share", 0.0)) if pd.notna(cluster_row.get("top_ticker_share", 0.0)) else 0.0
    top_ticker_n_value = int(cluster_row.get("top_ticker_n", 0)) if pd.notna(cluster_row.get("top_ticker_n", 0)) else 0

    packaged = {
        "post_cluster_label": post_label,
        "match_type": str(cluster_row["match_type"]),
        "match_label": str(cluster_row["match_label"]),
        "post_cluster_size": int(cluster_row["cluster_size"]),
        "post_period_share": float(cluster_row["period_share"]),
        "ticker_count": int(cluster_row["ticker_count"]),
        "filing_count": int(cluster_row["filing_count"]),
        "top_terms": str(cluster_row["top_terms"]),
        "top_tickers": str(cluster_row.get("top_tickers", "")),
        "top_ticker_share": top_ticker_share_value,
        "top_ticker_n": top_ticker_n_value,
        "prime_share": float(cluster_row.get("prime", 0.0)),
        "supplier_share": float(cluster_row.get("supplier", 0.0)),
        "post_year_distribution": post_year_distribution,
        "best_pre_cluster_label": pre_label,
        "best_pre_similarity": float(cluster_row.get("best_pre_similarity", np.nan)),
        "best_pre_top_terms": str(cluster_row.get("best_pre_top_terms", "")),
        "shared_terms": str(cluster_row.get("shared_terms", "")),
        "top_pre_candidates": str(cluster_row.get("top_pre_candidates", "")),
        "central_post_examples": central_post_examples,
        "mid_post_examples": mid_post_examples,
        "peripheral_post_examples": peripheral_post_examples,
        "matched_pre_examples": matched_pre_examples,
    }
    return packaged


def build_post_year_distribution(examples: list[dict[str, Any]]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for example in examples:
        year = example.get("filing_year")
        if year is None:
            continue
        key = str(int(year))
        distribution[key] = distribution.get(key, 0) + 1
    return dict(sorted(distribution.items()))


def build_all_evidence_packages(
    interesting_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    representative_df: pd.DataFrame,
    embedding_model_name: str,
    embedding_payload: tuple[np.ndarray, np.ndarray] | None,
    allow_reembed: bool,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    cluster_labels = interesting_df["period_cluster_label"].astype(str).tolist()
    pre_labels = [str(value) for value in interesting_df["best_pre_cluster_label"].dropna().tolist() if str(value).strip()]
    unique_labels = sorted(set(cluster_labels + pre_labels))
    cluster_distance_df = compute_cluster_distance_frame(
        sampled_df,
        unique_labels,
        embedding_model_name,
        embedding_payload=embedding_payload,
        allow_reembed=allow_reembed,
    )

    packages = []
    for _, row in interesting_df.iterrows():
        packages.append(
            build_cluster_evidence_package(
                row,
                sampled_df=sampled_df,
                representative_df=representative_df,
                cluster_distance_df=cluster_distance_df,
                args=args,
            )
        )
    return packages


def gemini_json_schema_cluster() -> dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "card_title",
            "headline",
            "why_interesting",
            "interpretation",
            "continuity_judgment",
            "continuity_note",
            "count_gap_role",
            "count_gap_note",
            "layer_reading",
            "concentration_note",
            "temporal_reading_post",
            "intra_cluster_churn_note",
            "category_blending_note",
            "evidence_points",
            "genre_caveat",
            "use_in_abstract",
            "abstract_candidate_sentence",
            "confidence",
            "supporting_example_ids",
        ],
        "properties": {
            "card_title": {"type": "string"},
            "headline": {"type": "string"},
            "why_interesting": {"type": "string"},
            "interpretation": {"type": "string"},
            "continuity_judgment": {
                "type": "string",
                "enum": [
                    "largely_continuous",
                    "same_cluster_shifted_contents",
                    "clear_structural_change",
                    "genuinely_new_theme",
                    "weak_or_unclear_change",
                ],
            },
            "continuity_note": {"type": "string"},
            "count_gap_role": {
                "type": "string",
                "enum": [
                    "genuine_novelty",
                    "split_or_refinement",
                    "denser_or_more_explicit_disclosure",
                    "little_or_no_count_gap_role",
                    "unclear",
                ],
            },
            "count_gap_note": {"type": "string"},
            "layer_reading": {"type": "string"},
            "concentration_note": {"type": "string"},
            "temporal_reading_post": {"type": "string"},
            "intra_cluster_churn_note": {"type": "string"},
            "category_blending_note": {"type": "string"},
            "evidence_points": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 4,
            },
            "genre_caveat": {"type": "string"},
            "use_in_abstract": {"type": "boolean"},
            "abstract_candidate_sentence": {"type": "string"},
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
            "supporting_example_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 6,
            },
        },
    }


def gemini_json_schema_abstract() -> dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "report_title",
            "primary_result",
            "abstract",
            "boilerplate_envelope_finding",
            "layer_asymmetries",
            "major_findings",
            "limitations",
            "intra_cluster_churn_caveat",
            "closing_caution",
        ],
        "properties": {
            "report_title": {"type": "string"},
            "primary_result": {"type": "string"},
            "abstract": {"type": "string"},
            "boilerplate_envelope_finding": {"type": "string"},
            "layer_asymmetries": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 4,
            },
            "major_findings": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 5,
            },
            "limitations": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 4,
            },
            "intra_cluster_churn_caveat": {"type": "string"},
            "closing_caution": {"type": "string"},
        },
    }


def call_gemini_json(
    api_key: str,
    model_name: str,
    prompt: str,
    schema: dict[str, Any],
    temperature: float,
) -> dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API request failed: {exc.code} {detail}") from exc

    candidates = body.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini API returned no candidates: {body}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        raise RuntimeError(f"Gemini API returned empty text: {body}")

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini returned non-JSON text: {text}") from exc


def build_cluster_prompt(cluster_package: dict[str, Any], report_context: dict[str, Any]) -> str:
    payload = json.dumps(cluster_package, indent=2, ensure_ascii=False)
    match_type = str(cluster_package.get("match_type", "")).strip()
    count_shift_summary = build_count_shift_summary(report_context)
    if match_type == EMERGENT_MATCH_TYPE:
        focus_guidance = textwrap.dedent(
            """
            Additional focus for this cluster:
            - Treat this first as a discovery question, not as a forced comparison question.
            - Ask whether this looks like a genuinely emergent post-2022 disclosure theme with breadth across firms and filings.
            - A post-only cluster can still be analytically important even if it is somewhat narrower than the broad structural-comparison clusters, as long as it is not just one-firm language.
            - Distinguish between a genuinely new theme and a cluster that only looks new because post-2022 wording became slightly more explicit or semantically denser.
            - If the cluster seems broad and coherent, say what became newly explicit after 2022.
            - If the cluster instead looks like firm-specific wording, boilerplate drift, or a weak novelty claim, say that clearly.
            - Do not force a pre-2022 analogue if the evidence package suggests there is none.
            """
        ).strip()
    else:
        focus_guidance = textwrap.dedent(
            """
            Additional focus for this cluster:
            - Treat this as a structural change question.
            - Judge whether the post cluster looks like a refinement, merger, or reframing of earlier pre-2022 themes.
            - Be explicit about what appears continuous versus what appears newly differentiated.
            - A high embedding similarity does not mean nothing changed. The cluster may still carry a slightly different disclosure center of gravity, risk emphasis, or example mix in the post period.
            - Distinguish between true structural change and a same-cluster case where the contents shifted but the embedding model still places the clusters close together.
            """
        ).strip()
    return textwrap.dedent(
        f"""
        You are helping with an academic exploratory analysis of U.S. defense-sector SEC 10-K Item 1A risk factor disclosures.

        Critical genre context:
        - These texts are risk disclosures from annual filings, not neutral event logs.
        - They are shaped by legal, strategic, and investor-relations incentives.
        - Boilerplate and broad cautionary language are common.
        - You should describe shifts in disclosure emphasis or thematic structure, not claim direct changes in real-world risk incidence unless the text package clearly supports it.
        - Be especially cautious when the evidence is concentrated in a small number of firms or looks like routine filing boilerplate.

        Non-fabrication rule:
        - Every claim you make must be grounded in the evidence package or the global count-shift context supplied below.
        - Do not invent events, firms, dates, or regulatory actions that are not present in the text of the examples or in the provided fields.
        - If a field such as `temporal_reading_post` or `layer_reading` cannot be answered from the evidence package, say so explicitly rather than guessing.

        Your task:
        - Analyze one interesting post-2022 cluster and its best pre-2022 comparison context.
        - Focus on why this cluster was flagged as interesting.
        - Use the central examples to understand the core of the cluster.
        - Use the mid-distance examples to see whether the theme still holds once you move away from the centroid.
        - Use the peripheral examples to test how broad the cluster still is near its outer edge.
        - Use the matched pre examples to judge whether this is genuinely new, more specific, or simply a renamed variant of an earlier theme.
        - Do not rely on centroid similarity alone. Sometimes the embedding model will keep two clusters close even when the post examples show a shifted emphasis, more explicit wording, or a somewhat different internal mix of risks.
        - Your job is to say whether the post cluster is mostly continuous, subtly shifted in content, structurally changed, or genuinely new.
        - The broader run produced {count_shift_summary['pre_clusters']} pre clusters versus {count_shift_summary['post_clusters']} post clusters. Use this cluster to judge whether that count gap looks more like thematic splitting, genuine novelty, or denser post disclosure language.
        - Be critical and restrained. If the evidence is weak or mixed, say so.
        - For broad new_post_only clusters, treat emergence itself as a potential finding.

        {focus_guidance}

        Required interpretation rule:
        - Fill `continuity_judgment` with exactly one of:
          - `largely_continuous`: pre and post appear mostly similar in substance.
          - `same_cluster_shifted_contents`: the cluster still looks like roughly the same thematic family, but the post examples show a noticeable shift in emphasis, explicitness, or internal composition.
          - `clear_structural_change`: the post cluster looks substantively reorganized, split/refined, merged, or reframed relative to the pre examples.
          - `genuinely_new_theme`: the post cluster looks meaningfully new rather than just a variant of the older theme.
          - `weak_or_unclear_change`: evidence is too mixed, thin, or boilerplate-heavy to say confidently.
        - Use `continuity_note` to explain that judgment in one concise sentence.
        - Fill `count_gap_role` with exactly one of:
          - `genuine_novelty`: this cluster plausibly contributes to the higher post cluster count because it looks like a new theme.
          - `split_or_refinement`: this cluster plausibly contributes to the higher post cluster count because a broader earlier theme has split into a more specific post cluster.
          - `denser_or_more_explicit_disclosure`: this cluster plausibly contributes to the count gap mainly because post wording is more explicit, denser, or more semantically differentiated rather than because the theme is truly new.
          - `little_or_no_count_gap_role`: this cluster does not seem central to explaining the overall count difference.
          - `unclear`: the evidence is too mixed to say.
        - Use `count_gap_note` to explain that judgment in one concise sentence.

        Required secondary readings (each one concise sentence, grounded in the evidence package):

        1. `layer_reading`
        - Read `prime_share` and `supplier_share` and the `company_layer` field on individual examples.
        - If the cluster leans strongly toward primes or suppliers (for example, one layer above 0.75), name that asymmetry and briefly say what it implies for the theme.
        - If the cluster is mixed or roughly balanced, say so and do not invent an asymmetry.
        - If the shares sum to roughly zero because layer metadata is missing, write "Layer composition unavailable from the evidence package."

        2. `concentration_note`
        - Read `top_tickers`, `top_ticker_share`, and `top_ticker_n`.
        - If `top_ticker_share` is above roughly 0.25, flag this cluster as top-ticker concentrated and name the dominant ticker.
        - If the cluster spreads across many tickers, say so and note that this strengthens the claim that the theme is sectoral rather than firm-specific.
        - Do not override the clustering screens that already admitted this cluster; just report what the concentration numbers show.

        3. `temporal_reading_post`
        - Read `post_year_distribution` and the `filing_year` field on post examples.
        - If the post evidence clusters into a narrow year range (for example, only 2024-2025), say so and note that the theme may track a specific policy or event window.
        - If the post evidence is spread relatively evenly across the post period, say so and note that the theme reads as persistent post-2022 disclosure rather than a single-year spike.
        - If the year distribution is too thin or mixed to judge, say so explicitly.

        4. `intra_cluster_churn_note`
        - Embeddings can keep a cluster stable across periods even when the contents inside the cluster have shifted substantially (for example, a supply-chain cluster whose internal mix moves from generic sourcing language toward semiconductor and titanium sourcing after 2022).
        - Compare the matched pre examples against the central and mid post examples. If the cluster family looks continuous but the internal emphasis has clearly changed (new named risks, new regulations, new firms or geographies), name that intra-cluster churn even if `continuity_judgment` is `largely_continuous` or `same_cluster_shifted_contents`.
        - If you genuinely see no such internal churn, say so.
        - This field should make explicit what pure cluster-identity analysis risks hiding.

        5. `category_blending_note`
        - Embedding clusters sometimes bundle together text that a conceptual taxonomy would separate. The project's working taxonomy distinguishes, among others, Government budget and procurement; Contract execution and economics; Supply chain and industrial base; Labor, clearances, and human capital; Cyber, data, and IT systems; Legal, regulatory, and compliance; Geopolitical, international, sanctions, and export controls; Macro, financial, and capital markets; Technology, product, AI, and autonomy; Intellectual property; M&A and portfolio strategy; Environmental, climate, ESG, and safety; Governance, securities, and shareholder risk; and Pandemic and public-health risk.
        - If this embedding cluster visibly mixes two or more of those conceptual categories (for example, legal compliance plus cyber, or export controls plus broader international risk), name the mixed categories in one concise sentence.
        - If the cluster cleanly matches a single conceptual category, say so. Do not over-claim blending.

        Return JSON only, matching the schema exactly.

        Global count-shift context:
        {json.dumps(count_shift_summary, indent=2, ensure_ascii=False)}

        Evidence package:
        {payload}
        """
    ).strip()


def build_abstract_prompt(
    cluster_packages: list[dict[str, Any]],
    cluster_analyses: list[dict[str, Any]],
    metadata: dict[str, Any],
    report_context: dict[str, Any],
) -> str:
    count_shift_summary = build_count_shift_summary(report_context)
    is_full_corpus_run = bool(report_context.get("is_full_corpus_run", False))
    method_scope = (
        "same embeddings for the full corpus, separate pre/post cluster discovery, approximate post-to-pre matching"
        if is_full_corpus_run
        else "same embeddings for the clustered rows, separate pre/post cluster discovery, approximate post-to-pre matching"
    )
    count_gap_guidance = (
        "This run clusters the full corpus rather than a balanced sample, so the result is no longer subject to sampling variance, but modest pre/post corpus-volume differences can still matter and should be discussed."
        if is_full_corpus_run
        else "Balanced sampling removes raw row-count imbalance, but denser post language remains a possible explanation and should be discussed."
    )
    payload = {
        "analysis_context": {
            "corpus": "U.S. defense-sector SEC 10-K Item 1A risk factor disclosures",
            "comparison": "2018-2021 versus 2022-2025",
            "method": method_scope,
            "embedding_model": metadata.get("model_name", ""),
        },
        "count_shift_summary": count_shift_summary,
        "report_context": report_context,
        "cluster_packages": cluster_packages,
        "cluster_analyses": cluster_analyses,
    }
    new_post_only_count = int(count_shift_summary.get("new_post_only_count", 0))
    return textwrap.dedent(
        f"""
        You are writing the executive abstract for an exploratory report on defense-sector SEC 10-K Item 1A risk disclosures.

        Important framing:
        - This is disclosure analysis, not direct measurement of realized external risk.
        - The data come from legally cautious corporate filings, so boilerplate and strategic emphasis matter.
        - You should summarize only the strongest major findings supported by the supplied cluster analyses.
        - Broad post-only clusters can be findings in their own right; if they are well-supported, treat them as discoveries rather than as leftovers from comparison.
        - Broad structural-comparison clusters and emergent discovery clusters do not use identical narration filters here; emergent clusters are allowed to be somewhat narrower as long as they still span multiple firms and filings.
        - Do not claim one-to-one cluster identity across periods.
        - Use careful language such as "post-2022 disclosures appear to..." or "the post period shows a more explicit subtheme around..."
        - Treat "same cluster, shifted contents" as a real result category when the cluster family appears stable but the post examples show a changed center of gravity or more explicit disclosure emphasis.
        - The report's first result is the cluster count shift itself: {count_shift_summary['pre_clusters']} pre-2022 clusters versus {count_shift_summary['post_clusters']} post-2022 clusters.
        - You must explicitly assess whether that count gap looks more like thematic differentiation, genuinely new themes, or denser/more explicit post disclosure language.
        - {count_gap_guidance}
        - The abstract should synthesize the major findings across the interesting changed clusters, not repeat every detail.
        - If the strongest evidence points to genuinely emergent post-2022 themes, lead with those before the split/refined or merged cases.
        - Mention limitations explicitly.
        - Fill `primary_result` with a concise statement that leads with the cluster count difference and your best interpretation of what it means.

        Non-fabrication rule:
        - Every claim must be grounded in the supplied `cluster_packages`, `cluster_analyses`, or `count_shift_summary`.
        - Do not invent events, firms, dates, or regulatory actions that are not present in those inputs.

        Required: `boilerplate_envelope_finding`
        - This run flagged {new_post_only_count} cluster(s) as clean `new_post_only` under the narration filters.
        - If that count is zero or low and the dominant change category is split/refined or same-cluster-shifted-contents, write one or two sentences stating that the post period shows no genuinely detached new themes and that observable change is absorbed into the existing disclosure families as splitting, refinement, or denser wording. Frame this as a finding about the genre of risk disclosures, not a failure of the method.
        - If genuinely new themes are present and well-supported, instead briefly describe what they are and why they look novel rather than refined.
        - Do not repeat numbers already in `primary_result`; speak to what the pattern means.

        Required: `layer_asymmetries`
        - Scan `cluster_packages[*].prime_share` and `supplier_share` alongside the per-cluster `layer_reading` entries in `cluster_analyses`.
        - List up to 3 concise sentences describing the clearest prime-versus-supplier asymmetries you can ground in those fields. For example, which theme looks supplier-heavy and which looks prime-heavy, and what that implies for the reading of the post-2022 shift.
        - If there is no clear asymmetry across the narrated clusters, return an empty list. Do not fabricate an asymmetry.

        Required: `intra_cluster_churn_caveat`
        - Aggregate the per-cluster `intra_cluster_churn_note` fields.
        - Write one or two sentences explaining that embedding-based cluster identity cannot by itself reveal churn inside a cluster: a cluster that stays geometrically stable can still have its internal content reshape substantially across periods.
        - If the per-cluster evidence suggests this happened in several narrated clusters (for example, supply-chain language adding semiconductors and critical minerals, or cyber language absorbing ransomware, AI-driven threats, and the SEC cyber disclosure rule), name that briefly.
        - Use this field to make intra-cluster churn a live interpretive category alongside the count-gap reading.

        Required: `limitations`
        - Must include an explicit limitation about this being disclosure analysis rather than incidence measurement.
        - Must include at least one limitation about the alternative "denser disclosure language" explanation for apparent thematic differentiation.
        - May include limitations about approximate matching, concentration effects, or LLM non-determinism when grounded in the inputs.

        Return JSON only, matching the schema exactly.

        Input:
        {json.dumps(payload, indent=2, ensure_ascii=False)}
        """
    ).strip()


def build_report_context(
    full_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    interesting_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    pre_summary = summary_df[summary_df["period_bucket"] == PRE_PERIOD].copy()
    post_summary = summary_df[summary_df["period_bucket"] == POST_PERIOD].copy()

    def compact_rows(df: pd.DataFrame, columns: list[str], n: int = 8) -> list[dict[str, Any]]:
        if df.empty:
            return []
        subset = df.sort_values(["period_share", "cluster_size"], ascending=[False, False]).head(n).copy()
        return subset[columns].to_dict(orient="records")

    interesting_rows = []
    if not interesting_df.empty:
        interesting_rows = interesting_df[
            [
                "period_cluster_label",
                "match_type",
                "match_label",
                "period_share",
                "cluster_size",
                "ticker_count",
                "filing_count",
                "top_terms",
                "best_pre_cluster_label",
                "best_pre_similarity",
            ]
        ].to_dict(orient="records")

    sampled_stats = sampled_df.copy()
    sampled_stats["word_count"] = sampled_stats["text"].astype(str).str.split().map(len)
    word_stats = (
        sampled_stats.groupby("period_bucket")["word_count"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    word_stats_records = []
    for _, row in word_stats.iterrows():
        word_stats_records.append(
            {
                "period_bucket": row["period_bucket"],
                "clustered_rows": int(row["count"]),
                "mean_words": float(row["mean"]),
                "median_words": float(row["median"]),
            }
        )

    all_post_catalog = post_summary[post_summary["period_cluster"] != -1].copy()
    all_post_catalog = all_post_catalog.merge(
        matches_df,
        left_on="period_cluster_label",
        right_on="post_cluster_label",
        how="left",
    )
    all_post_catalog["match_type"] = all_post_catalog["match_type"].fillna("new_post_only")
    non_noise_match_counts = all_post_catalog["match_type"].value_counts().to_dict() if not all_post_catalog.empty else {}
    broad_post_summary = post_summary[(post_summary["period_cluster"] != -1) & (post_summary["eligible_period_theme"])].copy()
    broad_post_catalog = broad_post_summary.merge(
        matches_df,
        left_on="period_cluster_label",
        right_on="post_cluster_label",
        how="left",
    )
    broad_post_catalog["match_type"] = broad_post_catalog["match_type"].fillna("new_post_only")
    broad_match_counts = broad_post_catalog["match_type"].value_counts().to_dict() if not broad_post_catalog.empty else {}
    pre_cluster_count = int((pre_summary["period_cluster"] != -1).sum())
    post_cluster_count = int((post_summary["period_cluster"] != -1).sum())
    is_full_corpus_run = int(len(sampled_df)) == int(len(full_df))

    return {
        "full_rows": int(len(full_df)),
        "clustered_rows": int(len(sampled_df)),
        "is_full_corpus_run": is_full_corpus_run,
        "companies": int(full_df["ticker"].nunique()),
        "pre_cluster_count": pre_cluster_count,
        "post_cluster_count": post_cluster_count,
        "cluster_count_gap": int(post_cluster_count - pre_cluster_count),
        "interesting_cluster_count": int(len(interesting_df)),
        "emergent_cluster_count": int((interesting_df["match_type"] == EMERGENT_MATCH_TYPE).sum()) if not interesting_df.empty else 0,
        "interesting_cluster_rows": interesting_rows,
        "emergent_cluster_rows": [row for row in interesting_rows if row.get("match_type") == EMERGENT_MATCH_TYPE],
        "clustered_word_stats": word_stats_records,
        "non_noise_post_match_type_counts": non_noise_match_counts,
        "broad_post_match_type_counts": broad_match_counts,
        "broad_pre_cluster_count": int(((pre_summary["period_cluster"] != -1) & (pre_summary["eligible_period_theme"])).sum()),
        "broad_post_cluster_count": int(((post_summary["period_cluster"] != -1) & (post_summary["eligible_period_theme"])).sum()),
        "narration_filters": {
            "structural": {
                "min_cluster_size": FOCUS_MIN_CLUSTER_SIZE,
                "min_ticker_count": FOCUS_MIN_TICKER_COUNT,
                "min_filing_count": FOCUS_MIN_FILING_COUNT,
                "max_top_ticker_share": FOCUS_MAX_TOP_TICKER_SHARE,
            },
            "emergent": {
                "min_cluster_size": args.emergent_min_cluster_size,
                "min_ticker_count": args.emergent_min_ticker_count,
                "min_filing_count": args.emergent_min_filing_count,
                "max_top_ticker_share": args.emergent_max_top_ticker_share,
            },
        },
        "top_pre_clusters": compact_rows(pre_summary, ["period_cluster_label", "period_share", "cluster_size", "ticker_count", "filing_count", "top_terms"]),
        "top_post_clusters": compact_rows(post_summary, ["period_cluster_label", "period_share", "cluster_size", "ticker_count", "filing_count", "top_terms"]),
        "match_type_counts": non_noise_match_counts,
    }


def build_count_shift_summary(report_context: dict[str, Any]) -> dict[str, Any]:
    pre_clusters = int(report_context.get("pre_cluster_count", 0))
    post_clusters = int(report_context.get("post_cluster_count", 0))
    gap = int(report_context.get("cluster_count_gap", 0))
    broad_pre_clusters = int(report_context.get("broad_pre_cluster_count", 0))
    broad_post_clusters = int(report_context.get("broad_post_cluster_count", 0))
    is_full_corpus_run = bool(report_context.get("is_full_corpus_run", False))
    word_stats = {row["period_bucket"]: row for row in report_context.get("clustered_word_stats", [])}
    pre_words = word_stats.get(PRE_PERIOD, {})
    post_words = word_stats.get(POST_PERIOD, {})
    match_counts = report_context.get("non_noise_post_match_type_counts", {})
    split_count = int(match_counts.get("split/refined", 0))
    merged_count = int(match_counts.get("merged", 0))
    persistent_count = int(match_counts.get("persistent", 0))
    new_count = int(match_counts.get("new_post_only", 0))
    run_scope = "Full-corpus clustering" if is_full_corpus_run else "Balanced clustering"
    if gap > 0:
        headline_note = (
            f"{run_scope} yields {pre_clusters} pre-2022 versus {post_clusters} post-2022 non-noise clusters, "
            f"so the post period resolves into {gap} more clusters."
        )
    elif gap < 0:
        headline_note = (
            f"{run_scope} yields {pre_clusters} pre-2022 versus {post_clusters} post-2022 non-noise clusters, "
            f"so the post period resolves into {abs(gap)} fewer clusters."
        )
    else:
        headline_note = (
            f"{run_scope} yields {pre_clusters} pre-2022 versus {post_clusters} post-2022 non-noise clusters, "
            "so the two periods resolve into the same number of clusters."
        )
    decomposition_note = (
        f"Across all non-noise post clusters, {split_count} are classified as split/refined, "
        f"{merged_count} as merged, {persistent_count} as persistent, and {new_count} as clean new_post_only themes."
    )
    broad_note = (
        f"Among the broader clusters that clear the theme screen, counts move from {broad_pre_clusters} pre to "
        f"{broad_post_clusters} post, so the gap is not only being created by tiny fringe fragments."
    )
    if is_full_corpus_run:
        volume_note = (
            "Because this run clusters the full corpus rather than a balanced sample, the results are no longer subject "
            "to sampling variance, but they may still reflect the modest difference in pre/post corpus volume. "
            f"Clustered row length is {pre_words.get('mean_words', 0.0):.1f} vs {post_words.get('mean_words', 0.0):.1f} mean words "
            f"and {pre_words.get('median_words', 0.0):.1f} vs {post_words.get('median_words', 0.0):.1f} median words "
            "(pre vs post), so denser post disclosure language remains a live alternative explanation alongside genuine thematic differentiation."
        )
    else:
        volume_note = (
            "Because the per-period sample size is balanced, the count gap is not a raw row-count artifact. "
            f"Clustered row length is {pre_words.get('mean_words', 0.0):.1f} vs {post_words.get('mean_words', 0.0):.1f} mean words "
            f"and {pre_words.get('median_words', 0.0):.1f} vs {post_words.get('median_words', 0.0):.1f} median words "
            "(pre vs post), so denser post disclosure language remains a live alternative explanation alongside genuine thematic differentiation."
        )
    return {
        "headline_note": headline_note,
        "decomposition_note": decomposition_note,
        "broad_note": broad_note,
        "volume_note": volume_note,
        "is_full_corpus_run": is_full_corpus_run,
        "run_scope": run_scope,
        "pre_clusters": pre_clusters,
        "post_clusters": post_clusters,
        "cluster_gap": gap,
        "broad_pre_clusters": broad_pre_clusters,
        "broad_post_clusters": broad_post_clusters,
        "split_refined_count": split_count,
        "merged_count": merged_count,
        "persistent_count": persistent_count,
        "new_post_only_count": new_count,
    }


def run_cluster_analysis(
    cluster_packages: list[dict[str, Any]],
    api_key: str,
    model_name: str,
    temperature: float,
    skip_llm: bool,
    llm_max_concurrency: int,
    llm_request_stagger_seconds: float,
    progress_path: Path,
    output_path: Path,
    report_context: dict[str, Any],
) -> list[dict[str, Any]]:
    saved_progress: dict[str, dict[str, Any]] = {}
    if progress_path.exists():
        try:
            progress_payload = read_json(progress_path)
            if isinstance(progress_payload, dict):
                rows = progress_payload.get("rows", [])
                progress_version = int(progress_payload.get("prompt_version", -1))
            else:
                rows = progress_payload
                progress_version = -1
            if progress_version == CLUSTER_ANALYSIS_PROMPT_VERSION and isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and "post_cluster_label" in row and "analysis" in row:
                        saved_progress[str(row["post_cluster_label"])] = normalize_cluster_analysis(row["analysis"])
        except Exception:
            saved_progress = {}

    def persist_progress() -> None:
        progress_rows = []
        output_rows = []
        for package in cluster_packages:
            label = str(package["post_cluster_label"])
            if label in saved_progress:
                progress_rows.append({"post_cluster_label": label, "analysis": saved_progress[label]})
                output_rows.append(saved_progress[label])
        write_json(
            progress_path,
            {
                "prompt_version": CLUSTER_ANALYSIS_PROMPT_VERSION,
                "rows": progress_rows,
            },
        )
        write_json(output_path, output_rows)

    def build_skip_analysis(package: dict[str, Any]) -> dict[str, Any]:
        analysis = {
            "card_title": package["post_cluster_label"],
            "headline": "LLM step skipped.",
            "why_interesting": f"{package['match_label']} cluster selected for narrative review.",
            "interpretation": "No Gemini interpretation was generated because --skip-llm was used.",
            "continuity_judgment": "weak_or_unclear_change",
            "continuity_note": "Continuity judgment unavailable because Gemini was skipped for this run.",
            "count_gap_role": "unclear",
            "count_gap_note": "Cluster-count role unavailable because Gemini was skipped for this run.",
            "layer_reading": "Layer reading unavailable because Gemini was skipped for this run.",
            "concentration_note": "Concentration reading unavailable because Gemini was skipped for this run.",
            "temporal_reading_post": "Temporal reading unavailable because Gemini was skipped for this run.",
            "intra_cluster_churn_note": "Intra-cluster churn reading unavailable because Gemini was skipped for this run.",
            "category_blending_note": "Category-blending reading unavailable because Gemini was skipped for this run.",
            "evidence_points": [
                "Central examples capture the cluster core.",
                "Peripheral examples test whether the theme remains coherent away from the centroid.",
            ],
            "genre_caveat": "Interpretation skipped.",
            "use_in_abstract": True,
            "abstract_candidate_sentence": f"{package['post_cluster_label']} was selected for review but not summarized by Gemini.",
            "confidence": "low",
            "supporting_example_ids": [example["example_id"] for example in package["central_post_examples"][:1]],
        }
        return normalize_cluster_analysis(analysis)

    def analyze_single_package(package: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        cluster_label = str(package["post_cluster_label"])
        if skip_llm:
            return cluster_label, build_skip_analysis(package)
        prompt = build_cluster_prompt(package, report_context=report_context)
        analysis = call_gemini_json(
            api_key=api_key,
            model_name=model_name,
            prompt=prompt,
            schema=gemini_json_schema_cluster(),
            temperature=temperature,
        )
        return cluster_label, normalize_cluster_analysis(analysis)

    pending_packages = [
        package for package in cluster_packages if str(package["post_cluster_label"]) not in saved_progress
    ]
    if not pending_packages:
        return [saved_progress[str(package["post_cluster_label"])] for package in cluster_packages]

    if saved_progress:
        print(
            f"Reusing cached Gemini analyses for {len(saved_progress)} cluster(s); "
            f"submitting {len(pending_packages)} new cluster request(s)."
        )

    max_workers = 1 if skip_llm else max(1, int(llm_max_concurrency))
    stagger_seconds = max(0.0, float(llm_request_stagger_seconds))

    if max_workers == 1:
        total_pending = len(pending_packages)
        print(f"Running {total_pending} cluster-level Gemini request(s) sequentially.")
        for completed, package in enumerate(pending_packages, start=1):
            cluster_label, analysis = analyze_single_package(package)
            saved_progress[cluster_label] = analysis
            persist_progress()
            print(f"[{completed}/{total_pending}] Cluster analysis ready: {cluster_label}")
        return [saved_progress[str(package["post_cluster_label"])] for package in cluster_packages]

    print(
        f"Running {len(pending_packages)} cluster-level Gemini request(s) with "
        f"max concurrency {max_workers} and {stagger_seconds:.2f}s submit stagger."
    )
    failures: list[tuple[str, str]] = []
    future_to_label: dict[Any, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, package in enumerate(pending_packages):
            cluster_label = str(package["post_cluster_label"])
            future_to_label[executor.submit(analyze_single_package, package)] = cluster_label
            if stagger_seconds > 0 and index < len(pending_packages) - 1:
                time.sleep(stagger_seconds)

        completed = 0
        for future in as_completed(future_to_label):
            cluster_label = future_to_label[future]
            try:
                returned_label, analysis = future.result()
            except Exception as exc:
                failures.append((cluster_label, str(exc)))
                print(f"[error] Cluster analysis failed for {cluster_label}: {exc}")
                continue
            saved_progress[returned_label] = analysis
            persist_progress()
            completed += 1
            print(f"[{completed}/{len(pending_packages)}] Cluster analysis ready: {returned_label}")

    if failures:
        failed_labels = ", ".join(label for label, _ in failures)
        raise RuntimeError(
            "Gemini cluster analysis failed for "
            f"{len(failures)} cluster(s): {failed_labels}. "
            f"Completed results were checkpointed to {progress_path}."
        )

    return [saved_progress[str(package["post_cluster_label"])] for package in cluster_packages]


def run_abstract_analysis(
    cluster_packages: list[dict[str, Any]],
    cluster_analyses: list[dict[str, Any]],
    metadata: dict[str, Any],
    report_context: dict[str, Any],
    api_key: str,
    model_name: str,
    temperature: float,
    skip_llm: bool,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        try:
            payload = read_json(output_path)
            if (
                isinstance(payload, dict)
                and payload
                and int(payload.get("_prompt_version", -1)) == ABSTRACT_PROMPT_VERSION
            ):
                return normalize_abstract(payload)
        except Exception:
            pass

    if skip_llm:
        abstract = normalize_abstract({
            "report_title": "LLM abstract skipped",
            "primary_result": "The cluster-count shift was not interpreted because Gemini was skipped for this run.",
            "abstract": "The evidence package was built successfully, but Gemini was not called because --skip-llm was used.",
            "boilerplate_envelope_finding": "Boilerplate-envelope reading unavailable because Gemini was skipped for this run.",
            "layer_asymmetries": [],
            "major_findings": [
                "Interesting post clusters were selected from the period-shift artifacts.",
                "Each selected cluster includes centroid-near and moderately peripheral evidence.",
                "Matched pre-cluster context is packaged for comparison.",
            ],
            "limitations": [
                "No model-written interpretation is included.",
                "This summary is a placeholder rather than an analytical abstract.",
            ],
            "intra_cluster_churn_caveat": "Intra-cluster churn caveat unavailable because Gemini was skipped for this run.",
            "closing_caution": "Run without --skip-llm to generate the actual narrative synthesis.",
        })
        write_json(output_path, abstract)
        return abstract

    prompt = build_abstract_prompt(cluster_packages, cluster_analyses, metadata, report_context)
    abstract = call_gemini_json(
        api_key=api_key,
        model_name=model_name,
        prompt=prompt,
        schema=gemini_json_schema_abstract(),
        temperature=temperature,
    )
    abstract = normalize_abstract(abstract)
    abstract["_prompt_version"] = ABSTRACT_PROMPT_VERSION
    write_json(output_path, abstract)
    return abstract


def cluster_count_shift_figure(summary_df: pd.DataFrame, template_name: str) -> go.Figure:
    plot_rows = []
    for period in [PRE_PERIOD, POST_PERIOD]:
        period_summary = summary_df[summary_df["period_bucket"] == period].copy()
        plot_rows.append(
            {
                "period_title": PERIOD_META[period]["title"],
                "count_type": "All non-noise clusters",
                "count": int((period_summary["period_cluster"] != -1).sum()),
            }
        )
        plot_rows.append(
            {
                "period_title": PERIOD_META[period]["title"],
                "count_type": "Broad narrated themes",
                "count": int(((period_summary["period_cluster"] != -1) & (period_summary["eligible_period_theme"])).sum()),
            }
        )
    plot_df = pd.DataFrame(plot_rows)
    fig = px.bar(
        plot_df,
        x="period_title",
        y="count",
        color="count_type",
        barmode="group",
        template=template_name,
        title="Cluster counts by period",
        labels={"period_title": "Period", "count": "Clusters", "count_type": "Cluster scope"},
        color_discrete_map={
            "All non-noise clusters": "#0f4c5c",
            "Broad narrated themes": "#c56b3c",
        },
    )
    fig.update_layout(height=360)
    return fig


def period_text_density_figure(sampled_df: pd.DataFrame, template_name: str) -> go.Figure:
    plot_df = sampled_df.copy()
    plot_df["word_count"] = plot_df["text"].astype(str).str.split().map(len)
    plot_df["period_title"] = plot_df["period_bucket"].map(lambda value: PERIOD_META.get(value, {}).get("title", value))
    fig = px.box(
        plot_df,
        x="period_title",
        y="word_count",
        color="period_title",
        template=template_name,
        title="Clustered row length by period",
        labels={"period_title": "Period", "word_count": "Words per clustered row"},
        color_discrete_map={
            PERIOD_META[PRE_PERIOD]["title"]: "#3e6c8f",
            PERIOD_META[POST_PERIOD]["title"]: "#c56b3c",
        },
        points=False,
    )
    fig.update_layout(height=360, showlegend=False)
    return fig


def interesting_cluster_figure(interesting_df: pd.DataFrame, template_name: str) -> go.Figure:
    if interesting_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No interesting clusters matched the current filters.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(template=template_name, height=360, xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    plot_df = interesting_df.sort_values(["match_priority", "period_share"], ascending=[False, True])
    fig = px.bar(
        plot_df,
        x="period_share",
        y="period_cluster_label",
        orientation="h",
        color="match_type",
        template=template_name,
        title="Narrated post-2022 clusters across discovery types",
        labels={"period_share": "Share of clustered post rows", "period_cluster_label": "Post cluster", "match_type": "Match type"},
        hover_data={"best_pre_cluster_label": True, "best_pre_similarity": ":.2f", "top_terms": True},
        color_discrete_map={
            "new_post_only": "#0f4c5c",
            "split/refined": "#1b998b",
            "merged": "#c56b3c",
            "persistent": "#5a8a3e",
            "approximate_overlap": "#3e6c8f",
        },
    )
    fig.update_layout(height=420)
    return fig


def confidence_figure(cluster_cards: list[dict[str, Any]], template_name: str) -> go.Figure:
    if not cluster_cards:
        fig = go.Figure()
        fig.add_annotation(text="No cluster analyses available.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(template=template_name, height=320, xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    plot_df = pd.DataFrame(
        {
            "cluster_label": [card["post_cluster_label"] for card in cluster_cards],
            "confidence": [card["analysis"]["confidence"] for card in cluster_cards],
            "match_type": [card["match_type"] for card in cluster_cards],
        }
    )
    fig = px.bar(
        plot_df,
        x="cluster_label",
        y=np.ones(len(plot_df)),
        color="confidence",
        template=template_name,
        title="Gemini confidence labels across narrated clusters",
        labels={"x": "Cluster", "y": "", "confidence": "Confidence"},
        color_discrete_map={"low": "#c56b3c", "medium": "#3e6c8f", "high": "#5a8a3e"},
    )
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_layout(height=320)
    return fig


def is_emergent_card(card: dict[str, Any]) -> bool:
    judgment = str(card["analysis"].get("continuity_judgment", "")).strip()
    return card["match_type"] == EMERGENT_MATCH_TYPE or judgment == EMERGENT_JUDGMENT


def is_shifted_content_card(card: dict[str, Any]) -> bool:
    judgment = str(card["analysis"].get("continuity_judgment", "")).strip()
    return judgment == CONTENT_SHIFT_JUDGMENT and not is_emergent_card(card)


def render_plot(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})


def build_cluster_cards(
    cluster_packages: list[dict[str, Any]],
    cluster_analyses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cards = []
    for package, analysis in zip(cluster_packages, cluster_analyses):
        analysis = normalize_cluster_analysis(analysis)
        match_type = package["match_type"]
        cards.append(
            {
                "post_cluster_label": package["post_cluster_label"],
                "match_type": match_type,
                "match_label": package["match_label"],
                "is_emergent": match_type == EMERGENT_MATCH_TYPE,
                "period_share": package["post_period_share"],
                "cluster_size": package["post_cluster_size"],
                "ticker_count": package["ticker_count"],
                "filing_count": package["filing_count"],
                "top_terms": package["top_terms"],
                "best_pre_cluster_label": package["best_pre_cluster_label"],
                "best_pre_similarity": package["best_pre_similarity"],
                "best_pre_top_terms": package["best_pre_top_terms"],
                "shared_terms": package["shared_terms"],
                "top_tickers": package["top_tickers"],
                "top_ticker_share": package.get("top_ticker_share", 0.0),
                "top_ticker_n": package.get("top_ticker_n", 0),
                "prime_share": package.get("prime_share", 0.0),
                "supplier_share": package.get("supplier_share", 0.0),
                "post_year_distribution": package.get("post_year_distribution", {}),
                "central_post_examples": package["central_post_examples"],
                "mid_post_examples": package["mid_post_examples"],
                "peripheral_post_examples": package["peripheral_post_examples"],
                "matched_pre_examples": package["matched_pre_examples"],
                "analysis": analysis,
                "continuity_label": CONTINUITY_LABELS.get(
                    analysis.get("continuity_judgment", ""),
                    str(analysis.get("continuity_judgment", "")).replace("_", " ").strip().title(),
                ),
                "count_gap_role_label": COUNT_GAP_ROLE_LABELS.get(
                    analysis.get("count_gap_role", ""),
                    str(analysis.get("count_gap_role", "")).replace("_", " ").strip().title(),
                ),
                "narration_filter": package.get("narration_filter", ""),
            }
        )
    return cards


def split_cluster_cards(
    cluster_cards: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    emergent_cards = [card for card in cluster_cards if is_emergent_card(card)]
    shifted_content_cards = [card for card in cluster_cards if is_shifted_content_card(card)]
    comparison_cards = [
        card for card in cluster_cards if not is_emergent_card(card) and not is_shifted_content_card(card)
    ]
    return emergent_cards, shifted_content_cards, comparison_cards


def cluster_card_sort_key(card: dict[str, Any]) -> tuple[Any, ...]:
    continuity = str(card["analysis"].get("continuity_judgment", "")).strip()
    count_gap_role = str(card["analysis"].get("count_gap_role", "")).strip()
    return (
        CONTINUITY_READING_PRIORITY.get(continuity, 99),
        COUNT_GAP_ROLE_PRIORITY.get(count_gap_role, 99),
        -float(card.get("period_share", 0.0)),
        -int(card.get("cluster_size", 0)),
        str(card.get("post_cluster_label", "")),
    )


def build_reading_groups(
    cluster_cards: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    finding_cards = []
    baseline_cards = []
    for card in cluster_cards:
        continuity = str(card["analysis"].get("continuity_judgment", "")).strip()
        if continuity == "largely_continuous":
            baseline_cards.append(card)
        else:
            finding_cards.append(card)
    return sorted(finding_cards, key=cluster_card_sort_key), sorted(baseline_cards, key=cluster_card_sort_key)


def emergent_cluster_figure(cluster_cards: list[dict[str, Any]], template_name: str) -> go.Figure:
    emergent_cards = [card for card in cluster_cards if is_emergent_card(card)]
    if not emergent_cards:
        fig = go.Figure()
        fig.add_annotation(
            text="No narrated clusters were ultimately judged emergent in the final synthesis.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        fig.update_layout(template=template_name, height=360, xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    plot_df = pd.DataFrame(
        {
            "period_cluster_label": [card["post_cluster_label"] for card in emergent_cards],
            "period_share": [card["period_share"] for card in emergent_cards],
            "ticker_count": [card["ticker_count"] for card in emergent_cards],
            "cluster_size": [card["cluster_size"] for card in emergent_cards],
            "filing_count": [card["filing_count"] for card in emergent_cards],
            "best_pre_similarity": [card["best_pre_similarity"] for card in emergent_cards],
            "match_type": [card["match_type"] for card in emergent_cards],
            "continuity_label": [card["continuity_label"] for card in emergent_cards],
        }
    ).sort_values(["period_share", "cluster_size"], ascending=[False, False])
    fig = px.bar(
        plot_df,
        x="period_share",
        y="period_cluster_label",
        orientation="h",
        color="ticker_count",
        template=template_name,
        title="Emergent themes after combining geometry and LLM judgment",
        labels={
            "period_share": "Share of clustered post rows",
            "period_cluster_label": "Emergent post cluster",
            "ticker_count": "Ticker count",
        },
        hover_data={
            "cluster_size": True,
            "filing_count": True,
            "match_type": True,
            "continuity_label": True,
            "best_pre_similarity": ":.2f",
        },
        color_continuous_scale=["#d8ebe7", "#1b998b", "#0f4c5c"],
    )
    fig.update_layout(height=max(360, 110 + 44 * len(plot_df)))
    return fig


def content_shift_cluster_figure(cluster_cards: list[dict[str, Any]], template_name: str) -> go.Figure:
    shifted_cards = [card for card in cluster_cards if is_shifted_content_card(card)]
    if not shifted_cards:
        fig = go.Figure()
        fig.add_annotation(
            text="No narrated clusters were judged as stable families with shifted contents.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        fig.update_layout(template=template_name, height=360, xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    plot_df = pd.DataFrame(
        {
            "period_cluster_label": [card["post_cluster_label"] for card in shifted_cards],
            "period_share": [card["period_share"] for card in shifted_cards],
            "best_pre_similarity": [card["best_pre_similarity"] for card in shifted_cards],
            "cluster_size": [card["cluster_size"] for card in shifted_cards],
            "continuity_label": [card["continuity_label"] for card in shifted_cards],
            "match_type": [card["match_type"] for card in shifted_cards],
        }
    ).sort_values(["period_share", "cluster_size"], ascending=[False, False])
    fig = px.bar(
        plot_df,
        x="period_share",
        y="period_cluster_label",
        orientation="h",
        color="best_pre_similarity",
        template=template_name,
        title="Stable cluster families with shifted post-2022 contents",
        labels={
            "period_share": "Share of clustered post rows",
            "period_cluster_label": "Shifted-content cluster",
            "best_pre_similarity": "Best pre cosine",
        },
        hover_data={"match_type": True, "continuity_label": True, "cluster_size": True},
        color_continuous_scale=["#f3e5da", "#c56b3c", "#7a4023"],
    )
    fig.update_layout(height=max(360, 110 + 44 * len(plot_df)))
    return fig


def build_selection_diagnostics(
    post_selection_df: pd.DataFrame,
    cluster_cards: list[dict[str, Any]],
    interesting_match_types: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    diagnostics = post_selection_df.copy()
    diagnostics["interesting_match_types"] = ",".join(interesting_match_types)
    diagnostics["persistent_audit_clusters"] = int(args.persistent_audit_clusters)
    card_lookup = {
        card["post_cluster_label"]: {
            "continuity_judgment": card["analysis"].get("continuity_judgment", ""),
            "continuity_label": card["continuity_label"],
            "report_section": (
                "emergent_discoveries"
                if is_emergent_card(card)
                else "shifted_contents"
                if is_shifted_content_card(card)
                else "structural_comparisons"
            ),
            "gemini_confidence": card["analysis"].get("confidence", ""),
        }
        for card in cluster_cards
    }
    diagnostics["continuity_judgment"] = diagnostics["period_cluster_label"].map(
        lambda label: card_lookup.get(str(label), {}).get("continuity_judgment", "")
    )
    diagnostics["continuity_label"] = diagnostics["period_cluster_label"].map(
        lambda label: card_lookup.get(str(label), {}).get("continuity_label", "")
    )
    diagnostics["report_section"] = diagnostics["period_cluster_label"].map(
        lambda label: card_lookup.get(str(label), {}).get("report_section", "")
    )
    diagnostics["gemini_confidence"] = diagnostics["period_cluster_label"].map(
        lambda label: card_lookup.get(str(label), {}).get("gemini_confidence", "")
    )
    keep_columns = [
        "period_cluster_label",
        "match_type",
        "match_label",
        "cluster_size",
        "period_share",
        "ticker_count",
        "filing_count",
        "top_ticker_share",
        "best_pre_cluster_label",
        "best_pre_similarity",
        "eligible_period_theme",
        "eligible_emergent_theme",
        "eligible_for_narration",
        "main_candidate",
        "main_candidate_rank",
        "selected_main",
        "persistent_audit_candidate",
        "persistent_audit_rank",
        "selected_persistent_audit",
        "selected_for_llm",
        "selection_reason",
        "narration_filter",
        "continuity_judgment",
        "continuity_label",
        "report_section",
        "gemini_confidence",
        "top_terms",
        "interesting_match_types",
        "persistent_audit_clusters",
    ]
    existing_columns = [column for column in keep_columns if column in diagnostics.columns]
    return diagnostics[existing_columns].sort_values(
        ["selected_for_llm", "main_candidate", "persistent_audit_candidate", "period_share", "cluster_size"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def render_report(
    args: argparse.Namespace,
    summary_metrics: dict[str, str],
    count_shift_summary: dict[str, Any],
    abstract_data: dict[str, Any],
    figures: dict[str, str],
    cluster_cards: list[dict[str, Any]],
    finding_cards: list[dict[str, Any]],
    baseline_cards: list[dict[str, Any]],
    emergent_cards: list[dict[str, Any]],
    shifted_content_cards: list[dict[str, Any]],
    comparison_cards: list[dict[str, Any]],
    is_full_corpus_run: bool,
    source_dataset: str,
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
        title=abstract_data.get("report_title", "Gemini-assisted period shift report"),
        subtitle="How defense-sector risk disclosure themes changed after 2022, and whether the extra post-2022 structure reflects new themes, thematic splitting, or denser disclosure language.",
        plotly_js=get_plotlyjs(),
        summary=summary_metrics,
        count_shift=count_shift_summary,
        abstract_data=abstract_data,
        figures=figures,
        cluster_cards=cluster_cards,
        finding_cards=finding_cards,
        baseline_cards=baseline_cards,
        emergent_cards=emergent_cards,
        shifted_content_cards=shifted_content_cards,
        comparison_cards=comparison_cards,
        model_name=args.model_name or os.getenv("GEMINI_MODEL", ""),
        is_full_corpus_run=is_full_corpus_run,
        source_dataset=source_dataset,
    )


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(args)
    sampled_df = inputs["sampled_df"]
    summary_df = ensure_theme_flags(inputs["summary_df"], args)
    matches_df = inputs["matches_df"]
    pairwise_df = inputs["pairwise_df"]
    representative_df = inputs["representative_df"]
    metadata = inputs["metadata"]
    dataset_path = resolve_dataset_path(args, metadata)
    sampled_embeddings_path = resolve_sampled_embeddings_path(args, metadata)
    embedding_payload = load_saved_embeddings(sampled_embeddings_path) if sampled_embeddings_path and sampled_embeddings_path.exists() else None
    full_df = pd.read_csv(dataset_path)
    full_df = full_df[full_df["comparison_window"].isin(["pre_2018_2021", "post_2022_2025"])].copy()
    full_df["text"] = full_df["text"].fillna("").astype(str).str.strip()
    full_df = full_df[full_df["text"] != ""].copy()

    interesting_match_types = parse_match_types(args.interesting_match_types)
    post_selection_df = build_post_selection_frame(
        summary_df=summary_df,
        matches_df=matches_df,
        interesting_match_types=interesting_match_types,
        max_clusters=args.max_clusters,
        args=args,
    )
    interesting_df = select_interesting_post_clusters(
        summary_df=summary_df,
        matches_df=matches_df,
        interesting_match_types=interesting_match_types,
        max_clusters=args.max_clusters,
        args=args,
    )

    embedding_model_name = resolve_embedding_model_name(args, metadata)
    cluster_packages = build_all_evidence_packages(
        interesting_df=interesting_df,
        sampled_df=sampled_df,
        representative_df=representative_df,
        embedding_model_name=embedding_model_name,
        embedding_payload=embedding_payload,
        allow_reembed=args.allow_reembed,
        args=args,
    )
    cluster_evidence_path = artifacts_dir / "llm_cluster_evidence.json"
    cluster_analysis_progress_path = artifacts_dir / "llm_cluster_analysis_progress.json"
    cluster_analysis_path = artifacts_dir / "llm_cluster_analyses.json"
    abstract_path = artifacts_dir / "llm_abstract.json"
    write_json(cluster_evidence_path, cluster_packages)

    api_key = ""
    model_name = ""
    if not args.skip_llm:
        api_key = resolve_gemini_api_key()
        model_name = resolve_gemini_model_name(args)

    report_context = build_report_context(
        full_df=full_df,
        sampled_df=sampled_df,
        summary_df=summary_df,
        interesting_df=interesting_df,
        matches_df=matches_df,
        args=args,
    )
    cluster_analyses = run_cluster_analysis(
        cluster_packages=cluster_packages,
        api_key=api_key,
        model_name=model_name,
        temperature=args.temperature,
        skip_llm=args.skip_llm,
        llm_max_concurrency=args.llm_max_concurrency,
        llm_request_stagger_seconds=args.llm_request_stagger_seconds,
        progress_path=cluster_analysis_progress_path,
        output_path=cluster_analysis_path,
        report_context=report_context,
    )
    abstract_data = run_abstract_analysis(
        cluster_packages=cluster_packages,
        cluster_analyses=cluster_analyses,
        metadata=metadata,
        report_context=report_context,
        api_key=api_key,
        model_name=model_name,
        temperature=args.abstract_temperature,
        skip_llm=args.skip_llm,
        output_path=abstract_path,
    )

    cluster_cards = build_cluster_cards(cluster_packages, cluster_analyses)
    finding_cards, baseline_cards = build_reading_groups(cluster_cards)
    emergent_cards, shifted_content_cards, comparison_cards = split_cluster_cards(cluster_cards)
    selection_diagnostics_df = build_selection_diagnostics(
        post_selection_df=post_selection_df,
        cluster_cards=cluster_cards,
        interesting_match_types=interesting_match_types,
        args=args,
    )
    selection_diagnostics_path = artifacts_dir / "llm_selection_diagnostics.csv"
    selection_diagnostics_df.to_csv(selection_diagnostics_path, index=False)
    selection_summary_path = artifacts_dir / "llm_selection_summary.json"
    selection_summary = {
        "total_post_clusters": int(len(post_selection_df)),
        "selected_for_llm": int(post_selection_df["selected_for_llm"].sum()),
        "selected_main": int(post_selection_df["selected_main"].sum()),
        "selected_persistent_audit": int(post_selection_df["selected_persistent_audit"].sum()),
        "selection_reason_counts": selection_diagnostics_df["selection_reason"].value_counts(dropna=False).to_dict(),
        "report_section_counts": {
            "emergent_discoveries": len(emergent_cards),
            "shifted_contents": len(shifted_content_cards),
            "structural_comparisons": len(comparison_cards),
        },
    }
    write_json(selection_summary_path, selection_summary)
    selection_log_lines = [
        f"Total post clusters: {selection_summary['total_post_clusters']}",
        f"Selected for LLM: {selection_summary['selected_for_llm']}",
        f"Selected via main match filter: {selection_summary['selected_main']}",
        f"Selected via persistent audit: {selection_summary['selected_persistent_audit']}",
        "",
        "Selection reasons:",
    ]
    selection_log_lines.extend(
        f"- {reason}: {count}" for reason, count in selection_summary["selection_reason_counts"].items()
    )
    selection_log_lines.extend(
        [
            "",
            "Final report sections:",
            f"- emergent_discoveries: {selection_summary['report_section_counts']['emergent_discoveries']}",
            f"- shifted_contents: {selection_summary['report_section_counts']['shifted_contents']}",
            f"- structural_comparisons: {selection_summary['report_section_counts']['structural_comparisons']}",
        ]
    )
    selection_log_path = artifacts_dir / "llm_selection_summary.txt"
    selection_log_path.write_text("\n".join(selection_log_lines) + "\n", encoding="utf-8")
    count_shift_summary = build_count_shift_summary(report_context)

    template_name = build_plotly_template()
    pre_summary_df = summary_df[summary_df["period_bucket"] == PRE_PERIOD].copy()
    post_summary_df = summary_df[summary_df["period_bucket"] == POST_PERIOD].copy()
    post_catalog_df = post_summary_df.merge(
        matches_df,
        left_on="period_cluster_label",
        right_on="post_cluster_label",
        how="left",
    )
    post_catalog_df["match_type"] = post_catalog_df["match_type"].fillna("new_post_only")
    post_catalog_df["match_label"] = post_catalog_df["match_type"].map(MATCH_TYPE_LABELS).fillna("New post-only")
    post_catalog_df["match_priority"] = post_catalog_df["match_type"].map(MATCH_TYPE_PRIORITY).fillna(0).astype(int)
    global_display_df = build_display_sample(sampled_df, 4000, 42, "period_bucket")
    pre_display_df = build_display_sample(sampled_df[sampled_df["period_bucket"] == PRE_PERIOD].copy(), 2000, 42, "period_cluster_label")
    post_display_df = build_display_sample(sampled_df[sampled_df["period_bucket"] == POST_PERIOD].copy(), 2000, 42, "period_cluster_label")
    figure_objects = {
        "cluster_count_shift": cluster_count_shift_figure(summary_df, template_name),
        "period_text_density": period_text_density_figure(sampled_df, template_name),
        "corpus_overview": corpus_overview_figure(full_df, template_name),
        "sample_mix": sample_mix_figure(sampled_df, template_name),
        "shared_umap_period": shared_umap_period_figure(global_display_df, template_name),
        "pre_cluster_space": period_cluster_space_figure(pre_display_df, PRE_PERIOD, template_name),
        "post_cluster_space": period_cluster_space_figure(post_display_df, POST_PERIOD, template_name),
        "pre_cluster_share": period_cluster_share_figure(pre_summary_df, PRE_PERIOD, 10, template_name),
        "post_cluster_share": period_cluster_share_figure(post_summary_df, POST_PERIOD, 10, template_name),
        "match_heatmap": match_heatmap_figure(pairwise_df, pre_summary_df, post_summary_df, 10, template_name),
        "post_match_status": post_match_status_figure(post_catalog_df, 10, template_name),
        "emergent_clusters": emergent_cluster_figure(cluster_cards, template_name),
        "content_shift_clusters": content_shift_cluster_figure(cluster_cards, template_name),
        "interesting_clusters": interesting_cluster_figure(interesting_df, template_name),
        "analysis_confidence": confidence_figure(cluster_cards, template_name),
    }
    figures = {key: render_plot(fig) for key, fig in figure_objects.items()}

    summary_metrics = {
        "interesting_clusters": f"{len(cluster_cards):,}",
        "emergent_discoveries": f"{len(emergent_cards):,}",
        "content_shifts": f"{len(shifted_content_cards):,}",
        "structural_shifts": f"{len(comparison_cards):,}",
        "cluster_gap": f"{count_shift_summary['cluster_gap']:+d}",
        "match_types": ", ".join(interesting_match_types),
        "embedding_model": embedding_model_name,
        "llm_model": model_name or "skipped",
        "clustered_rows": f"{len(sampled_df):,}",
        "companies": f"{full_df['ticker'].nunique():,}",
        "pre_clusters": f"{int((pre_summary_df['period_cluster'] != -1).sum())}",
        "post_clusters": f"{int((post_summary_df['period_cluster'] != -1).sum())}",
    }

    write_json(cluster_analysis_path, cluster_analyses)
    write_json(abstract_path, abstract_data)

    html = render_report(
        args=args,
        summary_metrics=summary_metrics,
        count_shift_summary=count_shift_summary,
        abstract_data=abstract_data,
        figures=figures,
        cluster_cards=cluster_cards,
        finding_cards=finding_cards,
        baseline_cards=baseline_cards,
        emergent_cards=emergent_cards,
        shifted_content_cards=shifted_content_cards,
        comparison_cards=comparison_cards,
        is_full_corpus_run=bool(report_context.get("is_full_corpus_run", False)),
        source_dataset=str(dataset_path.as_posix()),
    )
    output_html.write_text(html, encoding="utf-8")

    metadata_out = {
        "output_html": str(output_html),
        "sampled_rows": args.sampled_rows,
        "dataset": str(dataset_path),
        "period_cluster_summary": args.period_cluster_summary,
        "pairwise_similarities": args.pairwise_similarities,
        "cluster_matches": args.cluster_matches,
        "representative_examples": args.representative_examples,
        "sampled_embeddings": str(sampled_embeddings_path) if sampled_embeddings_path else "",
        "embedding_model_name": embedding_model_name,
        "llm_model_name": model_name or "",
        "interesting_match_types": interesting_match_types,
        "selection_diagnostics": str(selection_diagnostics_path),
        "selection_summary": str(selection_summary_path),
        "emergent_min_cluster_size": args.emergent_min_cluster_size,
        "emergent_min_ticker_count": args.emergent_min_ticker_count,
        "emergent_min_filing_count": args.emergent_min_filing_count,
        "emergent_max_top_ticker_share": args.emergent_max_top_ticker_share,
        "persistent_audit_clusters": args.persistent_audit_clusters,
        "skip_llm": bool(args.skip_llm),
        "allow_reembed": bool(args.allow_reembed),
        "narrated_clusters": len(cluster_cards),
    }
    write_json(artifacts_dir / "llm_report_metadata.json", metadata_out)

    print(f"Rendered HTML report to: {output_html}")
    print(f"Selection diagnostics written to: {selection_diagnostics_path}")
    print(f"Selection summary written to: {selection_summary_path}")
    print(f"Selection log written to: {selection_log_path}")
    print(f"Artifacts written to: {artifacts_dir}")


if __name__ == "__main__":
    main()
