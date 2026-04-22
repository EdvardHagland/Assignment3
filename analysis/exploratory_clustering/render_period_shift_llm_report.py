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
import json
import os
import textwrap
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
    MATCH_TYPE_LABELS,
    MATCH_TYPE_PRIORITY,
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
        "--max-clusters",
        type=int,
        default=6,
        help="Maximum number of interesting post clusters to send to Gemini.",
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


def select_interesting_post_clusters(
    summary_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    interesting_match_types: list[str],
    max_clusters: int,
) -> pd.DataFrame:
    post_summary = summary_df[summary_df["period_bucket"] == POST_PERIOD].copy()
    merged = post_summary.merge(
        matches_df,
        left_on="period_cluster_label",
        right_on="post_cluster_label",
        how="left",
    )
    merged["match_type"] = merged["match_type"].fillna("new_post_only")
    merged["match_label"] = merged["match_label"].fillna("New post-only")
    merged["match_priority"] = merged["match_priority"].fillna(0)

    interesting = merged[
        (merged["period_cluster"] != -1)
        & (merged["eligible_period_theme"].fillna(False))
        & (merged["match_type"].isin(interesting_match_types))
    ].copy()

    interesting = interesting.sort_values(
        ["match_priority", "period_share", "cluster_size", "best_pre_similarity"],
        ascending=[True, False, False, False],
    ).head(max_clusters)
    return interesting.reset_index(drop=True)


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
        "prime_share": float(cluster_row.get("prime", 0.0)),
        "supplier_share": float(cluster_row.get("supplier", 0.0)),
        "best_pre_cluster_label": pre_label,
        "best_pre_similarity": float(cluster_row.get("best_pre_similarity", np.nan)),
        "best_pre_top_terms": str(cluster_row.get("best_pre_top_terms", "")),
        "shared_terms": str(cluster_row.get("shared_terms", "")),
        "top_pre_candidates": str(cluster_row.get("top_pre_candidates", "")),
        "central_post_examples": format_example_rows(central_post_rows, f"{post_label}_core"),
        "mid_post_examples": format_example_rows(mid_post_rows, f"{post_label}_mid"),
        "peripheral_post_examples": format_example_rows(peripheral_post_rows, f"{post_label}_edge"),
        "matched_pre_examples": format_example_rows(pre_rows, f"{pre_label or 'pre'}_match"),
    }
    return packaged


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
        "required": ["report_title", "abstract", "major_findings", "limitations", "closing_caution"],
        "properties": {
            "report_title": {"type": "string"},
            "abstract": {"type": "string"},
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


def build_cluster_prompt(cluster_package: dict[str, Any]) -> str:
    payload = json.dumps(cluster_package, indent=2, ensure_ascii=False)
    return textwrap.dedent(
        f"""
        You are helping with an academic exploratory analysis of U.S. defense-sector SEC 10-K Item 1A risk factor disclosures.

        Critical genre context:
        - These texts are risk disclosures from annual filings, not neutral event logs.
        - They are shaped by legal, strategic, and investor-relations incentives.
        - Boilerplate and broad cautionary language are common.
        - You should describe shifts in disclosure emphasis or thematic structure, not claim direct changes in real-world risk incidence unless the text package clearly supports it.
        - Be especially cautious when the evidence is concentrated in a small number of firms or looks like routine filing boilerplate.

        Your task:
        - Analyze one interesting post-2022 cluster and its best pre-2022 comparison context.
        - Focus on why this cluster was flagged as interesting.
        - Use the central examples to understand the core of the cluster.
        - Use the mid-distance examples to see whether the theme still holds once you move away from the centroid.
        - Use the peripheral examples to test how broad the cluster still is near its outer edge.
        - Use the matched pre examples to judge whether this is genuinely new, more specific, or simply a renamed variant of an earlier theme.
        - Be critical and restrained. If the evidence is weak or mixed, say so.

        Return JSON only, matching the schema exactly.

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
    payload = {
        "analysis_context": {
            "corpus": "U.S. defense-sector SEC 10-K Item 1A risk factor disclosures",
            "comparison": "2018-2021 versus 2022-2025",
            "method": "same embeddings for all rows, separate pre/post cluster discovery, approximate post-to-pre matching",
            "embedding_model": metadata.get("model_name", ""),
        },
        "report_context": report_context,
        "cluster_packages": cluster_packages,
        "cluster_analyses": cluster_analyses,
    }
    return textwrap.dedent(
        f"""
        You are writing the executive abstract for an exploratory report on defense-sector SEC 10-K Item 1A risk disclosures.

        Important framing:
        - This is disclosure analysis, not direct measurement of realized external risk.
        - The data come from legally cautious corporate filings, so boilerplate and strategic emphasis matter.
        - You should summarize only the strongest major findings supported by the supplied cluster analyses.
        - Do not claim one-to-one cluster identity across periods.
        - Use careful language such as "post-2022 disclosures appear to..." or "the post period shows a more explicit subtheme around..."
        - The abstract should synthesize the major findings across the interesting changed clusters, not repeat every detail.
        - Mention limitations explicitly.

        Return JSON only, matching the schema exactly.

        Input:
        {json.dumps(payload, indent=2, ensure_ascii=False)}
        """
    ).strip()


def build_report_context(
    full_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    interesting_df: pd.DataFrame,
    matches_df: pd.DataFrame,
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

    return {
        "full_rows": int(len(full_df)),
        "companies": int(full_df["ticker"].nunique()),
        "pre_cluster_count": int((pre_summary["period_cluster"] != -1).sum()),
        "post_cluster_count": int((post_summary["period_cluster"] != -1).sum()),
        "interesting_cluster_count": int(len(interesting_df)),
        "interesting_cluster_rows": interesting_rows,
        "top_pre_clusters": compact_rows(pre_summary, ["period_cluster_label", "period_share", "cluster_size", "ticker_count", "filing_count", "top_terms"]),
        "top_post_clusters": compact_rows(post_summary, ["period_cluster_label", "period_share", "cluster_size", "ticker_count", "filing_count", "top_terms"]),
        "match_type_counts": matches_df["match_type"].fillna("missing").value_counts().to_dict() if not matches_df.empty else {},
    }


def run_cluster_analysis(
    cluster_packages: list[dict[str, Any]],
    api_key: str,
    model_name: str,
    temperature: float,
    skip_llm: bool,
    progress_path: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    saved_progress: dict[str, dict[str, Any]] = {}
    if progress_path.exists():
        try:
            rows = read_json(progress_path)
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and "post_cluster_label" in row and "analysis" in row:
                        saved_progress[str(row["post_cluster_label"])] = row["analysis"]
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
        write_json(progress_path, progress_rows)
        write_json(output_path, output_rows)

    analyses = []
    for package in cluster_packages:
        cluster_label = str(package["post_cluster_label"])
        if cluster_label in saved_progress:
            analyses.append(saved_progress[cluster_label])
            continue

        if skip_llm:
            analysis = {
                "card_title": package["post_cluster_label"],
                "headline": "LLM step skipped.",
                "why_interesting": f"{package['match_label']} cluster selected for narrative review.",
                "interpretation": "No Gemini interpretation was generated because --skip-llm was used.",
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
            saved_progress[cluster_label] = analysis
            analyses.append(analysis)
            persist_progress()
            continue

        prompt = build_cluster_prompt(package)
        analysis = call_gemini_json(
            api_key=api_key,
            model_name=model_name,
            prompt=prompt,
            schema=gemini_json_schema_cluster(),
            temperature=temperature,
        )
        saved_progress[cluster_label] = analysis
        analyses.append(analysis)
        persist_progress()
    return analyses


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
            if isinstance(payload, dict) and payload:
                return payload
        except Exception:
            pass

    if skip_llm:
        abstract = {
            "report_title": "LLM abstract skipped",
            "abstract": "The evidence package was built successfully, but Gemini was not called because --skip-llm was used.",
            "major_findings": [
                "Interesting post clusters were selected from the period-shift artifacts.",
                "Each selected cluster includes centroid-near and moderately peripheral evidence.",
                "Matched pre-cluster context is packaged for comparison.",
            ],
            "limitations": [
                "No model-written interpretation is included.",
                "This summary is a placeholder rather than an analytical abstract.",
            ],
            "closing_caution": "Run without --skip-llm to generate the actual narrative synthesis.",
        }
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
    write_json(output_path, abstract)
    return abstract


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
        title="Interesting post-2022 clusters selected for Gemini review",
        labels={"period_share": "Share of sampled post rows", "period_cluster_label": "Post cluster", "match_type": "Match type"},
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


def render_plot(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})


def build_cluster_cards(
    cluster_packages: list[dict[str, Any]],
    cluster_analyses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cards = []
    for package, analysis in zip(cluster_packages, cluster_analyses):
        cards.append(
            {
                "post_cluster_label": package["post_cluster_label"],
                "match_type": package["match_type"],
                "match_label": package["match_label"],
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
                "central_post_examples": package["central_post_examples"],
                "mid_post_examples": package["mid_post_examples"],
                "peripheral_post_examples": package["peripheral_post_examples"],
                "matched_pre_examples": package["matched_pre_examples"],
                "analysis": analysis,
            }
        )
    return cards


def render_report(
    args: argparse.Namespace,
    summary_metrics: dict[str, str],
    abstract_data: dict[str, Any],
    figures: dict[str, str],
    cluster_cards: list[dict[str, Any]],
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
        subtitle="Narrative synthesis on top of separate pre/post cluster discovery",
        plotly_js=get_plotlyjs(),
        summary=summary_metrics,
        abstract_data=abstract_data,
        figures=figures,
        cluster_cards=cluster_cards,
        model_name=args.model_name or os.getenv("GEMINI_MODEL", ""),
        generated_from=str(Path(args.sampled_rows).as_posix()),
    )


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(args)
    sampled_df = inputs["sampled_df"]
    summary_df = inputs["summary_df"]
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
    interesting_df = select_interesting_post_clusters(
        summary_df=summary_df,
        matches_df=matches_df,
        interesting_match_types=interesting_match_types,
        max_clusters=args.max_clusters,
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

    cluster_analyses = run_cluster_analysis(
        cluster_packages=cluster_packages,
        api_key=api_key,
        model_name=model_name,
        temperature=args.temperature,
        skip_llm=args.skip_llm,
        progress_path=cluster_analysis_progress_path,
        output_path=cluster_analysis_path,
    )
    report_context = build_report_context(
        full_df=full_df,
        summary_df=summary_df,
        interesting_df=interesting_df,
        matches_df=matches_df,
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
        "corpus_overview": corpus_overview_figure(full_df, template_name),
        "sample_mix": sample_mix_figure(sampled_df, template_name),
        "shared_umap_period": shared_umap_period_figure(global_display_df, template_name),
        "pre_cluster_space": period_cluster_space_figure(pre_display_df, PRE_PERIOD, template_name),
        "post_cluster_space": period_cluster_space_figure(post_display_df, POST_PERIOD, template_name),
        "pre_cluster_share": period_cluster_share_figure(pre_summary_df, PRE_PERIOD, 10, template_name),
        "post_cluster_share": period_cluster_share_figure(post_summary_df, POST_PERIOD, 10, template_name),
        "match_heatmap": match_heatmap_figure(pairwise_df, pre_summary_df, post_summary_df, 10, template_name),
        "post_match_status": post_match_status_figure(post_catalog_df, 10, template_name),
        "interesting_clusters": interesting_cluster_figure(interesting_df, template_name),
        "analysis_confidence": confidence_figure(cluster_cards, template_name),
    }
    figures = {key: render_plot(fig) for key, fig in figure_objects.items()}

    summary_metrics = {
        "interesting_clusters": f"{len(cluster_cards):,}",
        "match_types": ", ".join(interesting_match_types),
        "embedding_model": embedding_model_name,
        "llm_model": model_name or "skipped",
        "sample_rows": f"{len(sampled_df):,}",
        "companies": f"{full_df['ticker'].nunique():,}",
        "pre_clusters": f"{int((pre_summary_df['period_cluster'] != -1).sum())}",
        "post_clusters": f"{int((post_summary_df['period_cluster'] != -1).sum())}",
    }

    write_json(cluster_analysis_path, cluster_analyses)
    write_json(abstract_path, abstract_data)

    html = render_report(
        args=args,
        summary_metrics=summary_metrics,
        abstract_data=abstract_data,
        figures=figures,
        cluster_cards=cluster_cards,
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
        "skip_llm": bool(args.skip_llm),
        "allow_reembed": bool(args.allow_reembed),
        "narrated_clusters": len(cluster_cards),
    }
    write_json(artifacts_dir / "llm_report_metadata.json", metadata_out)

    print(f"Rendered HTML report to: {output_html}")
    print(f"Artifacts written to: {artifacts_dir}")


if __name__ == "__main__":
    main()
