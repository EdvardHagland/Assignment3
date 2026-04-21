#!/usr/bin/env python3
'''
CPU-only post-processing for exploratory clustering artifacts.

This pass is meant to surface broad, sector-level shifts that can be hidden by
large semantic clusters and company-specific event clusters. It re-ranks saved
clusters by relative movement, penalizes concentration in a single ticker, and
optionally computes within-cluster pre/post contrast terms when the saved
sampled rows are available.
'''

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

PRE_PERIOD = 'pre_2022'
POST_PERIOD = 'post_2022'

GENERIC_STOPWORDS = {
    'risk', 'risks', 'factor', 'factors', 'company', 'companies', 'business', 'businesses',
    'may', 'could', 'would', 'also', 'us', 'our', 'we', 'including', 'certain', 'significant',
    'subject', 'related', 'current', 'future', 'material', 'adversely', 'affect', 'affected',
    'adverse', 'operations', 'operating', 'results', 'condition', 'financial', 'ability',
    'continue', 'within', 'among', 'item', 'sec', 'filing',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Render a CPU-only diagnostics pass for saved clustering artifacts.')
    parser.add_argument('--cluster-summary', default='analysis/exploratory_clustering/output/cluster_summary.csv', help='Path to cluster_summary.csv produced by render_exploratory_report.py.')
    parser.add_argument('--sampled-rows', default='analysis/exploratory_clustering/output/sampled_cluster_rows.csv', help='Optional path to sampled_cluster_rows.csv for within-cluster pre/post contrasts.')
    parser.add_argument('--representative-examples', default='analysis/exploratory_clustering/output/representative_examples.csv', help='Optional path to representative_examples.csv for example snippets.')
    parser.add_argument('--output-html', default='analysis/exploratory_clustering/output/cluster_shift_diagnostics.html', help='Path to the rendered diagnostics HTML.')
    parser.add_argument('--output-csv', default='analysis/exploratory_clustering/output/cluster_shift_diagnostics.csv', help='Path to the enriched cluster diagnostics CSV.')
    parser.add_argument('--output-contrast-csv', default='analysis/exploratory_clustering/output/cluster_shift_contrasts.csv', help='Path to the cluster contrast CSV.')
    parser.add_argument('--top-clusters', type=int, default=12, help='How many eligible clusters to foreground in the diagnostics output.')
    parser.add_argument('--top-terms', type=int, default=8, help='How many contrastive terms to surface per side within a cluster.')
    parser.add_argument('--min-cluster-size', type=int, default=150, help='Minimum cluster size for sector-level eligibility.')
    parser.add_argument('--min-ticker-count', type=int, default=10, help='Minimum distinct ticker count for sector-level eligibility.')
    parser.add_argument('--max-top-ticker-share', type=float, default=0.35, help='Maximum allowable share of a cluster held by its dominant ticker.')
    return parser.parse_args()


def parse_top_ticker_n(top_tickers: str) -> int:
    match = re.search(r'\((\d+)\)', str(top_tickers))
    return int(match.group(1)) if match else 0


def load_cluster_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'cluster', 'cluster_label', 'cluster_size', PRE_PERIOD, POST_PERIOD, 'post_minus_pre', 'ticker_count', 'top_tickers', 'top_terms'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'cluster summary is missing required columns: {sorted(missing)}')

    df = df.copy()
    df['abs_delta'] = df['post_minus_pre'].abs()
    df['top_ticker_n'] = df['top_tickers'].map(parse_top_ticker_n)
    df['top_ticker_share'] = np.where(df['cluster_size'] > 0, df['top_ticker_n'] / df['cluster_size'], np.nan)
    df['shift_direction'] = np.where(df['post_minus_pre'] >= 0, 'post_2022_up', 'pre_2022_up')
    df['sector_signal_score'] = (
        df['abs_delta']
        * np.log1p(df['cluster_size'])
        * np.log1p(df['ticker_count'].clip(lower=1))
        * (1 - df['top_ticker_share'].fillna(1).clip(lower=0, upper=0.95))
    )
    return df


def maybe_load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def enrich_with_sampled_rows(summary_df: pd.DataFrame, sampled_df: pd.DataFrame | None) -> pd.DataFrame:
    if sampled_df is None:
        return summary_df

    required = {'cluster', 'ticker', 'text', 'period_bucket'}
    missing = required.difference(sampled_df.columns)
    if missing:
        raise ValueError(f'sampled rows file is missing required columns: {sorted(missing)}')

    cluster_sizes = sampled_df.groupby('cluster').size().rename('sampled_cluster_size')
    top_counts = sampled_df.groupby(['cluster', 'ticker']).size().groupby('cluster').max().rename('top_ticker_n_exact')
    ticker_counts = sampled_df.groupby('cluster')['ticker'].nunique().rename('ticker_count_exact')

    enriched = summary_df.merge(cluster_sizes, on='cluster', how='left')
    enriched = enriched.merge(top_counts, on='cluster', how='left')
    enriched = enriched.merge(ticker_counts, on='cluster', how='left')
    enriched['sampled_cluster_size'] = enriched['sampled_cluster_size'].fillna(enriched['cluster_size'])
    enriched['ticker_count'] = enriched['ticker_count_exact'].fillna(enriched['ticker_count'])
    enriched['top_ticker_n'] = enriched['top_ticker_n_exact'].fillna(enriched['top_ticker_n'])
    enriched['top_ticker_share'] = np.where(
        enriched['sampled_cluster_size'] > 0,
        enriched['top_ticker_n'] / enriched['sampled_cluster_size'],
        enriched['top_ticker_share'],
    )
    enriched['sector_signal_score'] = (
        enriched['abs_delta']
        * np.log1p(enriched['cluster_size'])
        * np.log1p(enriched['ticker_count'].clip(lower=1))
        * (1 - enriched['top_ticker_share'].fillna(1).clip(lower=0, upper=0.95))
    )
    return enriched.drop(columns=[c for c in ['sampled_cluster_size', 'top_ticker_n_exact', 'ticker_count_exact'] if c in enriched.columns])


def apply_sector_filters(summary_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = summary_df.copy()
    df['eligible_sector_shift'] = (
        (df['cluster'] != -1)
        & (df['cluster_size'] >= args.min_cluster_size)
        & (df['ticker_count'] >= args.min_ticker_count)
        & (df['top_ticker_share'] <= args.max_top_ticker_share)
    )
    df['exclusion_reason'] = ''
    df.loc[df['cluster'] == -1, 'exclusion_reason'] = 'noise'
    df.loc[(df['cluster'] != -1) & (df['cluster_size'] < args.min_cluster_size), 'exclusion_reason'] = 'too_small'
    df.loc[(df['cluster'] != -1) & (df['ticker_count'] < args.min_ticker_count), 'exclusion_reason'] = 'too_few_tickers'
    df.loc[(df['cluster'] != -1) & (df['top_ticker_share'] > args.max_top_ticker_share), 'exclusion_reason'] = 'too_concentrated'
    df.loc[df['eligible_sector_shift'], 'exclusion_reason'] = ''
    return df


def distinct_terms_for_cluster(cluster_rows: pd.DataFrame, top_terms: int) -> tuple[str, str]:
    pre_rows = cluster_rows[cluster_rows['period_bucket'] == PRE_PERIOD]['text'].dropna().astype(str)
    post_rows = cluster_rows[cluster_rows['period_bucket'] == POST_PERIOD]['text'].dropna().astype(str)
    if len(pre_rows) < 5 or len(post_rows) < 5:
        return '', ''

    stopwords = sorted(set(ENGLISH_STOP_WORDS).union(GENERIC_STOPWORDS))
    vectorizer = CountVectorizer(
        stop_words=stopwords,
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
        token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z\-]+\b',
    )

    texts = pd.concat([pre_rows, post_rows], ignore_index=True)
    matrix = vectorizer.fit_transform(texts)
    if matrix.shape[1] == 0:
        return '', ''

    pre_matrix = matrix[: len(pre_rows)]
    post_matrix = matrix[len(pre_rows):]
    pre_counts = np.asarray(pre_matrix.sum(axis=0)).ravel()
    post_counts = np.asarray(post_matrix.sum(axis=0)).ravel()
    vocab = np.asarray(vectorizer.get_feature_names_out())

    pre_rate = (pre_counts + 1) / (pre_counts.sum() + len(vocab))
    post_rate = (post_counts + 1) / (post_counts.sum() + len(vocab))
    delta = post_rate - pre_rate

    post_rank = np.argsort(delta)[::-1]
    pre_rank = np.argsort(delta)
    top_post = [vocab[i] for i in post_rank if delta[i] > 0][:top_terms]
    top_pre = [vocab[i] for i in pre_rank if delta[i] < 0][:top_terms]
    return ', '.join(top_post), ', '.join(top_pre)


def build_contrast_rows(summary_df: pd.DataFrame, sampled_df: pd.DataFrame | None, representative_df: pd.DataFrame | None, args: argparse.Namespace) -> pd.DataFrame:
    if sampled_df is None:
        return pd.DataFrame()

    eligible = (
        summary_df[summary_df['eligible_sector_shift']]
        .sort_values(['sector_signal_score', 'abs_delta'], ascending=[False, False])
        .head(args.top_clusters)
    )

    representative_lookup: dict[int, list[str]] = {}
    if representative_df is not None and {'cluster', 'text', 'ticker', 'filing_year'}.issubset(representative_df.columns):
        for cluster_id, part in representative_df.groupby('cluster'):
            representative_lookup[int(cluster_id)] = [
                f"{row['ticker']} {int(row['filing_year'])}: {str(row['text'])[:220]}"
                for _, row in part.head(2).iterrows()
            ]

    rows: list[dict] = []
    for _, cluster in eligible.iterrows():
        cluster_rows = sampled_df[sampled_df['cluster'] == cluster['cluster']].copy()
        post_terms, pre_terms = distinct_terms_for_cluster(cluster_rows, args.top_terms)
        rows.append(
            {
                'cluster': int(cluster['cluster']),
                'cluster_label': cluster['cluster_label'],
                'cluster_size': int(cluster['cluster_size']),
                'ticker_count': int(cluster['ticker_count']),
                'top_ticker_share': float(cluster['top_ticker_share']),
                'post_minus_pre': float(cluster['post_minus_pre']),
                'sector_signal_score': float(cluster['sector_signal_score']),
                'top_terms': cluster['top_terms'],
                'top_post_terms': post_terms,
                'top_pre_terms': pre_terms,
                'representative_examples': ' || '.join(representative_lookup.get(int(cluster['cluster']), [])),
            }
        )
    return pd.DataFrame(rows)


def build_html(summary_df: pd.DataFrame, contrast_df: pd.DataFrame, args: argparse.Namespace, used_sampled_rows: bool) -> str:
    eligible = (
        summary_df[summary_df['eligible_sector_shift']]
        .sort_values(['sector_signal_score', 'abs_delta'], ascending=[False, False])
        .head(args.top_clusters)
        .copy()
    )
    excluded = (
        summary_df[(summary_df['cluster'] != -1) & (~summary_df['eligible_sector_shift'])]
        .sort_values(['abs_delta', 'cluster_size'], ascending=[False, False])
        .head(args.top_clusters)
        .copy()
    )

    for frame in [eligible, excluded, contrast_df]:
        if not frame.empty:
            for col in [PRE_PERIOD, POST_PERIOD, 'post_minus_pre', 'top_ticker_share']:
                if col in frame.columns:
                    frame[col] = [f'{value:.1%}' for value in frame[col]]
            if 'sector_signal_score' in frame.columns:
                frame['sector_signal_score'] = [f'{value:.4f}' for value in frame['sector_signal_score']]

    summary_note = (
        'Within-cluster contrast terms are included because sampled_cluster_rows.csv was available.'
        if used_sampled_rows
        else 'Within-cluster contrast terms are not included because sampled_cluster_rows.csv was not available for this run.'
    )

    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cluster Shift Diagnostics</title>
  <style>
    :root {{
      --paper: #f7f2e9;
      --panel: #fffdfa;
      --ink: #15232d;
      --muted: #5f6b73;
      --accent: #0f4c5c;
      --line: #deceb7;
    }}
    body {{ margin: 0; background: var(--paper); color: var(--ink); font: 16px/1.55 "Segoe UI", "Helvetica Neue", Arial, sans-serif; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 40px 28px 64px; }}
    h1, h2 {{ font-family: Georgia, "Times New Roman", serif; margin: 0 0 12px; }}
    h1 {{ font-size: 2.3rem; }}
    h2 {{ font-size: 1.5rem; margin-top: 32px; }}
    p {{ margin: 0 0 14px; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 24px 0; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 16px 18px; box-shadow: 0 8px 28px rgba(21, 35, 45, 0.05); }}
    .metric {{ font-size: 1.65rem; font-weight: 700; color: var(--accent); }}
    .label {{ color: var(--muted); font-size: 0.92rem; }}
    .table-wrap {{ background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 12px; overflow-x: auto; box-shadow: 0 8px 28px rgba(21, 35, 45, 0.05); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.95rem; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #eadfce; text-align: left; vertical-align: top; }}
    th {{ background: #f4ece0; font-weight: 700; }}
    tr:last-child td {{ border-bottom: none; }}
    .note {{ border-left: 4px solid var(--accent); padding-left: 14px; margin: 20px 0; }}
    code {{ background: #f1e9dc; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <main>
    <h1>Cluster Shift Diagnostics</h1>
    <p>This CPU-only pass re-ranks saved exploratory clusters by relative movement while penalizing firm-specific concentration. The goal is to foreground sector-wide changes rather than large but boilerplate or single-company clusters.</p>
    <div class="grid">
      <div class="card"><div class="metric">{int((summary_df['cluster'] != -1).sum())}</div><div class="label">non-noise clusters reviewed</div></div>
      <div class="card"><div class="metric">{int(summary_df['eligible_sector_shift'].sum())}</div><div class="label">eligible broad shift clusters</div></div>
      <div class="card"><div class="metric">{args.min_cluster_size}</div><div class="label">minimum cluster size</div></div>
      <div class="card"><div class="metric">{args.min_ticker_count}</div><div class="label">minimum ticker count</div></div>
      <div class="card"><div class="metric">{args.max_top_ticker_share:.0%}</div><div class="label">maximum top ticker share</div></div>
    </div>
    <div class="note">
      <p>{html.escape(summary_note)}</p>
      <p>The strongest signal here is <strong>relative change within broad clusters</strong>, not just cluster size. That is why the ranking uses <code>abs(post_minus_pre)</code> with penalties for narrow ticker concentration.</p>
    </div>
    <h2>Broad sector-level shift candidates</h2>
    <div class="table-wrap">{eligible[['cluster_label','cluster_size','ticker_count','top_ticker_share','post_minus_pre','sector_signal_score','top_terms']].to_html(index=False, escape=False)}</div>
    <h2>Potentially interesting but excluded clusters</h2>
    <div class="table-wrap">{excluded[['cluster_label','cluster_size','ticker_count','top_ticker_share','post_minus_pre','exclusion_reason','top_terms','top_tickers']].to_html(index=False, escape=False)}</div>
    <h2>Within-cluster pre/post contrasts</h2>
    <div class="table-wrap">{contrast_df.to_html(index=False, escape=False) if not contrast_df.empty else '<p>No contrast rows were generated for this run.</p>'}</div>
  </main>
</body>
</html>
'''


def main() -> None:
    args = parse_args()
    cluster_summary_path = Path(args.cluster_summary)
    sampled_rows_path = Path(args.sampled_rows)
    representative_examples_path = Path(args.representative_examples)
    output_html_path = Path(args.output_html)
    output_csv_path = Path(args.output_csv)
    output_contrast_csv_path = Path(args.output_contrast_csv)

    summary_df = load_cluster_summary(cluster_summary_path)
    sampled_df = maybe_load_csv(sampled_rows_path)
    representative_df = maybe_load_csv(representative_examples_path)

    summary_df = enrich_with_sampled_rows(summary_df, sampled_df)
    summary_df = apply_sector_filters(summary_df, args)
    summary_df = summary_df.sort_values(['eligible_sector_shift', 'sector_signal_score', 'abs_delta'], ascending=[False, False, False]).reset_index(drop=True)

    contrast_df = build_contrast_rows(summary_df, sampled_df, representative_df, args)

    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_contrast_csv_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(output_csv_path, index=False)
    if not contrast_df.empty:
        contrast_df.to_csv(output_contrast_csv_path, index=False)
    elif output_contrast_csv_path.exists():
        output_contrast_csv_path.unlink()

    html_text = build_html(summary_df, contrast_df, args, used_sampled_rows=sampled_df is not None)
    output_html_path.write_text(html_text, encoding='utf-8')

    print(f'Diagnostics HTML written to: {output_html_path}')
    print(f'Diagnostics CSV written to: {output_csv_path}')
    if sampled_df is not None:
        print(f'Contrast CSV written to: {output_contrast_csv_path}')
    else:
        print('Contrast CSV skipped because sampled_cluster_rows.csv was not available.')


if __name__ == '__main__':
    main()
