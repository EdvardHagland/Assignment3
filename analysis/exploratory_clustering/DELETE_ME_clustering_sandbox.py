"""
Disposable clustering sandbox.

This file is intentionally named DELETE_ME so we can experiment freely and
remove it later without worrying about the final repo structure.
"""

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_PATH = Path("data/final/sec_defense_risk_dataset.csv")
RANDOM_STATE = 42
SAMPLE_PER_PERIOD = 1000
N_CLUSTERS = 10


def load_sample() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    sampled = (
        df.groupby("period_bucket", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), SAMPLE_PER_PERIOD), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )
    return sampled


def run_tfidf_kmeans(df: pd.DataFrame) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(df["text"])
    model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
    df = df.copy()
    df["cluster"] = model.fit_predict(matrix)
    return df


def compare_clusters(df: pd.DataFrame) -> pd.DataFrame:
    cluster_counts = (
        df.groupby(["period_bucket", "cluster"])
        .size()
        .reset_index(name="n")
    )
    cluster_counts["share"] = (
        cluster_counts.groupby("period_bucket")["n"]
        .transform(lambda x: x / x.sum())
    )
    return (
        cluster_counts.pivot(index="cluster", columns="period_bucket", values="share")
        .fillna(0)
        .assign(delta=lambda x: x.get("post_2022", 0) - x.get("pre_2022", 0))
        .sort_values("delta", ascending=False)
    )


if __name__ == "__main__":
    sample = load_sample()
    clustered = run_tfidf_kmeans(sample)
    summary = compare_clusters(clustered)

    print("Sample shape:", sample.shape)
    print()
    print(summary)
