from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "netflix_titles.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
VISUALS_DIR = ROOT / "visuals"
REPORTS_DIR = ROOT / "reports"
SUMMARY_PATH = PROCESSED_DIR / "summary_metrics.json"
TESTS_PATH = PROCESSED_DIR / "statistical_tests.json"
MODEL_PATH = PROCESSED_DIR / "model_metrics.json"
CLUSTERS_PATH = PROCESSED_DIR / "content_clusters.csv"
COUNTRY_PATH = PROCESSED_DIR / "country_catalog.csv"
GENRE_PATH = PROCESSED_DIR / "genre_catalog.csv"
EXEC_SUMMARY_PATH = REPORTS_DIR / "executive_summary.md"

logger = logging.getLogger(__name__)


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
    (ROOT / ".mplconfig").mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the streaming content intelligence pipeline.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH, help="Path to the Netflix titles dataset.")
    return parser.parse_args()


def apply_theme() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (12, 7),
            "figure.dpi": 160,
            "axes.titlesize": 18,
            "axes.labelsize": 12,
            "axes.titleweight": "bold",
            "savefig.bbox": "tight",
        }
    )


def load_and_clean(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["rating"] = df["rating"].fillna("Unknown")
    df["country"] = df["country"].fillna("Unknown")
    df["director"] = df["director"].fillna("Unknown")
    df["cast"] = df["cast"].fillna("Unknown")
    df["duration"] = df["duration"].fillna("Unknown")
    df["date_added_year"] = df["date_added"].dt.year
    df["date_added_month"] = df["date_added"].dt.to_period("M").dt.to_timestamp()
    df["primary_country"] = df["country"].str.split(",").str[0].str.strip()
    df["primary_genre"] = df["listed_in"].str.split(",").str[0].str.strip()
    df["cast_size"] = df["cast"].apply(lambda x: 0 if x == "Unknown" else len([c for c in x.split(",") if c.strip()]))
    df["description_length"] = df["description"].fillna("").str.split().str.len()
    df["content_age_at_add"] = np.where(df["date_added_year"].notna(), df["date_added_year"] - df["release_year"], np.nan)
    duration_num = pd.to_numeric(df["duration"].str.extract(r"(\d+)")[0], errors="coerce")
    df["duration_minutes"] = np.where(df["type"] == "Movie", duration_num, np.nan)
    df["season_count"] = np.where(df["type"] == "TV Show", duration_num, np.nan)
    return df


def build_summary(df: pd.DataFrame) -> dict[str, float | int | str]:
    summary = {
        "catalog_titles": int(len(df)),
        "movies": int((df["type"] == "Movie").sum()),
        "tv_shows": int((df["type"] == "TV Show").sum()),
        "countries_covered": int(df["primary_country"].nunique()),
        "genres_covered": int(df["primary_genre"].nunique()),
        "catalog_start_year": int(df["release_year"].min()),
        "catalog_end_year": int(df["release_year"].max()),
        "average_description_length": round(float(df["description_length"].mean()), 2),
        "median_content_age_at_add": round(float(df["content_age_at_add"].dropna().median()), 2),
    }
    return summary


def create_country_catalog(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("primary_country")
        .agg(
            Titles=("show_id", "count"),
            Movies=("type", lambda x: int((x == "Movie").sum())),
            TVShows=("type", lambda x: int((x == "TV Show").sum())),
        )
        .sort_values("Titles", ascending=False)
        .reset_index()
    )


def create_genre_catalog(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("primary_genre")
        .agg(
            Titles=("show_id", "count"),
            MeanDescriptionLength=("description_length", "mean"),
        )
        .sort_values("Titles", ascending=False)
        .reset_index()
    )


def bootstrap_ci(series: pd.Series, n_boot: int = 1000, seed: int = 42) -> dict[str, float]:
    arr = series.dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "mean": round(float(arr.mean()), 2),
        "ci_95_low": round(float(low), 2),
        "ci_95_high": round(float(high), 2),
    }


def run_statistical_tests(df: pd.DataFrame) -> dict:
    rating_type_table = pd.crosstab(df["type"], df["rating"])
    chi2, p_value, _, _ = stats.chi2_contingency(rating_type_table)

    movie_age = df.loc[df["type"] == "Movie", "content_age_at_add"].dropna()
    tv_age = df.loc[df["type"] == "TV Show", "content_age_at_add"].dropna()
    sample_size = min(len(movie_age), len(tv_age), 1500)
    movie_sample = movie_age.sample(sample_size, random_state=42)
    tv_sample = tv_age.sample(sample_size, random_state=42)
    mannwhitney = stats.mannwhitneyu(movie_sample, tv_sample, alternative="two-sided")

    return {
        "description_length_bootstrap": bootstrap_ci(df["description_length"]),
        "rating_vs_type_chi_square": {
            "test": "Chi-square",
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
        },
        "movie_vs_tv_content_age_test": {
            "test": "Mann-Whitney U",
            "sample_size_per_group": int(sample_size),
            "movie_median_years": round(float(movie_sample.median()), 2),
            "tv_median_years": round(float(tv_sample.median()), 2),
            "p_value": float(mannwhitney.pvalue),
        },
    }


def cluster_content(df: pd.DataFrame, n_clusters: int = 6) -> pd.DataFrame:
    text = (
        df["title"].fillna("")
        + " "
        + df["listed_in"].fillna("")
        + " "
        + df["description"].fillna("")
    )
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1200, min_df=5)
    matrix = vectorizer.fit_transform(text)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(matrix)

    terms = np.array(vectorizer.get_feature_names_out())
    cluster_rows = []
    for idx in range(n_clusters):
        center = model.cluster_centers_[idx]
        top_terms = ", ".join(terms[center.argsort()[-8:]][::-1])
        subset = df.loc[labels == idx]
        cluster_rows.append(
            {
                "cluster_id": idx,
                "titles": int(len(subset)),
                "dominant_type": subset["type"].mode().iloc[0],
                "top_terms": top_terms,
                "sample_titles": " | ".join(subset["title"].head(3).tolist()),
            }
        )
    cluster_df = pd.DataFrame(cluster_rows).sort_values("titles", ascending=False)
    df = df.copy()
    df["cluster_id"] = labels
    return df, cluster_df


def train_type_classifier(df: pd.DataFrame) -> dict:
    feature_df = df[["primary_country", "rating", "description_length", "cast_size", "release_year"]].copy()
    target = df["type"]
    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, target, test_size=0.2, random_state=42, stratify=target
    )

    numeric_features = ["description_length", "cast_size", "release_year"]
    categorical_features = ["primary_country", "rating"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=["Movie", "TV Show"])
    report = classification_report(y_test, predictions, output_dict=True)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "movie_precision": round(float(report["Movie"]["precision"]), 4),
        "movie_recall": round(float(report["Movie"]["recall"]), 4),
        "tv_precision": round(float(report["TV Show"]["precision"]), 4),
        "tv_recall": round(float(report["TV Show"]["recall"]), 4),
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def save_fig(name: str) -> None:
    plt.savefig(VISUALS_DIR / name, facecolor="white")
    plt.close()


def plot_catalog_timeline(df: pd.DataFrame) -> None:
    timeline = df.dropna(subset=["date_added_month"]).groupby(["date_added_month", "type"]).size().reset_index(name="titles")
    plt.figure(figsize=(13, 7))
    sns.lineplot(data=timeline, x="date_added_month", y="titles", hue="type", linewidth=3)
    plt.title("Catalog Additions Over Time")
    plt.xlabel("Month Added")
    plt.ylabel("Titles Added")
    plt.xticks(rotation=45)
    save_fig("catalog_additions_timeline.png")


def plot_top_countries(country_df: pd.DataFrame) -> None:
    top = country_df.head(12).sort_values("Titles")
    plt.figure(figsize=(13, 8))
    plt.barh(top["primary_country"], top["Titles"], color=sns.color_palette("viridis", len(top)))
    plt.title("Top Countries by Catalog Size")
    plt.xlabel("Titles")
    plt.ylabel("Country")
    save_fig("top_countries_catalog.png")


def plot_top_genres(genre_df: pd.DataFrame) -> None:
    top = genre_df.head(12).sort_values("Titles")
    plt.figure(figsize=(13, 8))
    plt.barh(top["primary_genre"], top["Titles"], color=sns.color_palette("rocket", len(top)))
    plt.title("Top Primary Genres in the Catalog")
    plt.xlabel("Titles")
    plt.ylabel("Genre")
    save_fig("top_genres_catalog.png")


def plot_rating_mix(df: pd.DataFrame) -> None:
    rating_mix = (
        df[df["rating"] != "Unknown"]
        .groupby(["rating", "type"])
        .size()
        .reset_index(name="titles")
    )
    top_ratings = rating_mix.groupby("rating")["titles"].sum().nlargest(10).index
    subset = rating_mix[rating_mix["rating"].isin(top_ratings)]
    pivot = subset.pivot(index="rating", columns="type", values="titles").fillna(0).sort_values("Movie")
    pivot.plot(kind="barh", stacked=True, figsize=(12, 7), color=["#4c78a8", "#f58518"])
    plt.title("Content Rating Mix by Title Type")
    plt.xlabel("Titles")
    plt.ylabel("Rating")
    save_fig("rating_mix.png")


def plot_content_age_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 7))
    filtered = df[df["content_age_at_add"].between(0, 30)]
    sns.boxplot(
        data=filtered,
        x="type",
        y="content_age_at_add",
        hue="type",
        palette={"Movie": "#4c78a8", "TV Show": "#f58518"},
        legend=False,
    )
    plt.title("Content Age at Platform Arrival")
    plt.xlabel("Type")
    plt.ylabel("Years Between Release and Platform Addition")
    save_fig("content_age_distribution.png")


def plot_cluster_sizes(cluster_df: pd.DataFrame) -> None:
    ordered = cluster_df.sort_values("titles")
    plt.figure(figsize=(12, 7))
    positions = np.arange(len(ordered))
    plt.barh(positions, ordered["titles"], color=sns.color_palette("mako", len(ordered)))
    plt.yticks(positions, ordered["cluster_id"].astype(str))
    plt.title("Description-Based Content Clusters")
    plt.xlabel("Titles")
    plt.ylabel("Cluster ID")
    save_fig("content_clusters.png")


def plot_confusion_matrix(metrics: dict) -> None:
    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Movie", "TV Show"], yticklabels=["Movie", "TV Show"])
    plt.title("Type Classifier Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_fig("type_classifier_confusion_matrix.png")


def plot_dashboard(summary: dict, country_df: pd.DataFrame, genre_df: pd.DataFrame, model_metrics: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Streaming Content Intelligence Dashboard", fontsize=22, fontweight="bold")

    type_counts = pd.Series({"Movies": summary["movies"], "TV Shows": summary["tv_shows"]})
    axes[0, 0].pie(type_counts.values, labels=type_counts.index, autopct="%1.0f%%", colors=["#4c78a8", "#f58518"], startangle=120)
    axes[0, 0].set_title("Catalog Composition")

    top_countries = country_df.head(5).sort_values("Titles")
    axes[0, 1].barh(top_countries["primary_country"], top_countries["Titles"], color=sns.color_palette("crest", len(top_countries)))
    axes[0, 1].set_title("Top Countries")

    top_genres = genre_df.head(5).sort_values("Titles")
    axes[1, 0].barh(top_genres["primary_genre"], top_genres["Titles"], color=sns.color_palette("flare", len(top_genres)))
    axes[1, 0].set_title("Top Genres")

    axes[1, 1].axis("off")
    lines = [
        f"Titles: {summary['catalog_titles']:,}",
        f"Countries: {summary['countries_covered']}",
        f"Genres: {summary['genres_covered']}",
        f"Median age at add: {summary['median_content_age_at_add']} years",
        f"Classifier accuracy: {model_metrics['accuracy']:.2%}",
        f"Avg description length: {summary['average_description_length']} words",
    ]
    axes[1, 1].text(0.0, 0.9, "Key Signals", fontsize=18, fontweight="bold")
    for idx, line in enumerate(lines):
        axes[1, 1].text(0.0, 0.75 - idx * 0.12, line, fontsize=13)

    plt.tight_layout()
    save_fig("executive_dashboard.png")


def write_outputs(summary: dict, tests: dict, model_metrics: dict, country_df: pd.DataFrame, genre_df: pd.DataFrame, cluster_df: pd.DataFrame) -> None:
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    TESTS_PATH.write_text(json.dumps(tests, indent=2))
    MODEL_PATH.write_text(json.dumps(model_metrics, indent=2))
    COUNTRY_PATH.write_text(country_df.to_csv(index=False))
    GENRE_PATH.write_text(genre_df.to_csv(index=False))
    CLUSTERS_PATH.write_text(cluster_df.to_csv(index=False))


def write_exec_summary(summary: dict, tests: dict, model_metrics: dict, country_df: pd.DataFrame, genre_df: pd.DataFrame, cluster_df: pd.DataFrame) -> None:
    top_country = country_df.iloc[0]["primary_country"]
    top_genre = genre_df.iloc[0]["primary_genre"]
    top_cluster = cluster_df.iloc[0]
    desc_ci = tests["description_length_bootstrap"]
    content = f"""# Executive Summary

## Overview

This project analyzes {summary['catalog_titles']:,} titles in a global streaming catalog spanning {summary['catalog_start_year']} to {summary['catalog_end_year']}.

## Business Highlights

- The catalog contains {summary['movies']:,} movies and {summary['tv_shows']:,} TV shows across {summary['countries_covered']} countries.
- The most represented country is {top_country}.
- The most represented primary genre is {top_genre}.
- Median content age at platform arrival is {summary['median_content_age_at_add']} years.

## Statistical Findings

- Mean description length is {desc_ci['mean']} words with a 95% bootstrap confidence interval of {desc_ci['ci_95_low']} to {desc_ci['ci_95_high']} words.
- Rating distribution is significantly associated with title type under a chi-square test with p-value {tests['rating_vs_type_chi_square']['p_value']:.6f}.
- Content age at addition differs significantly between movies and TV shows with p-value {tests['movie_vs_tv_content_age_test']['p_value']:.6f}.
- A lightweight title-type classifier reached {model_metrics['accuracy']:.2%} accuracy using country, rating, release year, cast size, and description length.

## Machine Learning Insight

The largest content cluster is cluster {int(top_cluster['cluster_id'])}, characterized by terms: {top_cluster['top_terms']}.
"""
    EXEC_SUMMARY_PATH.write_text(content)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    ensure_dirs()
    apply_theme()
    logger.info("Loading dataset from %s", args.data_path)
    df = load_and_clean(args.data_path)
    summary = build_summary(df)
    tests = run_statistical_tests(df)
    country_df = create_country_catalog(df)
    genre_df = create_genre_catalog(df)
    clustered_df, cluster_df = cluster_content(df)
    model_metrics = train_type_classifier(df)

    plot_catalog_timeline(df)
    plot_top_countries(country_df)
    plot_top_genres(genre_df)
    plot_rating_mix(df)
    plot_content_age_distribution(df)
    plot_cluster_sizes(cluster_df)
    plot_confusion_matrix(model_metrics)
    plot_dashboard(summary, country_df, genre_df, model_metrics)

    write_outputs(summary, tests, model_metrics, country_df, genre_df, cluster_df)
    write_exec_summary(summary, tests, model_metrics, country_df, genre_df, cluster_df)

    logger.info("Saved visuals to %s", VISUALS_DIR)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
