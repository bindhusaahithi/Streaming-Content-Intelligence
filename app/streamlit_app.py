from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
VISUALS_DIR = ROOT / "visuals"
REPORTS_DIR = ROOT / "reports"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


st.set_page_config(
    page_title="Streaming Content Intelligence",
    page_icon="🎬",
    layout="wide",
)

st.title("Streaming Content Intelligence")
st.caption("Interactive portfolio app for catalog analytics, clustering, and predictive modeling.")

summary = load_json(PROCESSED_DIR / "summary_metrics.json")
tests = load_json(PROCESSED_DIR / "statistical_tests.json")
model_metrics = load_json(PROCESSED_DIR / "model_metrics.json")
clusters = pd.read_csv(PROCESSED_DIR / "content_clusters.csv")
countries = pd.read_csv(PROCESSED_DIR / "country_catalog.csv")
genres = pd.read_csv(PROCESSED_DIR / "genre_catalog.csv")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Titles", f"{summary['catalog_titles']:,}")
col2.metric("Countries", summary["countries_covered"])
col3.metric("Genres", summary["genres_covered"])
col4.metric("Model Accuracy", f"{model_metrics['accuracy']:.2%}")

st.image(str(VISUALS_DIR / "executive_dashboard.png"), use_container_width=True)

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Statistics", "Clusters", "Artifacts"])

with tab1:
    left, right = st.columns(2)
    with left:
        st.subheader("Catalog Composition")
        st.write(
            {
                "Movies": summary["movies"],
                "TV Shows": summary["tv_shows"],
                "Catalog Range": f"{summary['catalog_start_year']}–{summary['catalog_end_year']}",
                "Median Age at Addition": summary["median_content_age_at_add"],
            }
        )
        st.image(str(VISUALS_DIR / "catalog_additions_timeline.png"), use_container_width=True)
        st.image(str(VISUALS_DIR / "rating_mix.png"), use_container_width=True)
    with right:
        st.subheader("Top Countries")
        st.dataframe(countries.head(15), use_container_width=True)
        st.image(str(VISUALS_DIR / "top_countries_catalog.png"), use_container_width=True)
        st.subheader("Top Genres")
        st.dataframe(genres.head(15), use_container_width=True)
        st.image(str(VISUALS_DIR / "top_genres_catalog.png"), use_container_width=True)

with tab2:
    st.subheader("Statistical Findings")
    st.json(tests)
    st.image(str(VISUALS_DIR / "content_age_distribution.png"), use_container_width=True)
    st.image(str(VISUALS_DIR / "type_classifier_confusion_matrix.png"), use_container_width=True)
    st.write("Model metrics")
    st.json(model_metrics)

with tab3:
    st.subheader("Description-Based Content Clusters")
    st.dataframe(clusters, use_container_width=True)
    st.image(str(VISUALS_DIR / "content_clusters.png"), use_container_width=True)
    chosen_cluster = st.selectbox("Explore cluster", clusters["cluster_id"].tolist())
    cluster_row = clusters.loc[clusters["cluster_id"] == chosen_cluster].iloc[0]
    st.write(
        {
            "Titles": int(cluster_row["titles"]),
            "Dominant Type": cluster_row["dominant_type"],
            "Top Terms": cluster_row["top_terms"],
            "Sample Titles": cluster_row["sample_titles"],
        }
    )

with tab4:
    st.subheader("Executive Summary")
    st.markdown((REPORTS_DIR / "executive_summary.md").read_text())
    st.subheader("Project Files")
    st.code(
        "\n".join(
            [
                "scripts/streaming_content_intelligence.py",
                "notebooks/streaming_content_intelligence.ipynb",
                "reports/executive_summary.md",
                "data/processed/summary_metrics.json",
                "data/processed/statistical_tests.json",
                "data/processed/model_metrics.json",
            ]
        )
    )
