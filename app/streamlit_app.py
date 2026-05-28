from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
VISUALS_DIR = ROOT / "visuals"
REPORTS_DIR = ROOT / "reports"


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def add_global_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 20% 0%, rgba(83, 109, 254, 0.18), transparent 25%),
                radial-gradient(circle at 90% 10%, rgba(0, 191, 165, 0.14), transparent 20%),
                linear-gradient(180deg, #081120 0%, #0d1528 45%, #f4f7fb 45%, #f4f7fb 100%);
        }
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1240px;
        }
        h1, h2, h3 {
            letter-spacing: -0.02em;
        }
        .hero {
            padding: 2.2rem 2.4rem 2rem 2.4rem;
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(96, 165, 250, 0.16), rgba(16, 185, 129, 0.10)),
                rgba(7, 16, 33, 0.92);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 24px 80px rgba(7, 16, 33, 0.28);
            color: white;
            margin-bottom: 1.25rem;
        }
        .hero-kicker {
            color: #90caf9;
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.18em;
            font-weight: 700;
            margin-bottom: 0.85rem;
        }
        .hero-title {
            font-size: 3rem;
            line-height: 1.02;
            font-weight: 800;
            margin-bottom: 0.9rem;
        }
        .hero-text {
            font-size: 1rem;
            line-height: 1.7;
            color: rgba(255,255,255,0.78);
            max-width: 760px;
        }
        .metric-card {
            padding: 1.15rem 1.1rem 1rem 1.1rem;
            border-radius: 22px;
            background: white;
            border: 1px solid rgba(15, 23, 42, 0.06);
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
            min-height: 140px;
        }
        .metric-label {
            color: #64748b;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }
        .metric-value {
            font-size: 2rem;
            line-height: 1;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.55rem;
        }
        .metric-note {
            color: #475569;
            font-size: 0.92rem;
            line-height: 1.45;
        }
        .section-card {
            padding: 1.4rem 1.35rem 1.2rem 1.35rem;
            border-radius: 24px;
            background: white;
            border: 1px solid rgba(15, 23, 42, 0.06);
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
        }
        .section-heading {
            font-size: 1.35rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.35rem;
        }
        .section-copy {
            color: #475569;
            line-height: 1.7;
            font-size: 0.98rem;
        }
        .insight-banner {
            padding: 1rem 1.15rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(59,130,246,0.10), rgba(20,184,166,0.10));
            border: 1px solid rgba(59,130,246,0.12);
            color: #0f172a;
            margin-top: 0.7rem;
        }
        .insight-banner strong {
            font-size: 1.02rem;
        }
        .pill {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: #e0f2fe;
            color: #075985;
            font-size: 0.82rem;
            font-weight: 700;
            margin-right: 0.4rem;
            margin-bottom: 0.5rem;
        }
        div[data-testid="stImage"] img {
            border-radius: 18px;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #09111f 0%, #0d172a 100%);
        }
        [data-testid="stSidebar"] * {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-heading">{title}</div>
            <div class="section-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Streaming Content Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

add_global_styles()

summary = load_json(PROCESSED_DIR / "summary_metrics.json")
tests = load_json(PROCESSED_DIR / "statistical_tests.json")
model_metrics = load_json(PROCESSED_DIR / "model_metrics.json")
clusters = load_csv(PROCESSED_DIR / "content_clusters.csv")
countries = load_csv(PROCESSED_DIR / "country_catalog.csv")
genres = load_csv(PROCESSED_DIR / "genre_catalog.csv")
executive_summary = (REPORTS_DIR / "executive_summary.md").read_text()

st.sidebar.markdown("## Project Navigator")
selected_view = st.sidebar.radio(
    "Jump to section",
    ["Narrative Overview", "Statistical Lens", "Cluster Explorer", "Artifacts & Demo Notes"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Signals")
st.sidebar.markdown(f"- Model accuracy: **{model_metrics['accuracy']:.2%}**")
st.sidebar.markdown(f"- Countries covered: **{summary['countries_covered']}**")
st.sidebar.markdown(f"- Genres covered: **{summary['genres_covered']}**")
st.sidebar.markdown(f"- Median age at addition: **{summary['median_content_age_at_add']} years**")

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-kicker">Interactive Portfolio Demo</div>
        <div class="hero-title">Streaming Content Intelligence</div>
        <div class="hero-text">
            A cinematic analytics product built around the Netflix catalog dataset. The app blends
            statistical inference, unsupervised clustering, and lightweight predictive modeling into
            a recruiter-friendly story about how a global streaming catalog is structured, refreshed,
            and segmented.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
with metric_cols[0]:
    render_metric_card("Catalog Titles", f"{summary['catalog_titles']:,}", "Full global title inventory analyzed across the platform snapshot.")
with metric_cols[1]:
    render_metric_card("Geographic Reach", str(summary["countries_covered"]), "Countries represented in the catalog after primary-country parsing.")
with metric_cols[2]:
    render_metric_card("Genre Breadth", str(summary["genres_covered"]), "Distinct primary genres surfaced from the catalog taxonomy.")
with metric_cols[3]:
    render_metric_card("Classifier Accuracy", f"{model_metrics['accuracy']:.2%}", "Baseline interpretable model that predicts Movie vs TV Show.")

st.markdown("")

if selected_view == "Narrative Overview":
    render_section_intro(
        "Catalog Storyline",
        "This view is designed like a product review instead of a raw notebook dump. It highlights the scale of the catalog, the dominant content supply regions, the genre shape of the platform, and the timing patterns that suggest how fresh content is brought into the ecosystem.",
    )
    st.markdown(
        """
        <div class="insight-banner">
            <strong>What stands out:</strong> the catalog is heavily movie-skewed, spans 87 countries,
            and reaches the platform surprisingly quickly after release, with a median delay of just 1 year.
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.15, 0.85])
    with left:
        st.image(str(VISUALS_DIR / "executive_dashboard.png"), use_container_width=True)
    with right:
        st.markdown("### Content Mix")
        st.markdown(f'<span class="pill">Movies: {summary["movies"]:,}</span><span class="pill">TV Shows: {summary["tv_shows"]:,}</span><span class="pill">Release Range: {summary["catalog_start_year"]}–{summary["catalog_end_year"]}</span>', unsafe_allow_html=True)
        st.markdown("### Narrative Takeaways")
        st.write(
            {
                "Average description length": f"{summary['average_description_length']} words",
                "Median content age at addition": f"{summary['median_content_age_at_add']} years",
                "Top country by catalog size": countries.iloc[0]["primary_country"],
                "Top genre by volume": genres.iloc[0]["primary_genre"],
            }
        )
    gallery_top, gallery_bottom = st.columns(2)
    with gallery_top:
        st.image(str(VISUALS_DIR / "catalog_additions_timeline.png"), use_container_width=True)
        st.image(str(VISUALS_DIR / "top_countries_catalog.png"), use_container_width=True)
    with gallery_bottom:
        st.image(str(VISUALS_DIR / "top_genres_catalog.png"), use_container_width=True)
        st.image(str(VISUALS_DIR / "rating_mix.png"), use_container_width=True)
    st.markdown("### Country and Genre Tables")
    table_left, table_right = st.columns(2)
    with table_left:
        st.dataframe(countries.head(12), use_container_width=True, hide_index=True)
    with table_right:
        st.dataframe(genres.head(12), use_container_width=True, hide_index=True)

elif selected_view == "Statistical Lens":
    render_section_intro(
        "Inference and Modeling",
        "This section is aimed at technical reviewers. It surfaces inferential results, shows whether differences are likely to be real rather than visual noise, and demonstrates that the project goes beyond descriptive charts by adding a supervised baseline model.",
    )
    stat_cols = st.columns(3)
    with stat_cols[0]:
        render_metric_card(
            "Rating vs Type",
            f"{tests['rating_vs_type_chi_square']['p_value']:.2e}",
            "Chi-square p-value for whether content ratings are associated with title type.",
        )
    with stat_cols[1]:
        render_metric_card(
            "Movie vs TV Age Gap",
            f"{tests['movie_vs_tv_content_age_test']['p_value']:.2e}",
            "Mann-Whitney U p-value for differences in time-to-platform arrival.",
        )
    with stat_cols[2]:
        ci = tests["description_length_bootstrap"]
        render_metric_card(
            "Description Length CI",
            f"{ci['ci_95_low']}–{ci['ci_95_high']}",
            "95% bootstrap confidence interval for mean description length.",
        )
    left, right = st.columns([1, 1])
    with left:
        st.image(str(VISUALS_DIR / "content_age_distribution.png"), use_container_width=True)
        st.markdown("### Statistical Outputs")
        st.json(tests)
    with right:
        st.image(str(VISUALS_DIR / "type_classifier_confusion_matrix.png"), use_container_width=True)
        st.markdown("### Model Metrics")
        st.json(model_metrics)
        st.markdown(
            """
            <div class="insight-banner">
                <strong>Interpretation:</strong> even a simple metadata-based classifier can distinguish movies from
                TV shows with useful signal, which suggests there is structure in country, rating, release year,
                and content-description features that could support richer downstream models.
            </div>
            """,
            unsafe_allow_html=True,
        )

elif selected_view == "Cluster Explorer":
    render_section_intro(
        "Text-Driven Content Clusters",
        "Instead of treating the catalog only as metadata rows, this view uses title, genre, and description text to uncover thematic neighborhoods. It makes the project feel closer to real content discovery, recommendation, or catalog intelligence work.",
    )
    st.image(str(VISUALS_DIR / "content_clusters.png"), use_container_width=True)
    chooser_col, detail_col = st.columns([0.42, 0.58])
    with chooser_col:
        selected_cluster = st.selectbox(
            "Choose a cluster to inspect",
            clusters["cluster_id"].tolist(),
            format_func=lambda x: f"Cluster {x}",
        )
        st.dataframe(clusters[["cluster_id", "titles", "dominant_type"]], use_container_width=True, hide_index=True)
    cluster_row = clusters.loc[clusters["cluster_id"] == selected_cluster].iloc[0]
    with detail_col:
        st.markdown("### Cluster Profile")
        st.markdown(f'<span class="pill">Cluster {int(cluster_row["cluster_id"])}</span><span class="pill">{cluster_row["dominant_type"]}</span><span class="pill">{int(cluster_row["titles"])} titles</span>', unsafe_allow_html=True)
        st.markdown("#### Top semantic terms")
        st.write(cluster_row["top_terms"])
        st.markdown("#### Example titles")
        samples = [title.strip() for title in str(cluster_row["sample_titles"]).split("|") if title.strip()]
        for sample in samples:
            st.markdown(f"- {sample}")
        st.markdown(
            """
            <div class="insight-banner">
                <strong>Why this matters:</strong> these clusters can support recommendation systems, editorial curation,
                search taxonomy refinement, or content-acquisition strategy by revealing hidden structure in catalog text.
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    render_section_intro(
        "Artifacts, Delivery, and Reviewability",
        "This final view helps a reviewer understand that the project is not just a screenshot gallery. It includes reproducible code, processed outputs, a notebook, and an executive summary packaged in a way that is easy to inspect on GitHub.",
    )
    art_left, art_right = st.columns([0.55, 0.45])
    with art_left:
        st.markdown("### Executive Summary")
        st.markdown(executive_summary)
        st.markdown("### Included Deliverables")
        st.code(
            "\n".join(
                [
                    "scripts/streaming_content_intelligence.py",
                    "notebooks/streaming_content_intelligence.ipynb",
                    "app/streamlit_app.py",
                    "reports/executive_summary.md",
                    "data/processed/summary_metrics.json",
                    "data/processed/statistical_tests.json",
                    "data/processed/model_metrics.json",
                    "data/processed/content_clusters.csv",
                ]
            )
        )
    with art_right:
        st.markdown("### Processed Outputs")
        st.dataframe(
            pd.DataFrame(
                {
                    "artifact": [
                        "summary_metrics.json",
                        "statistical_tests.json",
                        "model_metrics.json",
                        "country_catalog.csv",
                        "genre_catalog.csv",
                        "content_clusters.csv",
                    ]
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            """
            <div class="insight-banner">
                <strong>Demo note:</strong> this app is intentionally styled as a lightweight analytics product, so a recruiter
                can scan the project in a few minutes without opening every script first.
            </div>
            """,
            unsafe_allow_html=True,
        )
