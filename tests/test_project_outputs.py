from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def test_core_artifacts_exist() -> None:
    required = [
        ROOT / "README.md",
        ROOT / "pyproject.toml",
        ROOT / "app" / "streamlit_app.py",
        ROOT / "notebooks" / "streaming_content_intelligence.ipynb",
        ROOT / "scripts" / "streaming_content_intelligence.py",
        ROOT / "reports" / "executive_summary.md",
        ROOT / "data" / "processed" / "summary_metrics.json",
        ROOT / "data" / "processed" / "statistical_tests.json",
        ROOT / "data" / "processed" / "model_metrics.json",
        ROOT / "visuals" / "executive_dashboard.png",
    ]
    for path in required:
        assert path.exists(), f"Missing required artifact: {path}"


def test_summary_shape() -> None:
    data = json.loads((ROOT / "data" / "processed" / "summary_metrics.json").read_text())
    assert data["catalog_titles"] > 5000
    assert data["countries_covered"] > 20
