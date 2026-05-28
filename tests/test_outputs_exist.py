from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def test_core_outputs_exist() -> None:
    required = [
        ROOT / "README.md",
        ROOT / "pyproject.toml",
        ROOT / "notebooks" / "applied_science_retail_analytics.ipynb",
        ROOT / "scripts" / "applied_science_retail_analysis.py",
        ROOT / "data" / "processed" / "summary_metrics.json",
        ROOT / "data" / "processed" / "statistical_tests.json",
        ROOT / "reports" / "executive_summary.md",
        ROOT / "visuals" / "executive_dashboard.png",
    ]
    for path in required:
        assert path.exists(), f"Missing required artifact: {path}"


def test_summary_metrics_shape() -> None:
    data = json.loads((ROOT / "data" / "processed" / "summary_metrics.json").read_text())
    assert data["records_after_cleaning"] > 100000
    assert data["unique_customers"] > 1000
