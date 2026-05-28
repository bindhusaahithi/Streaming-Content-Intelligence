from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "online_retail.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"
VISUALS_DIR = ROOT / "visuals"
SUMMARY_PATH = PROCESSED_DIR / "summary_metrics.json"
RFM_PATH = PROCESSED_DIR / "customer_rfm_segments.csv"
MONTHLY_PATH = PROCESSED_DIR / "monthly_revenue.csv"


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    (ROOT / ".mplconfig").mkdir(exist_ok=True)


def load_and_clean_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["CustomerID"] = df["CustomerID"].astype("Int64")
    df["Description"] = df["Description"].fillna("Unknown")
    df = df.dropna(subset=["InvoiceDate", "CustomerID"])
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["Hour"] = df["InvoiceDate"].dt.hour
    return df


def build_summary(df: pd.DataFrame) -> dict[str, float | int | str]:
    invoice_totals = df.groupby("InvoiceNo")["Revenue"].sum()
    summary = {
        "records_after_cleaning": int(len(df)),
        "unique_customers": int(df["CustomerID"].nunique()),
        "countries_covered": int(df["Country"].nunique()),
        "study_period_start": df["InvoiceDate"].min().strftime("%Y-%m-%d"),
        "study_period_end": df["InvoiceDate"].max().strftime("%Y-%m-%d"),
        "total_revenue": round(float(df["Revenue"].sum()), 2),
        "average_order_value": round(float(invoice_totals.mean()), 2),
        "median_order_value": round(float(invoice_totals.median()), 2),
        "average_items_per_order": round(float(df.groupby("InvoiceNo")["Quantity"].sum().mean()), 2),
    }
    return summary


def create_monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby("InvoiceMonth")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique"),
            Customers=("CustomerID", "nunique"),
        )
        .reset_index()
    )
    monthly["AverageOrderValue"] = monthly["Revenue"] / monthly["Orders"]
    return monthly


def create_country_summary(df: pd.DataFrame) -> pd.DataFrame:
    country = (
        df.groupby("Country")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique"),
            Customers=("CustomerID", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
        .reset_index()
    )
    return country


def create_product_summary(df: pd.DataFrame) -> pd.DataFrame:
    product = (
        df.groupby("Description")
        .agg(
            Revenue=("Revenue", "sum"),
            Quantity=("Quantity", "sum"),
            Orders=("InvoiceNo", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
        .reset_index()
    )
    return product


def create_rfm_segments(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
        )
        .reset_index()
    )

    rfm["R_score"] = pd.qcut(rfm["Recency"].rank(method="first", ascending=False), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

    conditions = [
        rfm["RFM_score"] >= 10,
        rfm["RFM_score"].between(7, 9),
        rfm["RFM_score"].between(5, 6),
        rfm["RFM_score"] <= 4,
    ]
    choices = ["Champions", "Loyal", "Promising", "At Risk"]
    rfm["Segment"] = np.select(conditions, choices, default="At Risk")
    return rfm.sort_values(["Segment", "Monetary"], ascending=[True, False])


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


def save_fig(name: str) -> Path:
    path = VISUALS_DIR / name
    plt.savefig(path, facecolor="white")
    plt.close()
    return path


def plot_monthly_revenue(monthly: pd.DataFrame) -> None:
    plt.figure()
    plt.plot(monthly["InvoiceMonth"], monthly["Revenue"], color="#1f77b4", linewidth=3, marker="o")
    plt.fill_between(monthly["InvoiceMonth"], monthly["Revenue"], color="#a6d8ff", alpha=0.35)
    plt.title("Monthly Revenue Trend")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    save_fig("monthly_revenue_trend.png")


def plot_top_countries(country: pd.DataFrame) -> None:
    top = country.head(10).sort_values("Revenue")
    plt.figure()
    plt.barh(top["Country"], top["Revenue"], color=sns.color_palette("viridis", len(top)))
    plt.title("Top 10 Countries by Revenue")
    plt.xlabel("Revenue")
    plt.ylabel("Country")
    save_fig("top_countries_revenue.png")


def plot_top_products(product: pd.DataFrame) -> None:
    top = product.head(12).sort_values("Revenue")
    plt.figure(figsize=(14, 8))
    plt.barh(top["Description"], top["Revenue"], color=sns.color_palette("magma", len(top)))
    plt.title("Top 12 Products by Revenue")
    plt.xlabel("Revenue")
    plt.ylabel("Product")
    save_fig("top_products_revenue.png")


def plot_order_value_distribution(df: pd.DataFrame) -> None:
    invoice_totals = df.groupby("InvoiceNo")["Revenue"].sum()
    plt.figure()
    sns.histplot(invoice_totals, bins=50, color="#ff7f0e", kde=True)
    plt.title("Distribution of Order Values")
    plt.xlabel("Order Value")
    plt.ylabel("Number of Orders")
    plt.xlim(0, np.quantile(invoice_totals, 0.99))
    save_fig("order_value_distribution.png")


def plot_heatmap(df: pd.DataFrame) -> None:
    heatmap_data = (
        df.assign(Weekday=df["InvoiceDate"].dt.day_name())
        .pivot_table(index="Weekday", columns="Hour", values="Revenue", aggfunc="sum")
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"])
        .fillna(0)
    )
    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_data, cmap="YlGnBu")
    plt.title("Revenue Heatmap by Weekday and Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Weekday")
    save_fig("revenue_heatmap.png")


def plot_segment_counts(rfm: pd.DataFrame) -> None:
    segment_counts = rfm["Segment"].value_counts().reindex(["Champions", "Loyal", "Promising", "At Risk"]).fillna(0)
    plt.figure()
    plt.bar(segment_counts.index, segment_counts.values, color=["#00b894", "#0984e3", "#fdcb6e", "#d63031"])
    plt.title("Customer Segments from RFM Analysis")
    plt.xlabel("Segment")
    plt.ylabel("Customers")
    save_fig("rfm_segments.png")


def plot_segment_scatter(rfm: pd.DataFrame) -> None:
    segment_palette = {
        "Champions": "#00b894",
        "Loyal": "#0984e3",
        "Promising": "#fdcb6e",
        "At Risk": "#d63031",
    }
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=rfm,
        x="Frequency",
        y="Monetary",
        hue="Segment",
        palette=segment_palette,
        alpha=0.75,
        s=70,
    )
    plt.title("Customer Value Map: Frequency vs Monetary")
    plt.xlabel("Order Frequency")
    plt.ylabel("Monetary Value")
    plt.yscale("log")
    save_fig("customer_value_map.png")


def plot_executive_dashboard(summary: dict, monthly: pd.DataFrame, country: pd.DataFrame, rfm: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Online Retail Executive Dashboard", fontsize=22, fontweight="bold")

    axes[0, 0].plot(monthly["InvoiceMonth"], monthly["Revenue"], color="#6c5ce7", linewidth=3, marker="o")
    axes[0, 0].set_title("Monthly Revenue")
    axes[0, 0].tick_params(axis="x", rotation=45)

    top_country = country[country["Country"] != "United Kingdom"].head(5).sort_values("Revenue")
    axes[0, 1].barh(top_country["Country"], top_country["Revenue"], color=sns.color_palette("crest", len(top_country)))
    axes[0, 1].set_title("Top Countries Excluding UK")

    segment_counts = rfm["Segment"].value_counts().reindex(["Champions", "Loyal", "Promising", "At Risk"]).fillna(0)
    axes[1, 0].pie(
        segment_counts.values,
        labels=segment_counts.index,
        autopct="%1.0f%%",
        startangle=140,
        colors=["#00b894", "#0984e3", "#fdcb6e", "#d63031"],
    )
    axes[1, 0].set_title("Customer Segments")

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.0,
        0.9,
        "Key Metrics",
        fontsize=18,
        fontweight="bold",
        color="#2d3436",
    )
    lines = [
        f"Records after cleaning: {summary['records_after_cleaning']:,}",
        f"Unique customers: {summary['unique_customers']:,}",
        f"Countries covered: {summary['countries_covered']}",
        f"Total revenue: ${summary['total_revenue']:,.2f}",
        f"Average order value: ${summary['average_order_value']:,.2f}",
        f"Median order value: ${summary['median_order_value']:,.2f}",
    ]
    for idx, line in enumerate(lines):
        axes[1, 1].text(0.0, 0.75 - idx * 0.12, line, fontsize=13, color="#2d3436")

    plt.tight_layout()
    save_fig("executive_dashboard.png")


def write_outputs(summary: dict, monthly: pd.DataFrame, country: pd.DataFrame, product: pd.DataFrame, rfm: pd.DataFrame) -> None:
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    monthly.to_csv(MONTHLY_PATH, index=False)
    country.head(20).to_csv(PROCESSED_DIR / "top_countries.csv", index=False)
    product.head(20).to_csv(PROCESSED_DIR / "top_products.csv", index=False)
    rfm.to_csv(RFM_PATH, index=False)


def main() -> None:
    ensure_dirs()
    apply_theme()
    df = load_and_clean_data()
    summary = build_summary(df)
    monthly = create_monthly_revenue(df)
    country = create_country_summary(df)
    product = create_product_summary(df)
    rfm = create_rfm_segments(df)

    plot_monthly_revenue(monthly)
    plot_top_countries(country[country["Country"] != "United Kingdom"])
    plot_top_products(product)
    plot_order_value_distribution(df)
    plot_heatmap(df)
    plot_segment_counts(rfm)
    plot_segment_scatter(rfm)
    plot_executive_dashboard(summary, monthly, country, rfm)

    write_outputs(summary, monthly, country, product, rfm)
    print(json.dumps(summary, indent=2))
    print(f"Saved visuals to {VISUALS_DIR}")


if __name__ == "__main__":
    main()
