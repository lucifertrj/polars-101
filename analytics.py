from pathlib import Path
from typing import Optional

import polars as pl
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_CSV = DATA_DIR / "sales_data.csv"


@st.cache_data(show_spinner=False)
def load_df(csv_path: Optional[str] = None) -> pl.DataFrame:
    """Read the sales CSV into a Polars DataFrame (cached per path)."""
    path = Path(csv_path) if csv_path else DEFAULT_CSV
    df = pl.read_csv(path)
    # Parse the date column if present so time-series aggregations work.
    if "date" in df.columns and df.schema["date"] == pl.String:
        df = df.with_columns(pl.col("date").str.to_date(strict=False).alias("date"))
    return df


def kpis(df: pl.DataFrame) -> dict:
    """Headline numbers for the report's top row."""
    return {
        "total_revenue": float(df["total_revenue"].sum()),
        "total_orders": int(df.height),
        "units_sold": int(df["quantity"].sum()),
        "avg_order_value": float(df["total_revenue"].mean()),
        "unique_customers": int(df["customer_id"].n_unique()),
        "products": int(df["product"].n_unique()),
    }


def revenue_by(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Total revenue grouped by a categorical column, descending."""
    return (
        df.group_by(column)
        .agg(pl.col("total_revenue").sum().alias("revenue"))
        .sort("revenue", descending=True)
    )


def top_products(df: pl.DataFrame, n: int = 5) -> pl.DataFrame:
    """Top N products by revenue, with units sold alongside."""
    return (
        df.group_by("product")
        .agg(
            pl.col("total_revenue").sum().alias("revenue"),
            pl.col("quantity").sum().alias("units"),
        )
        .sort("revenue", descending=True)
        .head(n)
    )


def monthly_trend(df: pl.DataFrame) -> pl.DataFrame:
    """Revenue aggregated by month (requires a parsed `date` column)."""
    if "date" not in df.columns or df.schema["date"] != pl.Date:
        return pl.DataFrame({"month": [], "revenue": []})
    return (
        df.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month"))
        .group_by("month")
        .agg(pl.col("total_revenue").sum().alias("revenue"))
        .sort("month")
    )


def rep_leaderboard(df: pl.DataFrame) -> pl.DataFrame:
    """Sales rep performance: revenue and order count."""
    return (
        df.group_by("sales_rep")
        .agg(
            pl.col("total_revenue").sum().alias("revenue"),
            pl.len().alias("orders"),
        )
        .sort("revenue", descending=True)
    )
