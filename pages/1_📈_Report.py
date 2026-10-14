"""
Report page — a static, Polars-computed summary of the sales data.

No LLM calls: every number and chart here comes straight from Polars
aggregations in ``analytics.py``, so it always renders instantly during a demo.
"""

import polars as pl
import streamlit as st

from analytics import (
    kpis,
    load_df,
    monthly_trend,
    rep_leaderboard,
    revenue_by,
    top_products,
)
from nav import render_nav

st.set_page_config(page_title="Sales Report", page_icon="📈", layout="wide")
render_nav()

st.title("📈 Sales Report")
st.caption("Static summary computed with Polars — no LLM, always on.")

df = load_df()


def _money(x: float) -> str:
    return f"${x:,.0f}"


# ── KPI row ───────────────────────────────────────────────────────────────
k = kpis(df)
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Revenue", _money(k["total_revenue"]))
c2.metric("Orders", f"{k['total_orders']:,}")
c3.metric("Units Sold", f"{k['units_sold']:,}")
c4.metric("Avg Order Value", _money(k["avg_order_value"]))
c5.metric("Customers", f"{k['unique_customers']:,}")
c6.metric("Products", f"{k['products']:,}")

st.divider()

# ── Revenue by category / region ────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Revenue by Category")
    cat = revenue_by(df, "category")
    st.bar_chart(cat, x="category", y="revenue", color="#4C78A8")

with right:
    st.subheader("Revenue by Region")
    reg = revenue_by(df, "region")
    st.bar_chart(reg, x="region", y="revenue", color="#F58518")

st.divider()

# ── Monthly trend ─────────────────────────────────────────────────────────
st.subheader("Monthly Revenue Trend")
trend = monthly_trend(df)
if trend.height:
    st.line_chart(trend, x="month", y="revenue", color="#54A24B")
else:
    st.info("No parseable `date` column found — skipping the trend chart.")

st.divider()

# ── Top products & rep leaderboard ──────────────────────────────────────────
left2, right2 = st.columns(2)

with left2:
    st.subheader("Top 5 Products by Revenue")
    tp = top_products(df, 5)
    st.dataframe(
        tp.with_columns(pl.col("revenue").round(2)),
        use_container_width=True,
        hide_index=True,
    )

with right2:
    st.subheader("Sales Rep Leaderboard")
    reps = rep_leaderboard(df)
    st.dataframe(
        reps.with_columns(pl.col("revenue").round(2)),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.caption("All figures computed in-process with Polars · refresh re-reads the CSV")
