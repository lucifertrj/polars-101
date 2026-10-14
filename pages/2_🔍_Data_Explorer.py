"""
Data Explorer page — schema, interactive filtering/sorting, and a
lazy-vs-eager timing demo that makes the talk's Polars point concrete.
"""

import time

import polars as pl
import streamlit as st

from analytics import DEFAULT_CSV, load_df
from nav import render_nav

st.set_page_config(page_title="Data Explorer", page_icon="🔍", layout="wide")
render_nav()

st.title("🔍 Data Explorer")
st.caption("Inspect the dataset and feel the difference between eager and lazy Polars.")

df = load_df()

# ── Schema ──────────────────────────────────────────────────────────────────
st.subheader("Schema")
schema = pl.DataFrame(
    {
        "column": df.columns,
        "dtype": [str(d) for d in df.dtypes],
        "nulls": [df[c].null_count() for c in df.columns],
        "unique": [df[c].n_unique() for c in df.columns],
    }
)
st.dataframe(schema, use_container_width=True, hide_index=True)
st.caption(f"{df.height:,} rows × {df.width} columns")

st.divider()

# ── Interactive filter + sort ─────────────────────────────────────────────────
st.subheader("Filter & Sort")

cat_cols = [c for c, d in zip(df.columns, df.dtypes) if d == pl.String]
f1, f2, f3 = st.columns(3)

with f1:
    filter_col = st.selectbox("Filter column", ["(none)"] + cat_cols)
with f2:
    if filter_col != "(none)":
        values = sorted(df[filter_col].unique().to_list())
        picked = st.multiselect("Keep values", values, default=values)
    else:
        picked = None
with f3:
    sort_col = st.selectbox("Sort by", df.columns, index=df.columns.index("total_revenue") if "total_revenue" in df.columns else 0)
    descending = st.checkbox("Descending", value=True)

view = df
if filter_col != "(none)" and picked is not None:
    view = view.filter(pl.col(filter_col).is_in(picked))
view = view.sort(sort_col, descending=descending)

st.dataframe(view, use_container_width=True, hide_index=True)
st.caption(f"Showing {view.height:,} of {df.height:,} rows")

st.divider()

# ── Lazy vs eager timing demo ────────────────────────────────────────────────
st.subheader("⚡ Lazy vs Eager — same query, two engines")
st.markdown(
    "The same filter + group-by, run eagerly (`read_csv`, materialize everything) "
    "versus lazily (`scan_csv`, the optimizer pushes the filter into the scan and "
    "prunes unused columns). On this tiny CSV the gap is small — the point is the "
    "**query plan**, which you can inspect below."
)

repeats = st.slider("Repeat count (amplifies the timing)", 1, 2000, 500, step=100)

if st.button("Run benchmark"):
    path = str(DEFAULT_CSV)

    def eager():
        d = pl.read_csv(path)
        return (
            d.filter(pl.col("quantity") > 1)
            .group_by("category")
            .agg(pl.col("total_revenue").sum())
        )

    def lazy():
        return (
            pl.scan_csv(path)
            .filter(pl.col("quantity") > 1)
            .group_by("category")
            .agg(pl.col("total_revenue").sum())
            .collect()
        )

    t0 = time.perf_counter()
    for _ in range(repeats):
        eager()
    eager_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(repeats):
        lazy()
    lazy_ms = (time.perf_counter() - t0) * 1000

    c1, c2, c3 = st.columns(3)
    c1.metric("Eager total", f"{eager_ms:,.0f} ms")
    c2.metric("Lazy total", f"{lazy_ms:,.0f} ms")
    speedup = eager_ms / lazy_ms if lazy_ms else 1.0
    c3.metric("Lazy speedup", f"{speedup:.2f}×")

st.markdown("**Optimized query plan** (what the lazy engine actually runs):")
plan = (
    pl.scan_csv(str(DEFAULT_CSV))
    .filter(pl.col("quantity") > 1)
    .group_by("category")
    .agg(pl.col("total_revenue").sum())
    .explain()
)
st.code(plan, language="text")
st.caption(
    "Note the pushed-down predicate and projection in the CSV scan — the filter and "
    "column pruning happen while reading, not after."
)
