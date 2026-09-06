"""
Dashboard Sederhana - E-Commerce Public Dataset (Olist Brazil)
Menjalankan: streamlit run dashboard.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("main_data.csv")
    date_cols = ["order_purchase_timestamp", "order_delivered_customer_date",
                 "order_estimated_delivery_date"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    return df


main_df = load_data()
delivered_df = main_df[main_df["order_status"] == "delivered"].copy()

# ------------------------------------------------------------------
# Sidebar - filter
# ------------------------------------------------------------------
st.sidebar.title("E-Commerce Dashboard")
st.sidebar.markdown("Filter data untuk eksplorasi lebih lanjut.")

min_date = delivered_df["order_purchase_timestamp"].min().date()
max_date = delivered_df["order_purchase_timestamp"].max().date()

date_range = st.sidebar.date_input(
    "Rentang tanggal pembelian",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

state_options = sorted(delivered_df["customer_state"].dropna().unique())
selected_states = st.sidebar.multiselect(
    "Provinsi (state) pelanggan", options=state_options, default=state_options
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

mask = (
    (delivered_df["order_purchase_timestamp"].dt.date >= start_date)
    & (delivered_df["order_purchase_timestamp"].dt.date <= end_date)
    & (delivered_df["customer_state"].isin(selected_states))
)
filtered_df = delivered_df[mask]

# ------------------------------------------------------------------
# Header & KPI
# ------------------------------------------------------------------
st.title("Dashboard Analisis E-Commerce (Olist Brazil)")
st.markdown(
    "Dashboard ini merangkum hasil analisis data dari `notebook.ipynb`, "
    "mencakup performa kategori produk, waktu pengiriman antar provinsi, "
    "dan segmentasi pelanggan berbasis RFM."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah Order", f"{filtered_df['order_id'].nunique():,}")
col2.metric("Total Revenue (BRL)", f"{filtered_df['price'].sum():,.0f}")
col3.metric("Rata-rata Waktu Kirim (hari)", f"{filtered_df['delivery_time_days'].mean():.1f}")
col4.metric("Rata-rata Review Score", f"{filtered_df['review_score'].mean():.2f}")

st.divider()

# ------------------------------------------------------------------
# Pertanyaan 1: Kategori produk dengan revenue tertinggi & terendah
# ------------------------------------------------------------------
st.header("1. Kategori Produk: Revenue Tertinggi & Terendah")

category_revenue = (
    filtered_df.dropna(subset=["product_category_name_english"])
    .groupby("product_category_name_english")["price"]
    .sum()
    .sort_values(ascending=False)
)

n_top = st.slider("Jumlah kategori yang ditampilkan", 5, 15, 10)
top_cat = category_revenue.head(n_top)
bottom_cat = category_revenue.tail(n_top)

c1, c2 = st.columns(2)
with c1:
    st.subheader(f"{n_top} Kategori Revenue Tertinggi")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=top_cat.values, y=top_cat.index, color="#2a9d8f", ax=ax)
    ax.set_xlabel("Total Revenue (BRL)")
    ax.set_ylabel("")
    st.pyplot(fig)

with c2:
    st.subheader(f"{n_top} Kategori Revenue Terendah")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=bottom_cat.values, y=bottom_cat.index, color="#e76f51", ax=ax)
    ax.set_xlabel("Total Revenue (BRL)")
    ax.set_ylabel("")
    st.pyplot(fig)

st.divider()

# ------------------------------------------------------------------
# Pertanyaan 2: Waktu pengiriman per provinsi vs review score
# ------------------------------------------------------------------
st.header("2. Waktu Pengiriman per Provinsi vs Review Score")

state_summary = (
    filtered_df.groupby("customer_state")
    .agg(avg_delivery_days=("delivery_time_days", "mean"),
         avg_review_score=("review_score", "mean"),
         jumlah_order=("order_id", "nunique"))
    .sort_values("avg_delivery_days", ascending=False)
)

c3, c4 = st.columns(2)
with c3:
    st.subheader("10 Provinsi dengan Waktu Pengiriman Terlama")
    fig, ax = plt.subplots(figsize=(6, 5))
    top_state = state_summary.head(10)
    sns.barplot(x=top_state["avg_delivery_days"], y=top_state.index, color="#e9c46a", ax=ax)
    ax.set_xlabel("Rata-rata Waktu Pengiriman (hari)")
    ax.set_ylabel("Provinsi")
    st.pyplot(fig)

with c4:
    st.subheader("Korelasi Waktu Pengiriman vs Review Score")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=state_summary, x="avg_delivery_days", y="avg_review_score",
                     size="jumlah_order", sizes=(40, 400), legend=False, color="#264653", ax=ax)
    ax.set_xlabel("Rata-rata Waktu Pengiriman (hari)")
    ax.set_ylabel("Rata-rata Review Score")
    st.pyplot(fig)

corr = state_summary["avg_delivery_days"].corr(state_summary["avg_review_score"])
st.caption(f"Korelasi rata-rata waktu pengiriman vs rata-rata review score antar provinsi: **{corr:.3f}**")

st.divider()

# ------------------------------------------------------------------
# Analisis Lanjutan: RFM Segmentation
# ------------------------------------------------------------------
st.header("3. Analisis Lanjutan: Segmentasi Pelanggan (RFM)")
st.caption(
    "Segmentasi dilakukan dengan manual binning (kuartil) pada Recency, Frequency, "
    "dan Monetary, tanpa menggunakan algoritma machine learning."
)

snapshot_date = filtered_df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
rfm_df = filtered_df.groupby("customer_unique_id").agg(
    last_purchase=("order_purchase_timestamp", "max"),
    frequency=("order_id", "nunique"),
    monetary=("payment_value", "sum")
).reset_index()
rfm_df["recency"] = (snapshot_date - rfm_df["last_purchase"]).dt.days

if len(rfm_df) >= 4:
    rfm_df["R"] = pd.qcut(rfm_df["recency"], 4, labels=[4, 3, 2, 1], duplicates="drop").astype(int)
    rfm_df["F"] = pd.qcut(rfm_df["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm_df["M"] = pd.qcut(rfm_df["monetary"], 4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)

    def assign_segment(row):
        if row["R"] >= 3 and row["F"] >= 3 and row["M"] >= 3:
            return "Champions"
        elif row["R"] >= 3 and row["F"] < 3:
            return "Potential Loyalist"
        elif row["R"] < 3 and row["F"] >= 3:
            return "At Risk"
        elif row["R"] <= 2 and row["F"] <= 2 and row["M"] <= 2:
            return "Hibernating/Lost"
        else:
            return "Need Attention"

    rfm_df["segment"] = rfm_df.apply(assign_segment, axis=1)

    segment_summary = rfm_df.groupby("segment").agg(
        jumlah_pelanggan=("customer_unique_id", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
    ).sort_values("jumlah_pelanggan", ascending=False)

    c5, c6 = st.columns([1, 1])
    with c5:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.barplot(x=segment_summary["jumlah_pelanggan"], y=segment_summary.index,
                    color="#457b9d", ax=ax)
        ax.set_xlabel("Jumlah Pelanggan")
        ax.set_ylabel("Segmen")
        st.pyplot(fig)
    with c6:
        st.dataframe(segment_summary.round(2))
else:
    st.info("Data hasil filter terlalu sedikit untuk membentuk segmentasi RFM.")

st.divider()
st.caption("Dibuat oleh Fadli Nur Rofik — Submission Proyek Analisis Data Dicoding.")
