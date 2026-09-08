"""
Dashboard Sederhana - E-Commerce Public Dataset (Olist Brazil)
Menjalankan: streamlit run dashboard.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set_theme(style="whitegrid")

# Palet warna konsisten: abu-abu netral untuk nilai biasa,
# warna aksen dipakai HANYA untuk menyorot nilai tertinggi/terendah/penting
# (warna sebagai penunjuk informasi, bukan sekadar dekorasi).
COLOR_NEUTRAL = "#b0b0b0"
COLOR_HIGH = "#2a9d8f"
COLOR_LOW = "#e76f51"
COLOR_HIGHLIGHT = "#264653"

# Path data selalu mengikuti lokasi file dashboard.py ini,
# supaya tetap ketemu file-nya baik dijalankan dari folder dashboard/
# maupun dari root repository (seperti di Streamlit Cloud).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "main_data.csv")


# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    return df


main_df = load_data()
delivered_items = main_df[main_df["order_status"] == "delivered"].copy()
# level order (1 baris = 1 order) -> dipakai untuk waktu pengiriman, review score, dan RFM
# agar order dengan banyak item tidak dihitung berulang kali.
delivered_orders = delivered_items.drop_duplicates(subset="order_id").copy()

# ------------------------------------------------------------------
# Sidebar - filter
# ------------------------------------------------------------------
st.sidebar.title("E-Commerce Dashboard")
st.sidebar.markdown("Filter data untuk eksplorasi lebih lanjut.")

min_date = delivered_orders["order_purchase_timestamp"].min().date()
max_date = delivered_orders["order_purchase_timestamp"].max().date()

date_range = st.sidebar.date_input(
    "Rentang tanggal pembelian",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

state_options = sorted(delivered_orders["customer_state"].dropna().unique())
selected_states = st.sidebar.multiselect(
    "Provinsi (state) pelanggan", options=state_options, default=state_options
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

order_mask = (
    (delivered_orders["order_purchase_timestamp"].dt.date >= start_date)
    & (delivered_orders["order_purchase_timestamp"].dt.date <= end_date)
    & (delivered_orders["customer_state"].isin(selected_states))
)
filtered_orders = delivered_orders[order_mask]
filtered_order_ids = set(filtered_orders["order_id"])
filtered_items = delivered_items[delivered_items["order_id"].isin(filtered_order_ids)]

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
col1.metric("Jumlah Order", f"{filtered_orders['order_id'].nunique():,}")
col2.metric("Total Revenue (BRL)", f"{filtered_items['price'].sum():,.0f}")
col3.metric("Rata-rata Waktu Kirim (hari)", f"{filtered_orders['delivery_time_days'].mean():.1f}")
col4.metric("Rata-rata Review Score", f"{filtered_orders['review_score'].mean():.2f}")

st.divider()

# ------------------------------------------------------------------
# Pertanyaan 1: Kategori produk dengan revenue tertinggi & terendah
# ------------------------------------------------------------------
st.header("1. Kategori Produk: Revenue Tertinggi & Terendah")
st.caption("Dihitung pada level item pesanan, karena setiap baris item merepresentasikan satu transaksi penjualan produk.")

category_revenue = (
    filtered_items.dropna(subset=["product_category_name_english"])
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
    colors = [COLOR_HIGH if v == top_cat.max() else COLOR_NEUTRAL for v in top_cat.values]
    ax.barh(top_cat.index[::-1], top_cat.values[::-1], color=colors[::-1])
    ax.set_xlabel("Total Revenue (BRL)")
    st.pyplot(fig)

with c2:
    st.subheader(f"{n_top} Kategori Revenue Terendah")
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = [COLOR_LOW if v == bottom_cat.min() else COLOR_NEUTRAL for v in bottom_cat.values]
    ax.barh(bottom_cat.index[::-1], bottom_cat.values[::-1], color=colors[::-1])
    ax.set_xlabel("Total Revenue (BRL)")
    st.pyplot(fig)

st.divider()

# ------------------------------------------------------------------
# Pertanyaan 2: Waktu pengiriman per provinsi vs review score
# ------------------------------------------------------------------
st.header("2. Waktu Pengiriman per Provinsi vs Review Score")
st.caption("Dihitung pada level order (1 baris = 1 order) agar order multi-item tidak terhitung berulang.")

state_summary = (
    filtered_orders.groupby("customer_state")
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
    colors = [COLOR_LOW if v == top_state["avg_delivery_days"].max() else COLOR_NEUTRAL
              for v in top_state["avg_delivery_days"].values]
    ax.barh(top_state.index[::-1], top_state["avg_delivery_days"].values[::-1], color=colors[::-1])
    ax.set_xlabel("Rata-rata Waktu Pengiriman (hari)")
    ax.set_ylabel("Provinsi")
    st.pyplot(fig)

with c4:
    st.subheader("Korelasi Waktu Pengiriman vs Review Score")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(state_summary["avg_delivery_days"], state_summary["avg_review_score"],
               s=state_summary["jumlah_order"] / max(state_summary["jumlah_order"].max(), 1) * 400 + 30,
               color=COLOR_HIGHLIGHT, alpha=0.7)
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
    "dan Monetary di level order (bukan level item), tanpa menggunakan algoritma machine learning."
)

snapshot_date = filtered_orders["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
rfm_df = filtered_orders.groupby("customer_unique_id").agg(
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
        colors = [COLOR_HIGH if s == "Champions" else COLOR_LOW if s == "Hibernating/Lost" else COLOR_NEUTRAL
                  for s in segment_summary.index]
        ax.barh(segment_summary.index[::-1], segment_summary["jumlah_pelanggan"].values[::-1], color=colors[::-1])
        ax.set_xlabel("Jumlah Pelanggan")
        ax.set_ylabel("Segmen")
        st.pyplot(fig)
    with c6:
        st.dataframe(segment_summary.round(2))
else:
    st.info("Data hasil filter terlalu sedikit untuk membentuk segmentasi RFM.")

st.divider()
st.caption("Dibuat oleh Fadli Nur Rofik — Submission Proyek Analisis Data Dicoding.")
