"""
Prototipe Deployment (Tahap 6 CRISP-DM, BAB III Subbab 3.2.1) -- Aplikasi
Streamlit yang mendemonstrasikan skor risiko penurunan kemandirian digital
pelanggan menggunakan model Jalur I (XGBoost pada fitur RFMC), dilengkapi
penjelasan SHAP lokal per pelanggan.

CATATAN PENTING (jujur soal cakupan prototipe ini): BAB III Subbab 3.2.1
memvisikan integrasi dgn Customer Data Platform (CDP) perusahaan untuk
mendukung notifikasi personalisasi & penjadwalan kunjungan salesman. Karena
data transaksi riil Wings Store (B2B) dan akses ke CDP asli belum tersedia
pada tahap penelitian ini (penelitian memakai dataset publik REES46),
aplikasi ini adalah PROTOTIPE/DEMONSTRATOR -- bukan integrasi produksi
sungguhan. Logika scoring intinya (scoring_core.py) SENGAJA dipisah dari
antarmuka ini supaya bisa dipakai ulang oleh REST API atau batch job kalau
nanti benar-benar diintegrasikan ke CDP sesungguhnya.

Cara jalan: taruh file ini & scoring_core.py di folder yang sama, lalu:
    pip install -r requirements.txt
    streamlit run app.py
"""
import io

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from scoring_core import FITUR, THRESHOLD_RISIKO, muat_model

st.set_page_config(
    page_title="Prototipe Deteksi Risiko Penurunan Kemandirian Digital",
    page_icon="📉",
    layout="wide",
)


@st.cache_resource(show_spinner="Memuat model Jalur I (XGBoost)...")
def _muat_model_cached():
    return muat_model()


def _render_shap_waterfall(explainer_model, df_satu_baris: pd.DataFrame):
    """Grafik waterfall SHAP utk 1 pelanggan -- menunjukkan fitur mana
    (R/F/M/C) yang mendorong skor pelanggan ini naik/turun dari baseline."""
    shap_values = explainer_model.shap_lokal(df_satu_baris)
    fig = plt.figure(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig, clear_figure=True)


def _render_shap_beeswarm(explainer_model, df_banyak_baris: pd.DataFrame):
    """Grafik beeswarm SHAP agregat utk banyak pelanggan sekaligus (mode CSV)."""
    shap_values = explainer_model.shap_lokal(df_banyak_baris)
    fig = plt.figure(figsize=(8, 4))
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(fig, clear_figure=True)


st.title("📉 Prototipe Deteksi Dini Risiko Penurunan Kemandirian Digital")
st.caption(
    "Model: Jalur I (XGBoost pada fitur RFMC) -- model dengan performa terbaik "
    "pada perbandingan Tahap 5 (menang di 5 dari 5 metrik evaluasi). "
    f"Ambang batas klasifikasi (threshold): {THRESHOLD_RISIKO}."
)

try:
    model_wrapper = _muat_model_cached()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

tab_manual, tab_batch = st.tabs(["🧍 Input Manual (1 Pelanggan)", "📂 Upload CSV (Banyak Pelanggan)"])

# =============================================================================
with tab_manual:
    st.subheader("Input Profil RFMC Satu Pelanggan")
    st.write(
        "Masukkan nilai Recency (R), Frequency (F), Monetary (M), dan "
        "Clumpiness (C) pelanggan -- skala nilai harus konsisten dengan "
        "proses feature engineering RFMC pada BAB III (biasanya hasil "
        "normalisasi/standardisasi, BUKAN nilai mentah tanggal/rupiah)."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        r_input = st.number_input("Recency (R)", value=0.0, step=0.01, format="%.4f")
    with col2:
        f_input = st.number_input("Frequency (F)", value=0.0, step=0.01, format="%.4f")
    with col3:
        m_input = st.number_input("Monetary (M)", value=0.0, step=0.01, format="%.4f")
    with col4:
        c_input = st.number_input("Clumpiness (C)", value=0.0, step=0.01, format="%.4f")

    if st.button("🔍 Hitung Skor Risiko", type="primary"):
        hasil = model_wrapper.skor_satu_pelanggan(r_input, f_input, m_input, c_input)

        col_skor, col_label = st.columns(2)
        with col_skor:
            st.metric("Skor Risiko (Probabilitas)", f"{hasil.skor_risiko:.4f}")
        with col_label:
            if hasil.berisiko:
                st.error(f"Status: {hasil.label}")
            else:
                st.success(f"Status: {hasil.label}")

        st.write("**Penjelasan SHAP (kontribusi tiap fitur terhadap skor ini):**")
        df_satu = pd.DataFrame([{"R": r_input, "F": f_input, "M": m_input, "C": c_input}])
        _render_shap_waterfall(model_wrapper, df_satu)
        st.caption(
            "Grafik di atas menunjukkan fitur mana (R/F/M/C) yang mendorong skor "
            "pelanggan ini naik (merah) atau turun (biru) dari nilai dasar "
            "(base value) model -- gunakan untuk memahami ALASAN di balik skor, "
            "bukan hanya skornya saja, sesuai tujuan interpretasi SHAP pada BAB "
            "III Subbab 3.2.1 dan BAB II Subbab 2.5.5."
        )

# =============================================================================
with tab_batch:
    st.subheader("Scoring Banyak Pelanggan Sekaligus (CSV)")
    st.write(
        f"Upload file CSV dengan kolom wajib: `{', '.join(FITUR)}` "
        "(boleh ada kolom lain seperti `user_id`, akan dipertahankan di hasil)."
    )

    file_csv = st.file_uploader("Pilih file CSV", type=["csv"])

    if file_csv is not None:
        try:
            df_upload = pd.read_csv(file_csv)
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")
            st.stop()

        kolom_hilang = [k for k in FITUR if k not in df_upload.columns]
        if kolom_hilang:
            st.error(
                f"Kolom wajib tidak ditemukan di CSV: {kolom_hilang}. "
                f"Kolom yang ada di file Anda: {list(df_upload.columns)}"
            )
            st.stop()

        st.write(f"Jumlah baris terbaca: **{len(df_upload):,}**")

        if st.button("🔍 Jalankan Scoring Batch", type="primary"):
            with st.spinner("Menghitung skor risiko untuk seluruh baris..."):
                df_hasil = model_wrapper.skor_batch(df_upload)

            n_berisiko = int(df_hasil["berisiko"].sum())
            st.metric(
                "Jumlah pelanggan berisiko",
                f"{n_berisiko:,} dari {len(df_hasil):,}",
                f"{n_berisiko / len(df_hasil) * 100:.1f}%",
            )

            st.dataframe(df_hasil, use_container_width=True)

            buffer_csv = io.StringIO()
            df_hasil.to_csv(buffer_csv, index=False)
            st.download_button(
                "⬇️ Download Hasil Scoring (CSV)",
                data=buffer_csv.getvalue(),
                file_name="hasil_scoring_risiko_pelanggan.csv",
                mime="text/csv",
            )

            st.write("**Ringkasan SHAP Agregat (seluruh pelanggan di file ini):**")
            n_sample_shap = min(len(df_upload), 2000)
            if n_sample_shap < len(df_upload):
                st.caption(
                    f"(Ditampilkan dari {n_sample_shap:,} baris pertama saja demi "
                    f"kecepatan render, dari total {len(df_upload):,} baris.)"
                )
            _render_shap_beeswarm(model_wrapper, df_upload.head(n_sample_shap))
