"""
Prototipe prediksi risiko pelanggan toko kosmetik daring.

Aplikasi ini menerima berkas CSV berisi log event sesuai struktur dataset
"eCommerce Events History in Cosmetics Shop", menghitung fitur RFMC dari log
tersebut dengan prosedur yang identik dengan tahap rekayasa fitur penelitian,
lalu memprediksi pelanggan mana yang berisiko tidak kembali aktif berbelanja
pada periode berikutnya.

Cara menjalankan:
    pip install -r requirements.txt
    streamlit run app.py
"""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from rfmc_builder import (
    KOLOM_WAJIB, baca_events_csv, hitung_rfmc_dari_events, periode_default, ringkas_events,
)
from scoring_core import (
    FITUR, LABEL_BERISIKO, LABEL_PASIF, LABEL_TIDAK_BERISIKO, THRESHOLD_RISIKO, muat_model,
)

st.set_page_config(
    page_title="Prediksi Risiko Pelanggan Kosmetik",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Gaya tampilan
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  :root {
    --bg:#0f1116; --card:#171a21; --border:#262b36; --accent:#e0518a;
    --ok:#3ecf8e; --warn:#f0a830; --muted:#9aa0ac;
  }
  #MainMenu, footer {visibility:hidden;}
  .block-container {padding-top:2.2rem; padding-bottom:3rem; max-width:1320px;}

  .hero {
    background:linear-gradient(135deg, rgba(224,81,138,.14), rgba(224,81,138,.02));
    border:1px solid var(--border); border-radius:16px;
    padding:1.6rem 1.8rem; margin-bottom:1.6rem;
  }
  .hero h1 {margin:0; font-size:1.75rem; font-weight:700; letter-spacing:-.02em;}
  .hero p {margin:.45rem 0 0; color:var(--muted); font-size:.95rem; line-height:1.55;}
  .pill {
    display:inline-block; padding:.2rem .7rem; border-radius:999px;
    background:rgba(224,81,138,.16); color:var(--accent);
    font-size:.72rem; font-weight:600; letter-spacing:.04em; text-transform:uppercase;
  }

  .kpi {
    background:var(--card); border:1px solid var(--border); border-radius:14px;
    padding:1.05rem 1.2rem; height:100%;
  }
  .kpi .lbl {color:var(--muted); font-size:.74rem; font-weight:600;
             letter-spacing:.06em; text-transform:uppercase;}
  .kpi .val {font-size:1.75rem; font-weight:700; margin-top:.3rem; line-height:1.15;}
  .kpi .sub {color:var(--muted); font-size:.78rem; margin-top:.2rem;}
  .kpi.accent .val {color:var(--accent);}
  .kpi.ok .val {color:var(--ok);}
  .kpi.warn .val {color:var(--warn);}

  .panel {
  background:var(--card);
  border:1px solid var(--border);
  border-radius:14px;
  padding:1.3rem 1.4rem;
  margin-bottom:1.1rem;
    }
    .panel h3 {margin:0 0 .35rem; font-size:1.02rem; font-weight:650;}

    /* Modifikasi latar belakang & teks khusus untuk panel petunjuk unggah berkas */
    .panel-putih {
    background: #ffffff !important;
    color: #111111 !important;
    border: 1px solid #dcdcdc !important;
    border-radius:14px;
    padding:1.3rem 1.4rem;
    margin-bottom:1.1rem;
    }
    .panel-putih h3 {
    margin:0 0 .35rem; 
    font-size:1.02rem; 
    font-weight:650;
    color: #111111 !important;
    }
    .panel-putih .desc {
    color: #333333 !important;
    font-size:.95rem;
    }
  .stButton>button {
    background:var(--accent); color:#fff; border:none; border-radius:10px;
    padding:.6rem 1.3rem; font-weight:600; width:100%;
  }
  .stButton>button:hover {background:#c7447a; color:#fff;}
  .stDownloadButton>button {
    background:transparent; color:var(--accent);
    border:1px solid var(--accent); border-radius:10px; font-weight:600;
  }
  code {color:var(--accent);}
</style>
""", unsafe_allow_html=True)


def kpi(kolom, label, nilai, sub="", gaya=""):
    kolom.markdown(
        f'<div class="kpi {gaya}"><div class="lbl">{label}</div>'
        f'<div class="val">{nilai}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def panel_buka(judul, deskripsi=""):
    st.markdown(
        f"""<div class="panel-putih"><h3>{judul}</h3><div class="desc">{deskripsi}</div></div>""",
        unsafe_allow_html=True,
    )


def plot_shap_gelap(fungsi_plot, *args, judul="", tinggi=4.2, **kwargs):
    """Render plot SHAP dengan latar gelap supaya serasi dengan tema aplikasi."""
    warna_teks = "#e8eaed"
    with plt.rc_context({
        "figure.facecolor": "#171a21", "axes.facecolor": "#171a21",
        "text.color": warna_teks, "axes.labelcolor": warna_teks,
        "xtick.color": warna_teks, "ytick.color": warna_teks,
        "axes.edgecolor": "#262b36", "savefig.facecolor": "#171a21",
    }):
        fig = plt.figure(figsize=(9, tinggi))
        fungsi_plot(*args, show=False, **kwargs)
        if judul:
            plt.title(judul, color=warna_teks, fontsize=11, pad=12)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)


@st.cache_resource(show_spinner="Memuat model prediksi...")
def _model():
    return muat_model()


@st.cache_data(show_spinner="Membaca berkas CSV...")
def _baca(berkas_bytes: bytes) -> pd.DataFrame:
    return baca_events_csv(io.BytesIO(berkas_bytes))


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <span class="pill">Prototipe Penelitian</span>
  <h1>Prediksi Risiko Penurunan Perilaku Belanja E-Commerce Kosmetik</h1>
</div>
""", unsafe_allow_html=True)

try:
    model = _model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Sumber Data")
    berkas = st.file_uploader(
        "Berkas CSV log event", type=["csv"],
        help="Struktur kolom mengikuti dataset eCommerce Events History in Cosmetics Shop.",
    )
    st.markdown("---")
    st.markdown("### Konfigurasi Model")
    st.markdown(
        f"<div style='font-size:.85rem;line-height:1.9;color:#9aa0ac'>"
        f"Algoritma &nbsp;<b>XGBoost</b><br>"
        f"Fitur &nbsp;<b>{', '.join(FITUR)}</b><br>"
        f"Ambang batas &nbsp;<b>{THRESHOLD_RISIKO:.2f}</b><br>"
        f"Pelanggan tanpa pembelian &nbsp;<b>tidak diskor model</b>"
        f"</div>", unsafe_allow_html=True,
    )

# ===========================================================================
    if berkas is None:
        panel_buka(
            "Mulai dengan mengunggah berkas CSV",
            "Gunakan panel di sebelah kiri untuk mengunggah log aktivitas pelanggan. "
            "Setiap baris mewakili satu event, dan satu pelanggan dapat memiliki banyak "
            "baris pada beberapa sesi berbeda.",
        )
        with st.expander("Struktur kolom yang diharapkan", expanded=True):
            st.markdown(
                "Berkas harus memuat sembilan kolom berikut, sama seperti dataset "
                "**eCommerce Events History in Cosmetics Shop**:"
            )
            st.code(", ".join(KOLOM_WAJIB), language="text")
            st.markdown(
                "- `event_type` berisi salah satu dari `view`, `cart`, "
                "`remove_from_cart`, atau `purchase`\n"
                "- `event_time` berformat `2019-10-01 00:00:00 UTC`\n"
                "- `price` dipakai untuk menghitung Monetary, dan hanya baris "
                "`purchase` yang diperhitungkan\n"
                "- `category_code` dan `brand` boleh kosong"
            )
    else:
        try:
            df_events = _baca(berkas.getvalue())
        except ValueError as e:
            st.error(str(e))
            st.stop()

        info = ringkas_events(df_events)

        st.markdown("#### Ringkasan berkas")
        k = st.columns(4)
        kpi(k[0], "Baris event", f"{info['n_baris']:,}")
        kpi(k[1], "Pelanggan", f"{info['n_pelanggan']:,}", gaya="accent")
        kpi(k[2], "Sesi", f"{info['n_sesi']:,}")
        kpi(k[3], "Transaksi pembelian", f"{info['n_purchase']:,}")

        st.write("")
        kiri, kanan = st.columns([1.15, 1])
        with kiri:
            st.markdown("**Sebaran jenis event**")
            st.bar_chart(
                pd.Series(info["sebaran_event"], name="jumlah").sort_values(ascending=False),
                color="#e0518a", height=220,
            )
            if info["event_tak_dikenal"]:
                st.warning(
                    f"Jenis event di luar empat jenis yang dikenal ditemukan: "
                    f"{', '.join(info['event_tak_dikenal'])}. Baris tersebut tetap "
                    f"dihitung sebagai aktivitas, namun tidak dianggap pembelian."
                )

        with kanan:
            st.markdown("**Periode analisis**")
            st.caption(
                "Fitur R, F, dan M dinormalisasi terhadap panjang periode, sehingga "
                "periode yang dipilih menentukan skala nilainya. Nilai bawaan diambil "
                "dari rentang waktu berkas yang diunggah."
            )
            awal_default, akhir_default = periode_default(df_events)
            c1, c2 = st.columns(2)
            tgl_awal = c1.date_input("Tanggal awal", value=awal_default.date())
            tgl_akhir = c2.date_input("Tanggal akhir", value=akhir_default.date())

            periode_awal = pd.Timestamp(tgl_awal, tz="UTC")
            periode_akhir = pd.Timestamp(tgl_akhir, tz="UTC")
            n_hari = (periode_akhir - periode_awal).total_seconds() / 86400.0

            if n_hari <= 0:
                st.error("Tanggal akhir harus berada setelah tanggal awal.")
                st.stop()
            st.markdown(
                f"<div style='color:#9aa0ac;font-size:.86rem'>Panjang periode "
                f"<b style='color:#e0518a'>{n_hari:.0f} hari</b></div>",
                unsafe_allow_html=True,
            )

        st.write("")
        if st.button("Hitung RFMC dan Jalankan Prediksi", type="primary"):
            with st.spinner("Menghitung fitur RFMC dari log event..."):
                rfmc = hitung_rfmc_dari_events(df_events, periode_awal, n_hari)
            with st.spinner("Menjalankan prediksi..."):
                hasil = model.skor_rfmc(rfmc)
            st.session_state["hasil"] = hasil
            st.session_state["n_hari"] = n_hari

        if "hasil" in st.session_state:
            hasil = st.session_state["hasil"]
            n_total = len(hasil)
            n_model = int(hasil["diskor_model"].sum())
            n_berisiko = int((hasil["kategori"] == LABEL_BERISIKO).sum())
            n_aman = int((hasil["kategori"] == LABEL_TIDAK_BERISIKO).sum())
            n_pasif = int((hasil["kategori"] == LABEL_PASIF).sum())

            st.markdown("---")
            st.markdown("#### Hasil prediksi")
            k = st.columns(4)
            kpi(k[0], "Total pelanggan", f"{n_total:,}")
            kpi(k[1], "Berisiko", f"{n_berisiko:,}",
                f"{n_berisiko/n_total*100:.1f}% dari total" if n_total else "",
                gaya="warn")
            kpi(k[2], "Diperkirakan aktif", f"{n_aman:,}",
                f"{n_aman/n_total*100:.1f}% dari total" if n_total else "",
                gaya="ok")
            kpi(k[3], "Pasif", f"{n_pasif:,}", "tanpa transaksi, tidak diskor model")

            if n_pasif:
                st.info(
                    f"{n_pasif:,} pelanggan tidak memiliki satu pun transaksi pembelian "
                    f"pada periode ini. Mereka dikategorikan Pasif melalui aturan dan "
                    f"sengaja tidak dilewatkan ke model, sebab model hanya dilatih pada "
                    f"pelanggan yang pernah melakukan pembelian.",
                    icon="ℹ️",
                )

            kolom_tampil = [
                "user_id", "R", "F", "M", "C", "F_count", "M_sum",
                "skor_risiko", "kategori",
            ]
            tabel = hasil[kolom_tampil].sort_values(
                "skor_risiko", ascending=False, na_position="last"
            )
            st.dataframe(
                tabel.style.format({
                    "R": "{:.4f}", "F": "{:.4f}", "M": "{:.4f}", "C": "{:.4f}",
                    "M_sum": "{:.2f}", "skor_risiko": "{:.4f}",
                }),
                use_container_width=True, height=380,
            )

            buf = io.StringIO()
            hasil.to_csv(buf, index=False)
            st.download_button(
                "Unduh hasil lengkap (CSV)", data=buf.getvalue(),
                file_name="hasil_prediksi_risiko_pelanggan.csv", mime="text/csv",
            )

            if n_model:
                st.write("")
                kiri2, kanan2 = st.columns([1, 1.25])
                with kiri2:
                    st.markdown("**Sebaran skor risiko**")
                    st.caption(
                        f"Hanya mencakup {n_model:,} pelanggan yang diskor model. "
                        f"Garis keputusan berada di {THRESHOLD_RISIKO:.2f}."
                    )
                    skor = hasil.loc[hasil["diskor_model"], "skor_risiko"]
                    bins = pd.cut(skor, bins=20)
                    sebaran = bins.value_counts().sort_index()
                    sebaran.index = [f"{i.left:.2f}" for i in sebaran.index]
                    st.bar_chart(sebaran, color="#e0518a", height=260)

                with kanan2:
                    st.markdown("**Kontribusi fitur terhadap prediksi**")
                    st.caption(
                        "Nilai SHAP menjelaskan skor keluaran model. Warna merah "
                        "menandakan nilai fitur yang tinggi, biru menandakan rendah."
                    )
                    sub = hasil.loc[hasil["diskor_model"]].head(2000)
                    sv = model.shap_lokal(sub)
                    plot_shap_gelap(shap.plots.beeswarm, sv, tinggi=3.6)

# ===========================================================================