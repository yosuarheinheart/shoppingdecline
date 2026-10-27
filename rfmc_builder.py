"""
Pembangun fitur RFMC dari log event mentah.

Modul ini adalah PORTING PERSIS dari notebook Tahap 3.3 (Rekayasa Fitur RFMC)
yang dipakai saat melatih model. Setiap rumus di bawah disalin apa adanya dari
sana, termasuk kasus tepi dan nilai sentinel, karena nilai RFMC yang dihasilkan
aplikasi ini WAJIB identik dengan nilai yang dilihat model saat pelatihan. Kalau
berbeda sedikit saja, model akan menerima input di luar distribusi latihnya dan
menghasilkan prediksi yang keliru tanpa pesan error apa pun.

Ringkasan rumus (dari Tahap 3.3):
    t_days  = (event_time - awal_periode) dalam satuan hari
    F_count = jumlah event bertipe "purchase"
    M_sum   = jumlah price dari event bertipe "purchase"
    F       = F_count / N_hari          (rate per hari)
    M       = M_sum   / N_hari          (rate per hari)
    R       = (N_hari - t_days_terakhir) / N_hari   (proporsi 0..1)
    C       = clumpiness berbasis entropi dari jarak antar-pembelian

Kasus tepi (sesuai definisi di Tahap 3.3, bukan tebakan):
    n = 0  -> R=1,0 ; F=0 ; M=0 ; C=0 ; ada_pembelian=0   (sentinel)
    n = 1  -> C=1,0 persis
    n >= 2 -> C dihitung lewat compute_clumpiness()

Sumber rumus clumpiness: Zhang, Bradlow & Small, "New Measures of Clumpiness
for Incidence Data", companion open-access dari paper utama Marketing Science
34(2):195-208, DOI 10.1287/mksc.2014.0873
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KOLOM_WAJIB = [
    "event_time", "event_type", "product_id", "category_id",
    "category_code", "brand", "price", "user_id", "user_session",
]

JENIS_EVENT_DIKENAL = {"view", "cart", "remove_from_cart", "purchase"}


def compute_clumpiness(t_sorted, N: float) -> float:
    """Disalin persis dari Tahap 3.3. t_sorted terurut menaik, panjang n >= 2."""
    t = np.asarray(t_sorted, dtype=float)
    n = len(t)

    gaps = np.empty(n + 1, dtype=float)
    gaps[0] = t[0]
    gaps[1:n] = t[1:] - t[:-1]
    gaps[n] = N - t[-1]

    gaps[gaps == 0.0] = 1.0  # hindari log(0)

    p = gaps / gaps.sum()
    Hp_raw = float(np.sum(p * np.log(p)))
    Hp_min = float(np.log(1.0 / (n + 1)))
    Hp_max = 0.0
    return (Hp_raw - Hp_min) / (Hp_max - Hp_min)


def validasi_kolom_events(df: pd.DataFrame) -> list[str]:
    """Kembalikan daftar kolom wajib yang tidak ada. Kosong berarti lolos."""
    return [k for k in KOLOM_WAJIB if k not in df.columns]


def baca_events_csv(sumber) -> pd.DataFrame:
    """Baca CSV log event dan ubah event_time jadi datetime ber-timezone UTC.

    Format event_time pada dataset asli adalah '2019-10-01 00:00:00 UTC'.
    Akhiran ' UTC' dibuang lebih dulu agar parsing konsisten, lalu zona waktu
    ditetapkan sebagai UTC, sama seperti data periodisasi pada Tahap 3.2.
    """
    df = pd.read_csv(sumber)
    hilang = validasi_kolom_events(df)
    if hilang:
        raise ValueError(
            f"Kolom wajib tidak ditemukan pada CSV: {hilang}. "
            f"Kolom yang dibutuhkan: {KOLOM_WAJIB}"
        )

    waktu = df["event_time"].astype(str).str.replace(r"\s*UTC$", "", regex=True)
    df["event_time"] = pd.to_datetime(waktu, errors="coerce", utc=True)

    n_gagal = int(df["event_time"].isna().sum())
    if n_gagal == len(df):
        raise ValueError(
            "Seluruh nilai event_time gagal diurai. Pastikan formatnya "
            "seperti '2019-10-01 00:00:00 UTC'."
        )
    if n_gagal:
        df = df.loc[df["event_time"].notna()].copy()

    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    return df


def periode_default(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Usulan periode analisis: dari awal hari event pertama sampai akhir hari
    event terakhir pada data yang diunggah."""
    awal = df["event_time"].min().normalize()
    akhir = df["event_time"].max().normalize() + pd.Timedelta(days=1)
    return awal, akhir


def hitung_rfmc_dari_events(
    df_events: pd.DataFrame,
    periode_awal: pd.Timestamp,
    n_hari: float,
) -> pd.DataFrame:
    """Porting persis build_rfmc_table() dari Tahap 3.3.

    Populasi keluaran adalah SELURUH user_id yang punya minimal satu event
    jenis apa pun pada data, sama seperti definisi populasi di Tahap 3.3.
    """
    if n_hari <= 0:
        raise ValueError(f"n_hari harus lebih besar dari nol, diterima: {n_hari}")

    all_users = df_events["user_id"].unique()

    purchases = df_events.loc[
        df_events["event_type"] == "purchase", ["user_id", "event_time", "price"]
    ].copy()
    purchases["t_days"] = (
        purchases["event_time"] - periode_awal
    ).dt.total_seconds() / 86400.0

    if len(purchases):
        agg = purchases.groupby("user_id").agg(
            F_count=("event_time", "size"),
            M_sum=("price", "sum"),
            R_last_t=("t_days", "max"),
        )
    else:
        agg = pd.DataFrame(
            columns=["F_count", "M_sum", "R_last_t"],
            index=pd.Index([], name="user_id"),
        )

    rfmc = pd.DataFrame(index=pd.Index(all_users, name="user_id"))
    rfmc = rfmc.join(agg, how="left")

    rfmc["ada_pembelian"] = rfmc["F_count"].notna().astype("int8")
    rfmc["F_count"] = rfmc["F_count"].fillna(0)
    rfmc["M_sum"] = rfmc["M_sum"].fillna(0.0)

    rfmc["F"] = rfmc["F_count"] / n_hari
    rfmc["M"] = rfmc["M_sum"] / n_hari
    rfmc["R"] = np.where(
        rfmc["ada_pembelian"] == 1,
        (n_hari - rfmc["R_last_t"]) / n_hari,
        1.0,
    )

    c_values = pd.Series(0.0, index=rfmc.index)
    c_values.loc[rfmc["ada_pembelian"] == 1] = 1.0  # n=1, ditimpa bila n>=2

    users_multi = agg.index[agg["F_count"] >= 2] if len(agg) else []
    if len(users_multi) > 0:
        sub = purchases[purchases["user_id"].isin(users_multi)].sort_values(
            ["user_id", "t_days"]
        )
        grouped_c = sub.groupby("user_id")["t_days"].apply(
            lambda s: compute_clumpiness(s.values, n_hari)
        )
        c_values.loc[grouped_c.index] = grouped_c.values

    rfmc["C"] = c_values

    return rfmc.reset_index()[
        ["user_id", "R", "F", "M", "C", "ada_pembelian", "F_count", "M_sum"]
    ]


def ringkas_events(df_events: pd.DataFrame) -> dict:
    """Statistik ringkas untuk ditampilkan sebelum perhitungan dijalankan."""
    sebaran = df_events["event_type"].value_counts()
    tak_dikenal = sorted(set(sebaran.index) - JENIS_EVENT_DIKENAL)
    return {
        "n_baris": len(df_events),
        "n_pelanggan": int(df_events["user_id"].nunique()),
        "n_sesi": int(df_events["user_session"].nunique()),
        "n_produk": int(df_events["product_id"].nunique()),
        "waktu_awal": df_events["event_time"].min(),
        "waktu_akhir": df_events["event_time"].max(),
        "sebaran_event": sebaran.to_dict(),
        "n_purchase": int((df_events["event_type"] == "purchase").sum()),
        "event_tak_dikenal": tak_dikenal,
    }
