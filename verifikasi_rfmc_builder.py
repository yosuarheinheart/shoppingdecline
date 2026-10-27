"""
Pembuktian bahwa rfmc_builder.py menghasilkan nilai RFMC yang IDENTIK dengan
notebook Tahap 3.3 yang dipakai saat melatih model.

Skrip ini dijalankan di lingkungan SageMaker Anda, bukan di tempat lain, sebab
ia membaca dua berkas yang hanya ada di sana:
    output/tahap3_data_preparation/data_clean_tahap3_2_periodisasi.parquet
    output/tahap3_data_preparation/RFMC_P1.parquet

Cara kerja: log event Periode 1 dibaca ulang, dihitung RFMC-nya memakai
rfmc_builder.py, lalu hasilnya dibandingkan baris per baris dengan RFMC_P1.parquet
asli. Bila seluruh selisihnya nol, maka modul yang dipakai aplikasi terbukti
mereplikasi Tahap 3.3 dengan tepat.

Jalankan:
    python verifikasi_rfmc_builder.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rfmc_builder import hitung_rfmc_dari_events  # noqa: E402

PROJECT_ROOT = Path("/home/sagemaker-user/project")
TAHAP3_DIR = PROJECT_ROOT / "output" / "tahap3_data_preparation"
PERIODISASI_PATH = TAHAP3_DIR / "data_clean_tahap3_2_periodisasi.parquet"
RFMC_P1_PATH = TAHAP3_DIR / "RFMC_P1.parquet"

# Nilai ini disalin persis dari PERIODE_INFO pada notebook Tahap 3.3
N_HARI_P1 = 92.0
AWAL_P1 = pd.Timestamp("2019-10-01 00:00:00", tz="UTC")

for p in (PERIODISASI_PATH, RFMC_P1_PATH):
    if not p.exists():
        raise FileNotFoundError(f"Berkas tidak ditemukan: {p}")

print("Membaca data periodisasi...")
df = pd.read_parquet(PERIODISASI_PATH, columns=["user_id", "event_time", "event_type",
                                                "price", "periode"])
df_p1 = df.loc[df["periode"] == "Periode_1"]
print(f"  Periode 1: {len(df_p1):,} baris")

print("Menghitung ulang RFMC memakai rfmc_builder.py...")
hasil = hitung_rfmc_dari_events(df_p1, AWAL_P1, N_HARI_P1)
print(f"  Menghasilkan {len(hasil):,} baris")

print("Membaca RFMC_P1.parquet asli...")
asli = pd.read_parquet(RFMC_P1_PATH)
print(f"  Berisi {len(asli):,} baris")

assert len(hasil) == len(asli), (
    f"Jumlah baris berbeda: hasil hitung ulang {len(hasil):,}, asli {len(asli):,}"
)

gabung = asli.merge(hasil, on="user_id", how="inner", suffixes=("_asli", "_baru"))
assert len(gabung) == len(asli), "Ada user_id yang tidak cocok antara kedua tabel"

print("\n" + "=" * 70)
print("PERBANDINGAN PER KOLOM")
print("=" * 70)
semua_cocok = True
for kolom in ["R", "F", "M", "C", "ada_pembelian", "F_count", "M_sum"]:
    a = gabung[f"{kolom}_asli"].to_numpy(dtype=float)
    b = gabung[f"{kolom}_baru"].to_numpy(dtype=float)
    selisih_maks = float(np.nanmax(np.abs(a - b)))
    n_beda = int((np.abs(a - b) > 1e-9).sum())
    status = "IDENTIK" if n_beda == 0 else f"BERBEDA pada {n_beda:,} baris"
    semua_cocok &= (n_beda == 0)
    print(f"  {kolom:14s} selisih absolut maksimum = {selisih_maks:.3e}   {status}")

print("=" * 70)
if semua_cocok:
    print("HASIL: rfmc_builder.py TERBUKTI identik dengan notebook Tahap 3.3.")
    print("Aplikasi aman dipakai, sebab nilai RFMC yang masuk ke model akan sama")
    print("persis dengan nilai yang dilihat model saat pelatihan.")
else:
    print("HASIL: DITEMUKAN PERBEDAAN. JANGAN pakai aplikasi untuk mengambil")
    print("kesimpulan apa pun sebelum perbedaan ini ditelusuri. Kirimkan keluaran")
    print("lengkap skrip ini supaya dapat diperiksa.")
print("=" * 70)
