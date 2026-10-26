"""
Modul scoring inti (Tahap 6 Deployment) -- SENGAJA DIPISAH dari antarmuka
Streamlit (app.py) supaya logika prediksi + penjelasan SHAP ini bisa dipakai
ulang oleh pembungkus (wrapper) apapun di masa depan -- Streamlit sekarang,
REST API (FastAPI/Flask) atau batch job terjadwal nanti kalau benar-benar
diintegrasikan ke Customer Data Platform (CDP) Wings Store sesuai visi BAB
III Subbab 3.2.1 -- tanpa perlu menulis ulang logika prediksinya.

Model yang dipakai: Jalur I (XGBoost pada fitur RFMC) -- model dengan
performa terbaik pada perbandingan 3 arah Tahap 5 (menang di seluruh 5
metrik: F1, ROC-AUC, MCC, PR-AUC kelas0, PR-AUC kelas1).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap

# ---------------------------------------------------------------------------
# Konfigurasi -- NILAI INI HARUS SAMA PERSIS DENGAN NOTEBOOK TAHAP 4/5 JALUR I
# ---------------------------------------------------------------------------
# PENTING: path model dibuat RELATIF terhadap lokasi file ini sendiri (bukan
# absolute path SageMaker seperti versi sebelumnya), supaya file ini bisa
# jalan baik di SageMaker MAUPUN di Streamlit Community Cloud -- di kedua
# tempat itu, repo/folder tempat app.py + scoring_core.py berada HARUS juga
# menyertakan folder model/ berisi file .joblib ini (lihat panduan deploy).
DIREKTORI_MODUL = Path(__file__).resolve().parent
MODEL_PATH = DIREKTORI_MODUL / "model" / "xgboost_jalur1_model_final_tuned.joblib"

FITUR = ["R", "F", "M", "C"]  # URUTAN HARUS SAMA dgn saat training -- jangan diubah urutannya

# Dari ringkasan_tahap4_jalur1.json -> tahap4_threshold_tuning.threshold_terpilih
THRESHOLD_RISIKO = 0.45

LABEL_BERISIKO = "Berisiko (potensi penurunan kemandirian digital)"
LABEL_TIDAK_BERISIKO = "Tidak berisiko (self-service stabil)"


@dataclass
class HasilScoring:
    """Satu baris hasil scoring -- dipakai baik oleh mode manual (1 baris)
    maupun mode CSV batch (banyak baris, dikumpulkan jadi DataFrame)."""

    skor_risiko: float
    label: str
    berisiko: bool


class ModelScoringJalurI:
    """Pembungkus (wrapper) tipis di atas model Jalur I + explainer SHAP-nya.
    Dibuat sekali (mahal: load model + bangun explainer), dipakai berulang
    kali untuk setiap pelanggan/batch -- karena itu di app.py ini akan
    di-cache (st.cache_resource), bukan dibuat ulang setiap interaksi user."""

    def __init__(self, model_path: Path = MODEL_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan di: {model_path}\n"
                f"Pastikan notebook Tahap 4 Jalur I sudah dijalankan & model "
                f"tersimpan di path itu, atau ubah MODEL_PATH di scoring_core.py "
                f"kalau lokasi model Anda berbeda."
            )
        self.model = joblib.load(model_path)
        self.explainer = shap.Explainer(self.model)

    def skor_batch(self, df_rfmc: pd.DataFrame) -> pd.DataFrame:
        """Input: DataFrame dengan kolom R, F, M, C (boleh ada kolom lain,
        akan diabaikan -- kecuali kolom identitas yg ikut dipertahankan).
        Output: DataFrame yang sama + kolom skor_risiko, label, berisiko."""
        _validasi_kolom(df_rfmc)
        X = df_rfmc[FITUR].astype(float)
        proba = self.model.predict_proba(X)[:, 1]

        hasil = df_rfmc.copy()
        hasil["skor_risiko"] = proba
        hasil["berisiko"] = proba >= THRESHOLD_RISIKO
        hasil["label"] = np.where(hasil["berisiko"], LABEL_BERISIKO, LABEL_TIDAK_BERISIKO)
        return hasil

    def skor_satu_pelanggan(self, r: float, f: float, m: float, c: float) -> HasilScoring:
        """Mode manual (Tab 1 Streamlit): 1 pelanggan, 4 angka RFMC."""
        df_satu = pd.DataFrame([{"R": r, "F": f, "M": m, "C": c}])
        hasil = self.skor_batch(df_satu).iloc[0]
        return HasilScoring(
            skor_risiko=float(hasil["skor_risiko"]),
            label=str(hasil["label"]),
            berisiko=bool(hasil["berisiko"]),
        )

    def shap_lokal(self, df_rfmc: pd.DataFrame):
        """Hitung SHAP value utk baris-baris di df_rfmc (bisa 1 baris utk
        mode manual, atau banyak baris utk mode CSV batch) -- dipakai utk
        visualisasi waterfall (1 pelanggan) atau beeswarm/bar (agregat
        beberapa pelanggan) di app.py. Menggunakan TreeSHAP (otomatis
        dipilih oleh shap.Explainer utk model berbasis pohon seperti
        XGBoost) -- lihat https://shap.readthedocs.io/ utk referensi API."""
        _validasi_kolom(df_rfmc)
        X = df_rfmc[FITUR].astype(float)
        return self.explainer(X)


def _validasi_kolom(df_rfmc: pd.DataFrame) -> None:
    kolom_hilang = [k for k in FITUR if k not in df_rfmc.columns]
    if kolom_hilang:
        raise ValueError(
            f"Kolom wajib tidak ditemukan di data: {kolom_hilang}. "
            f"Data harus punya kolom: {FITUR} (huruf besar, sesuai definisi RFMC "
            f"di BAB III)."
        )


def muat_model(model_path: Optional[Path] = None) -> ModelScoringJalurI:
    """Factory function -- titik masuk tunggal utk memuat model, dipanggil
    oleh app.py (di-cache) maupun oleh skrip/API lain di masa depan."""
    return ModelScoringJalurI(model_path or MODEL_PATH)
