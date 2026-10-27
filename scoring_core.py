"""
Modul scoring inti untuk prototipe prediksi risiko pelanggan toko kosmetik daring.

Modul ini sengaja dipisah dari antarmuka Streamlit (app.py) supaya logika
prediksi dan penjelasan SHAP dapat dipakai ulang oleh pembungkus apa pun di
masa depan, baik antarmuka web, REST API, maupun proses batch terjadwal, tanpa
perlu menulis ulang logikanya.

Model: XGBoost pada empat fitur RFMC (Recency, Frequency, Monetary, Clumpiness),
yaitu model dengan performa terbaik pada perbandingan tiga model di tahap
evaluasi penelitian.

Definisi target: pelanggan diprediksi "berisiko" apabila pada periode
berikutnya ia diperkirakan jatuh ke segmen aktivitas terendah atau tidak
tercatat berbelanja sama sekali.

Gerbang pelanggan tanpa pembelian: model hanya dilatih pada pelanggan yang
memiliki minimal satu transaksi pembelian. Pelanggan tanpa pembelian sama
sekali tidak pernah dilewatkan ke model, melainkan langsung dikategorikan
"Pasif" melalui aturan, persis seperti perlakuan pada tahap segmentasi dalam
penelitian. Menskor mereka dengan model berarti memakai model di luar cakupan
data latihnya.
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
# Konfigurasi. Nilai di bawah HARUS sama dengan hasil akhir penelitian.
# ---------------------------------------------------------------------------
DIREKTORI_MODUL = Path(__file__).resolve().parent
MODEL_PATH = DIREKTORI_MODUL / "model" / "xgboost_jalur1_model_final_tuned.joblib"

FITUR = ["R", "F", "M", "C"]  # urutan harus sama dengan saat pelatihan

# Ambang batas hasil threshold tuning dengan objektif MCC pada validation set.
THRESHOLD_RISIKO = 0.80

LABEL_BERISIKO = "Berisiko tidak kembali aktif"
LABEL_TIDAK_BERISIKO = "Diperkirakan tetap aktif"
LABEL_PASIF = "Pasif (tanpa transaksi pembelian)"


@dataclass
class HasilScoring:
    skor_risiko: float
    label: str
    berisiko: bool


class ModelScoring:
    """Pembungkus tipis di atas model dan explainer SHAP-nya.

    Pembuatan objek ini mahal (memuat model dan membangun explainer), sehingga
    pada app.py objek ini di-cache dan dipakai berulang, bukan dibuat ulang
    setiap interaksi pengguna.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Berkas model tidak ditemukan di: {model_path}\n\n"
                f"Pastikan berkas 'xgboost_jalur1_model_final_tuned.joblib' berada "
                f"di dalam folder 'model/' pada direktori yang sama dengan "
                f"scoring_core.py, atau sesuaikan MODEL_PATH di berkas ini."
            )
        self.model = joblib.load(model_path)
        self.explainer = shap.Explainer(self.model)

    # -- inti ---------------------------------------------------------------
    def skor_rfmc(self, df_rfmc: pd.DataFrame) -> pd.DataFrame:
        """Skor tabel RFMC, dengan gerbang aturan untuk pelanggan tanpa pembelian.

        Masukan wajib memuat kolom R, F, M, C. Kolom 'ada_pembelian' bersifat
        opsional; bila tidak ada, seluruh baris dianggap pelanggan pembeli.

        Keluaran berupa salinan masukan ditambah kolom:
            skor_risiko : probabilitas keluaran model (NaN untuk kategori Pasif)
            kategori    : label akhir yang dapat dibaca manusia
            berisiko    : True / False / NA
            diskor_model: True bila baris ini benar-benar dilewatkan ke model
        """
        _validasi_kolom(df_rfmc)
        hasil = df_rfmc.copy()

        if "ada_pembelian" in hasil.columns:
            mask_model = hasil["ada_pembelian"].astype(int) == 1
        else:
            mask_model = pd.Series(True, index=hasil.index)

        hasil["skor_risiko"] = np.nan
        hasil["berisiko"] = pd.NA
        hasil["kategori"] = LABEL_PASIF
        hasil["diskor_model"] = mask_model.values

        if mask_model.any():
            X = hasil.loc[mask_model, FITUR].astype(float)
            proba = self.model.predict_proba(X)[:, 1]
            berisiko = proba >= THRESHOLD_RISIKO
            hasil.loc[mask_model, "skor_risiko"] = proba
            hasil.loc[mask_model, "berisiko"] = berisiko
            hasil.loc[mask_model, "kategori"] = np.where(
                berisiko, LABEL_BERISIKO, LABEL_TIDAK_BERISIKO
            )

        return hasil


    def shap_lokal(self, df_rfmc: pd.DataFrame):
        """Nilai SHAP untuk baris-baris yang diberikan.

        SHAP menjelaskan skor keluaran model, bukan label hasil pemotongan
        ambang batas, sehingga nilainya tidak berubah bila ambang batas diubah.
        """
        _validasi_kolom(df_rfmc)
        return self.explainer(df_rfmc[FITUR].astype(float))


def _validasi_kolom(df_rfmc: pd.DataFrame) -> None:
    hilang = [k for k in FITUR if k not in df_rfmc.columns]
    if hilang:
        raise ValueError(
            f"Kolom fitur tidak ditemukan: {hilang}. "
            f"Dibutuhkan kolom {FITUR} (huruf besar)."
        )


def muat_model(model_path: Optional[Path] = None) -> ModelScoring:
    """Titik masuk tunggal untuk memuat model."""
    return ModelScoring(model_path or MODEL_PATH)
