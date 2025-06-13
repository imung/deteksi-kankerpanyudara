
# 🩺 Aplikasi Web Deteksi Dini Kanker Payudara

Aplikasi ini memungkinkan pengguna melakukan deteksi dini kanker payudara berdasarkan fitur numerik dari citra histopatologi menggunakan model Machine Learning (Random Forest).

---

## 📁 Struktur File

```
.
├── app_streamlit.py             # Web App Streamlit
├── train_model_10fitur.py       # Skrip pelatihan ulang model dengan 10 fitur
├── breast-cancer.csv            # Dataset Breast Cancer Wisconsin
├── model_bc_randomforest.pkl    # Model ML (dibuat setelah training)
├── scaler_bc.pkl                # Skaler input (dibuat setelah training)
```

---

## 🚀 Langkah Menjalankan Aplikasi Web

### 1. Install Dependency
Pastikan Python 3.7+ dan pip sudah terpasang.

Install Streamlit dan scikit-learn:
```bash
pip install streamlit scikit-learn pandas numpy joblib
```

---

### 2. Latih Ulang Model
Jalankan skrip pelatihan model (gunakan 10 fitur utama):

```bash
python train_model_10fitur.py
```

Ini akan menghasilkan file:
- `model_bc_randomforest.pkl`
- `scaler_bc.pkl`

---

### 3. Jalankan Aplikasi Streamlit

```bash
streamlit run app_streamlit.py
```

Buka browser:
```
http://localhost:8501
```

---

## 🧪 Fitur Input

| Fitur                    | Keterangan                          |
|--------------------------|--------------------------------------|
| radius_mean              | Rata-rata radius sel                |
| texture_mean             | Rata-rata tekstur sel               |
| perimeter_mean           | Rata-rata perimeter sel             |
| area_mean                | Luas rata-rata sel                  |
| smoothness_mean          | Tingkat kehalusan sel               |
| compactness_mean         | Kompaksi sel                        |
| concavity_mean           | Cekungan sel                        |
| concave points_mean      | Jumlah titik cekungan               |
| symmetry_mean            | Simetri sel                         |
| fractal_dimension_mean   | Kompleksitas fraktal sel            |

---

## ✅ Output
Model akan memberikan hasil:
- 💖 Jinak (Benign)
- ⚠️ Ganas (Malignant)

---

## 📝 Catatan
Pastikan input jumlah fitur sesuai model yang dilatih. Model ini hanya bekerja dengan **10 fitur utama**, dan file `.pkl` harus dihasilkan ulang jika ingin menambah/mengurangi fitur.

---

## 📃 Lisensi
MIT License © 2025
