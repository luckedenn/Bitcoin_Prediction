# Laporan Proyek Machine Learning - Lucas Chandra

## Domain Proyek

Perdagangan aset kripto, khususnya Bitcoin, telah menjadi tren utama dalam dunia keuangan digital. Volatilitas tinggi pada harga Bitcoin menjadikannya sangat menarik namun juga berisiko bagi investor dan trader. Oleh karena itu, prediksi harga Bitcoin dalam jangka pendek sangat penting guna membantu pengambilan keputusan investasi yang lebih bijak dan terukur.

Banyak penelitian menunjukkan bahwa pendekatan deep learning seperti Long Short-Term Memory (LSTM) dapat memberikan performa yang lebih baik dibandingkan metode statistik tradisional dalam melakukan forecasting harga aset digital \[1]. Salah satu pendekatan umum adalah menggunakan arsitektur LSTM untuk memodelkan harga penutupan (closing price) di masa depan berdasarkan nilai historis.

\[1] J. McNally, J. Roche, and S. Caton, “Predicting the price of Bitcoin using Machine Learning,” in *2018 26th Euromicro International Conference on Parallel, Distributed and Network-based Processing (PDP)*, Cambridge, 2018, pp. 339–343. doi: [https://doi.org/10.1109/PDP2018.2018.00060](https://doi.org/10.1109/PDP2018.2018.00060)

## 📌 Business Understanding

### 🎯 Problem Statements

1. **Volatilitas harga Bitcoin** yang tinggi menyulitkan investor dalam mengambil keputusan investasi jangka pendek.  
2. Belum tersedia sistem prediksi yang mampu memberikan **perkiraan harga penutupan Bitcoin 7 hari ke depan** secara andal untuk membantu mitigasi risiko pasar.  
3. Sejauh mana **akurasi model deep learning** seperti LSTM mampu menangkap pola historis dan memprediksi harga secara akurat?

---

### 🎯 Goals

1. **Membangun model prediksi harga penutupan Bitcoin** 7 hari ke depan dengan memanfaatkan model LSTM berbasis deep learning.  
2. Menyediakan **sistem prediksi yang dapat mendukung pengambilan keputusan** investasi jangka pendek dengan memperkirakan tren harga.  
3. **Mengukur kinerja model** menggunakan metrik regresi seperti MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), dan R² Score.

---

### 🧩 Solution Statements

- Menggunakan **model LSTM (Long Short-Term Memory)** karena kemampuannya dalam memahami data sekuensial dan menangani long-term dependencies yang umum pada data time-series seperti harga aset kripto.  
- Melakukan **pembersihan data dan normalisasi** agar model dapat belajar dari data yang konsisten dan berkualitas tinggi.  
- Menggunakan **MAE, RMSE, dan R²** untuk mengevaluasi sejauh mana model mampu mendekati harga aktual dan seberapa besar error-nya dalam konteks bisnis.  

Berikut adalah versi **markdown lengkap dan akurat** dari notebook kamu, termasuk:

* Struktur dataset (`Timestamp`, `Close`)
* Pembentukan sequence LSTM
* Pembagian data train-test
* Proses training dengan **EarlyStopping**
* Evaluasi model

Markdown ini bisa langsung digunakan di notebook untuk keperluan submission atau dokumentasi:

---

## 📊 Data Understanding

Dataset yang digunakan adalah data historis harga Bitcoin dari Kaggle, terdiri dari 7.032.685 baris dan 6 kolom. Namun, untuk proyek ini, hanya kolom `Timestamp` dan `Close` yang digunakan.

### 🔍 Informasi Fitur Dataset

| Fitur       | Tipe Data | Deskripsi                                   |
| ----------- | --------- | ------------------------------------------- |
| `Timestamp` | datetime  | Tanggal dan waktu pencatatan harga Bitcoin  |
| `Close`     | float64   | Harga penutupan Bitcoin pada waktu tersebut |

Data telah dikonversi dari format UNIX timestamp ke `datetime`, dan diatur sebagai index time series.

```python
df.head()
```

| Timestamp  | Close    |
| ---------- | -------- |
| 2012-01-01 | 4.645697 |
| 2012-01-02 | 4.975000 |
| 2012-01-03 | 5.085500 |
| 2012-01-04 | 5.170396 |
| 2012-01-05 | 5.954361 |

### ✅ Pemeriksaan Kondisi Data

* **Missing values**: Tidak ditemukan nilai kosong
* **Data duplikat**: Tidak ditemukan duplikat dalam indeks `Timestamp`
* **Distribusi waktu**: Harian (daily) berdasarkan `Timestamp` sebagai index

---

## 🧠 Modeling

### Model: LSTM (Long Short-Term Memory)

#### 🧩 Teknik Preprocessing

* **Normalisasi**: `MinMaxScaler` pada kolom `Close` untuk merubah nilai ke skala \[0, 1]
* **Sliding window**: Data dibentuk menjadi urutan dengan `n_steps = 60` (menggunakan 60 hari terakhir untuk memprediksi hari berikutnya)
* **Reshaping**: Data diubah ke bentuk `[samples, timesteps, features]` agar sesuai dengan input LSTM

#### 🧪 Split Train & Test

* **Rasio**: 80% data untuk pelatihan, 20% untuk pengujian
* **Penyesuaian tanggal**: Data target (`test_dates`) disesuaikan dengan jumlah langkah (`n_steps`)

```python
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]
```

#### ⚙️ Arsitektur Model & Training

* **Model**: LSTM
* **Epochs**: 20
* **Batch size**: 64
* **Early Stopping**: Diterapkan dengan `patience=5` untuk menghentikan pelatihan jika `val_loss` tidak membaik
* **Validation**: Menggunakan data test sebagai validation set

```python
early_stop = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)
```

#### 🧠 Keunggulan LSTM

* Menangani ketergantungan jangka panjang dalam data time series
* Lebih stabil dibanding RNN standar dalam menangani data historis

---

## 📈 Evaluation

### 📊 Metrik Evaluasi

Model dievaluasi menggunakan metrik regresi berikut:

| Metrik                             | Deskripsi                                                       |
| ---------------------------------- | --------------------------------------------------------------- |
| **MAE** (Mean Absolute Error)      | Rata-rata dari selisih absolut antara nilai aktual dan prediksi |
| **RMSE** (Root Mean Squared Error) | Penalti lebih besar untuk error yang besar                      |
| **R² Score**                       | Seberapa besar variasi target dijelaskan oleh model             |

### 📌 Hasil Evaluasi

Contoh hasil evaluasi (nilai akan disesuaikan setelah evaluasi model dijalankan):

| Metrik | Nilai  |
| ------ | ------ |
| MAE    | \~2322 |
| RMSE   | \~3453 |
| R²     | \~0.98 |

### 🔗 Kaitan dengan Business Understanding

* **Problem Statement**: Membuat model prediksi harga penutupan Bitcoin
* **Goals**: Membangun model prediksi yang akurat berdasarkan data historis
* **Solution**: Model LSTM mampu mengenali pola harga sebelumnya dan digunakan untuk prediksi harga ke depan

---

Jika kamu ingin melengkapi bagian ini dengan **visualisasi loss** (`loss vs val_loss`) atau hasil prediksi vs aktual, saya bisa bantu buatkan juga. Cukup beri tahu!

## Data Understanding

Dataset yang digunakan adalah data historis harga Bitcoin dari Kaggle berjudul "Bitcoin Historical Data" yang mencakup fitur seperti `Open`, `High`, `Low`, `Close`, `Volume`, dan `Market Cap`.

Link dataset: [https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)

### Fitur yang digunakan:

* **Date**: Tanggal pencatatan.
* **Close**: Harga penutupan Bitcoin per hari.

Fokus utama fitur adalah kolom `Close` yang menjadi target prediksi.

## Data Preparation

Beberapa tahapan data preparation yang dilakukan:

1. **Parsing dan konversi tipe data**: Mengonversi kolom `Date` menjadi tipe `datetime`.
2. **Sorting data**: Mengurutkan berdasarkan tanggal.
3. **Normalisasi nilai harga**: Menghindari skala besar yang dapat mengganggu pelatihan model.
4. **Membuat windowing**: Menyusun `X` sebagai urutan harga historis dan `y` sebagai harga penutupan 1 hari ke depan hingga 7 hari ke depan.
5. **Split data**: Menggunakan data historis 100 hari terakhir untuk training dan validasi, serta membuat prediksi untuk 7 hari ke depan.

## Modeling

Model yang digunakan:

* **LSTM** dari pustaka `tensorflow.keras`

Alasan pemilihan:

* Mampu menangani dependensi jangka panjang pada data sekuensial seperti deret waktu harga.
* Lebih unggul dibanding model regresi linear dalam mengenali pola kompleks historis.

### Parameter

* Model Sequential dengan 1 atau lebih layer LSTM dan Dense output.
* Optimizer: Adam
* Loss Function: Mean Squared Error (MSE)
* Epochs dan batch size ditentukan melalui eksperimen (misalnya 50 epoch, batch size 32)

## Evaluation

Model dievaluasi menggunakan tiga metrik:

* **MAE (Mean Absolute Error)**: Rata-rata selisih absolut antara nilai aktual dan prediksi.
* **RMSE (Root Mean Squared Error)**: Akar dari rata-rata kuadrat error; memberikan penalti lebih tinggi untuk error besar.
* **R² Score (Koefisien Determinasi)**: Mengukur proporsi variansi data target yang dapat dijelaskan oleh model.

### Hasil Evaluasi:

* **MAE**  : 2322.14
* **RMSE** : 3452.64
* **R²**   : 0.98

Interpretasi:

* Nilai R² yang mendekati 1 menunjukkan bahwa model mampu menjelaskan sebagian besar variasi harga.
* Nilai MAE dan RMSE yang relatif rendah terhadap nilai harga Bitcoin (sekitar 90.000 - 100.000) menunjukkan performa yang baik untuk prediksi jangka pendek.

