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

Link dataset: [https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)

---

## Data Preparation

Beberapa tahapan data preparation yang dilakukan:

1. **Parsing dan konversi tipe data**: Mengonversi kolom `Date` menjadi tipe `datetime`.
2. **Sorting data**: Mengurutkan berdasarkan tanggal.
3. **Normalisasi nilai harga**: Menghindari skala besar yang dapat mengganggu pelatihan model.
4. **Membuat windowing**: Menyusun `X` sebagai urutan harga historis dan `y` sebagai harga penutupan 1 hari ke depan hingga 7 hari ke depan.
5. **Split data**: Menggunakan data historis 100 hari terakhir untuk training dan validasi, serta membuat prediksi untuk 7 hari ke depan.

---

## 🧠 Modeling

### 📌 Model 1: Long Short-Term Memory (LSTM)

#### 🔍 Cara Kerja LSTM

Long Short-Term Memory (LSTM) merupakan arsitektur dari Recurrent Neural Network (RNN) yang dirancang untuk mengatasi masalah *vanishing gradient* dan mengingat informasi dalam jangka waktu panjang. LSTM bekerja dengan menggunakan struktur yang disebut **sel memori**, serta tiga gerbang utama:

* **Forget Gate**: Memutuskan informasi mana dari state sebelumnya yang akan dibuang.
* **Input Gate**: Menentukan informasi baru apa yang akan disimpan di sel.
* **Output Gate**: Menentukan output dari sel berdasarkan state saat ini dan input.

Model ini sangat efektif untuk data deret waktu seperti harga Bitcoin karena dapat menangkap pola dan tren jangka panjang.

---

#### ⚙️ Parameter Model

Berikut parameter utama yang digunakan dalam model:

* `n_steps = 60`

  > Jumlah timestep (60 hari sebelumnya digunakan untuk memprediksi harga hari ke-61)

* **Lapisan LSTM**:

  * Jumlah unit: *default (misalnya 50/64)* – menangani representasi pola data secara sekuensial
  * `return_sequences=True` – untuk lapisan bertingkat

* **Lapisan Dense (output)**:

  * Unit: 1 – menghasilkan satu nilai prediksi (harga Bitcoin)

* **Fungsi Aktivasi**:

  * LSTM menggunakan `tanh` secara default
  * Dense output tidak menggunakan aktivasi karena ini adalah regresi

* **Optimizer**: `adam`

  * Optimizer adaptif default yang bekerja baik untuk sebagian besar kasus

* **Loss Function**: `mean_squared_error`

  * Cocok untuk regresi

* **Batch Size**: 64

  * Ukuran batch untuk setiap langkah pelatihan

* **Epochs**: 20

  * Jumlah maksimum iterasi pelatihan

* **Callbacks**:

  * `EarlyStopping(monitor='val_loss', patience=5)`

    > Menghentikan pelatihan lebih awal jika tidak ada perbaikan selama 5 epoch berturut-turut

---

#### ✅ Kelebihan LSTM (Opsional)

* Mampu mengingat pola jangka panjang
* Cocok untuk data time-series seperti harga Bitcoin
* Menangani fluktuasi yang tidak beraturan dalam data

#### ⚠️ Kekurangan LSTM (Opsional)

* Waktu pelatihan relatif lama dibanding model ML tradisional
* Membutuhkan tuning parameter yang cermat
* Bisa mengalami overfitting tanpa regularisasi atau dropout

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

| Metrik | Nilai  |
| ------ | ------ |
| MAE    | \~2322 |
| RMSE   | \~3453 |
| R²     | \~0.98 |

Interpretasi:

* Nilai R² yang mendekati 1 menunjukkan bahwa model mampu menjelaskan sebagian besar variasi harga.
* Nilai MAE dan RMSE yang relatif rendah terhadap nilai harga Bitcoin (sekitar 90.000 - 100.000) menunjukkan performa yang baik untuk prediksi jangka pendek.

### 🔗 Kaitan dengan Business Understanding

* **Problem Statement**: Membuat model prediksi harga penutupan Bitcoin
* **Goals**: Membangun model prediksi yang akurat berdasarkan data historis
* **Solution**: Model LSTM mampu mengenali pola harga sebelumnya dan digunakan untuk prediksi harga ke depan

Interpretasi:

* Nilai R² yang mendekati 1 menunjukkan bahwa model mampu menjelaskan sebagian besar variasi harga.
* Nilai MAE dan RMSE yang relatif rendah terhadap nilai harga Bitcoin (sekitar 90.000 - 100.000) menunjukkan performa yang baik untuk prediksi jangka pendek.

