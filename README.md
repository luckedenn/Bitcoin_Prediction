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

## 📊 Data Preparation

Beberapa tahapan *data preparation* yang dilakukan sebelum pelatihan model:

1. **Parsing dan konversi tipe data**
   Kolom `Timestamp` dikonversi menjadi format `datetime` agar dapat diolah sebagai data deret waktu, lalu diatur sebagai indeks dengan `set_index`.

2. **Resampling harian**
   Data harga Bitcoin di-*resample* menjadi data harian (`'D'`) untuk mengurangi kepadatan dan menyusun deret waktu yang konsisten.

3. **Pemilihan fitur**
   Hanya kolom `Close` yang digunakan untuk prediksi karena merepresentasikan harga penutupan harian Bitcoin.

4. **Normalisasi data**
   Skala harga dinormalisasi menggunakan `MinMaxScaler` ke rentang \[0, 1] agar model LSTM lebih stabil dan cepat belajar.

5. **Membuat sequence (windowing)**
   Sequence dibuat dengan **60 hari sebelumnya sebagai input (`X`)** dan **1 hari setelahnya sebagai target (`y`)**.
   Tujuan windowing ini adalah untuk melatih model mengenali pola harga historis jangka pendek.

6. **Pembagian data latih dan uji**
   Dataset dibagi menjadi data **train** dan **test** dengan rasio sekitar 80:20 setelah sequence dibuat.
   Ini memastikan model dievaluasi dengan data yang belum pernah dilihat selama pelatihan.

---

## 🧠 Modeling

### 📌 Model 1: Long Short-Term Memory (LSTM)

#### 🔍 Cara Kerja LSTM

LSTM (Long Short-Term Memory) adalah arsitektur dari Recurrent Neural Network (RNN) yang dirancang untuk mengenali pola dalam data sekuensial, terutama yang memiliki ketergantungan jangka panjang.
LSTM memiliki **sel memori internal** dan tiga gerbang utama yang mengatur informasi yang disimpan atau dibuang:

* **Forget Gate**: Menentukan informasi dari state sebelumnya yang perlu dilupakan.
* **Input Gate**: Menentukan informasi baru yang akan ditambahkan ke memori.
* **Output Gate**: Menghasilkan output berdasarkan memori dan input terkini.

Kemampuan ini sangat bermanfaat untuk memprediksi data deret waktu seperti harga Bitcoin.

---

### ⚙️ Arsitektur Model

Model dibangun menggunakan Keras `Sequential` API sebagai berikut:

```python
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(n_steps, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
```

Penjelasan setiap layer:

* **LSTM(50, return\_sequences=True)**

  * Layer LSTM pertama dengan 50 unit, memproses urutan input dan mengembalikan seluruh sequence ke layer berikutnya.
  * `return_sequences=True` diperlukan karena akan diteruskan ke layer LSTM kedua.

* **LSTM(50)**

  * Layer LSTM kedua dengan 50 unit, hanya mengembalikan output dari langkah terakhir untuk dijadikan input ke layer Dense.

* **Dense(1)**

  * Layer output dengan satu unit neuron karena output model adalah satu nilai harga (regresi).

---

### 🔧 Konfigurasi Pelatihan

| Parameter           | Nilai                    | Penjelasan                                                                |
| ------------------- | ------------------------ | ------------------------------------------------------------------------- |
| `n_steps`           | 60                       | Input berupa 60 hari data historis harga penutupan                        |
| Fungsi Aktivasi     | `tanh` (default di LSTM) | Cocok untuk data time-series                                              |
| Loss Function       | `mean_squared_error`     | Umum digunakan untuk regresi                                              |
| Optimizer           | `adam`                   | Optimizer adaptif yang cepat konvergen                                    |
| Batch Size          | 64                       | Ukuran mini-batch selama pelatihan                                        |
| Epochs              | 20                       | Iterasi maksimum pelatihan model                                          |
| Callback (opsional) | `EarlyStopping`          | Menghentikan pelatihan saat validasi stagnan (tidak disebut di kode awal) |

---

### ✅ Kelebihan LSTM

* Mampu mengingat pola jangka panjang secara efektif.
* Sangat cocok untuk data time-series yang memiliki pola musiman atau tren.
* Menangani ketidakberaturan data lebih baik daripada model klasik.

### ⚠️ Kekurangan LSTM

* Pelatihan memerlukan waktu lebih lama.
* Rentan terhadap overfitting jika tidak disertai dengan teknik regulasi atau validasi silang.
* Parameter cukup banyak dan memerlukan tuning agar optimal.

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
Seperti yang dijabarkan dalam Business Understanding, model ini ditujukan untuk membantu investor dalam mengantisipasi pergerakan harga Bitcoin.
Model berhasil memberikan prediksi harga yang akurat dan stabil, dengan error relatif kecil terhadap harga aktual. Hal ini dapat digunakan sebagai dasar pengambilan keputusan investasi seperti entry/exit point dan analisis risiko.


