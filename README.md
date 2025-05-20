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

