# Laporan Proyek Machine Learning - Lucas Chandra

## Domain Proyek

Perdagangan aset kripto, khususnya Bitcoin, telah menjadi tren utama dalam dunia keuangan digital. Volatilitas tinggi pada harga Bitcoin menjadikannya sangat menarik namun juga berisiko bagi investor dan trader. Oleh karena itu, prediksi harga Bitcoin dalam jangka pendek sangat penting guna membantu pengambilan keputusan investasi yang lebih bijak dan terukur.

Banyak penelitian menunjukkan bahwa pendekatan machine learning dapat memberikan performa yang lebih baik dibandingkan metode statistik tradisional dalam melakukan forecasting harga aset digital \[1]. Salah satu pendekatan umum adalah menggunakan regresi untuk memodelkan harga penutupan (closing price) di masa depan berdasarkan nilai historis.

\[1] S. J. B. Yoo, S. Kim, and S. Yoon, "Bitcoin price forecasting using deep learning algorithm," *IEEE Access*, vol. 9, pp. 99900–99909, 2021.

## Business Understanding

### Problem Statements

1. Bagaimana cara memprediksi harga penutupan Bitcoin dalam 7 hari ke depan?
2. Seberapa baik akurasi hasil prediksi harga Bitcoin menggunakan model regresi berbasis machine learning?

### Goals

1. Membangun model prediksi harga penutupan Bitcoin berbasis regresi machine learning.
2. Mengevaluasi performa model menggunakan metrik evaluasi regresi seperti MAE, RMSE, dan R².

### Solution Statements

* Menggunakan model Linear Regression untuk melakukan prediksi harga penutupan 7 hari ke depan.
* Mengimprovisasi akurasi model dengan preprocessing data dan eksplorasi fitur.
* Evaluasi performa dilakukan dengan menggunakan MAE, RMSE, dan R² agar hasil prediksi dapat dinilai secara kuantitatif.

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
4. **Membuat label target**: Menyusun `y` sebagai harga penutupan 1 hari ke depan hingga 7 hari ke depan.
5. **Split data**: Menggunakan data historis 100 hari terakhir untuk training dan validasi, serta membuat prediksi untuk 7 hari ke depan.

## Modeling

Model yang digunakan:

* **Linear Regression** dari pustaka `sklearn.linear_model`

Alasan pemilihan:

* Sederhana dan cepat dalam inferensi.
* Cocok untuk baseline prediksi deret waktu sederhana.

### Parameter

Tidak dilakukan tuning parameter eksplisit karena Linear Regression tidak memerlukan parameter awal selain input `X` dan target `y`.

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

**---Ini adalah bagian akhir laporan---**

> *Catatan tambahan:*
>
> * Visualisasi hasil prediksi disertakan dalam notebook untuk menunjukkan prediksi terhadap harga aktual.
> * Proyek ini dapat dikembangkan lebih lanjut dengan LSTM atau algoritma time series lainnya untuk peningkatan performa.
