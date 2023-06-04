# Laporan Proyek Machine Learning - Jelvin Krisna Putra 

## Domain Proyek
Scope yang diambil merupakan scope **Ekonomi dan bisnis**. Seperti yang kita ketahui hotel sekarang sudah menjadi peluang bisnis yang menjanjikan, dan kebanyakan pengusaha hotel sudah menerapkan sistem informasi ke dalam layanan mereka, contohnya aplikasi untuk pemesanan hotel hingga check-in bisa melalui gadget pintar di tangan konsumen.

Banyak yang menjadi pertimbangan konsumen dalam memilih merk hotel hingga tempat hotel, namun dalam dunia bisnis, pertimbangan terbesar pastilah jatuh ke pada harga sewa dari suatu hotel. dikutip dari Tjiptono dan Chandra (dalam [Pengaruh Harga Dan Fasilitas Terhadap Keputusan Menginap Tamu Di Hotel Best Western Premier The Hive Jakarta](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=pengaruh+harga+sewa+hotel&oq=#d=gs_qabs&t=1685846684825&u=%23p%3DmEx_4ez85jYJ)) dikatakan bahwa ...ada segmen pembeli yang sangat sensitif terhadap faktor harga (menjadikan harga sebagai satu-satunya pertimbangan membeli produk) dan ada pula yang tidak.

sehingga dalam project kali ini saya sebagai data science diminta air-bnb untuk membuat model machine learning untuk memprediksi harga sewa yang cocok terhadap suatu hotel air-bnb berdasarkan data-data yang ada seperti lokasi, nama, dan lainnya.


## Business Understanding

### Problem Statements

Adapun permasalahan yang dialami oleh air-bnb adalah sebagai berikut:

1. Rumitnya menentukan harga sewa yang optimal yang sesuai dengan data-data yang ada

2. Pasar penyewaan Airbnb di NYC sangat kompetitif, karena setiap pesaing ingin mendapatkan pelanggan mereka masing-masing

3. Meningkatkan pendapatan perusahaan agar bisa bersaing untuk jangka waktu yang panjang

### Goals

Dari permasalahan di atas dapat diketahui tujuan dari proyek ini adalah:

1. Menentukan harga yang optimal pada suatu hotel berdasarkan data-data lampau yang tersedia dari perusahaan

2. Menentukan harga yang sesuai dan tepat agar tidak tersaingi oleh kompetitor yang lain

3. Memajukan dan mempertahankan perusahaan agar tetap bisa berdiri kokoh dan tahan lama

Beberapa solusi yang bisa diterapkan untuk melatih model bisa dengan menggunakan lebih dari satu algoritma seeprti, K Nearest Neighbors, Boosting Algorithm, dan Random Forest dan menentukan algoritma mana yang paling akurat dan terbaik untuk menentukan harga sewa, serta mengukur nilai atau tingkat akurasi model menggunakan metrik Mean Square Root

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah New York City Airbnb Open Data. Dataset ini berisi informasi tentang listing Airbnb di New York City. Dataset ini dapat diunduh melalui platform Kaggle melalui tautan berikut: [New York City Airbnb Open Data on Kaggle](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data).

Variabel-variabel pada dataset New York City Airbnb Open Data adalah sebagai berikut:

1. id: ID unik untuk setiap record

2. name: Nama hotel

3. host_id: ID unik untuk setiap host Airbnb

4. host_name: Nama host Airbnb

5. neighbourhood_group: Kelompok/neighbourhood geografis di New York City (misalnya Manhattan, Brooklyn, dll.)

6. neighbourhood: Nama lingkungan/neighbourhood di New York City

7. latitude: Koordinat lintang dari lokasi hotel

8. longitude: Koordinat bujur dari lokasi hotel

9. room_type: Tipe kamar (misalnya "Entire home/apt", "Private room", "Shared room")

10. price: Harga sewa per malam dalam dolar AS

11. minimum_nights: Jumlah malam minimum yang harus dipesan

12. number_of_reviews: Jumlah ulasan yang diberikan oleh tamu sebelumnya

13. last_review: Tanggal ulasan terakhir

14. reviews_per_month: Jumlah ulasan per bulan

15. calculated_host_listings_count: Jumlah listing yang dimiliki oleh host

16. availability_365: Jumlah hari dalam setahun dimana listing tersedia untuk disewa

Untuk memahami data dengan lebih baik, beberapa tahapan yang dapat dilakukan antara lain adalah:

1. Menggunakan visualisasi data seperti histogram, scatter plot, atau box plot untuk mendapatkan distribusi data dan adanya data outlier.

2. Menggunakan EDA untuk menganalisis korelasi antara harga sewa dengan atribut lainnya atau distribusi harga sewa berdasarkan tipe kamar atau lingkungan geografis.

## Data Preparation

Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 

- Menjelaskan proses data preparation yang dilakukan

- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 

- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.

- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.

- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation

Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:

- Penjelasan mengenai metrik yang digunakan

- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 

- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_

- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_

- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
