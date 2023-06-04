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

15. calculated_host_listings_count: Jumlah hotel yang dimiliki oleh host

16. availability_365: Jumlah hari dalam setahun dimana hotel tersedia untuk disewa

Untuk memahami data dengan lebih baik, beberapa tahapan yang dapat dilakukan antara lain adalah:

1. Menggunakan visualisasi data seperti histogram, scatter plot, atau box plot untuk mendapatkan distribusi data dan adanya data outlier.

2. Menggunakan EDA untuk menganalisis korelasi antara harga sewa dengan atribut lainnya atau distribusi harga sewa berdasarkan tipe kamar atau lingkungan geografis.

## Data Preparation

Beberapa teknik yang dapat diterapkan untuk membersihkan dan mempersiapkan data Airbnb New York City adalah sebagai berikut:

1. Data Cleaning:

   - Menghapus kolom yang tidak relevan atau tidak digunakan dalam analisis, seperti 'id', 'host_id', 'name', 'host_name', dan 'last_review' yang tidak memberikan pengaruh dalam pemodelan harga sewa.

   - Menangani missing values: Melakukan input nilai pada review_per_month dengan nilai mean-nya dan penghapusan baris price missing values.

2. Outlier Detection:

   - Mengidentifikasi outlier pada variabel numerik seperti 'latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365'

   - Menangani outliers dengan Inter Quartile Range method.

   - yang dilanjutkan dengan Univariate Analisis dan Multivariate Analysis

3. Normalize Data:

   - mulai dari encoding fitur kategori menjadi angka 1 0 dengan one hot encoding.

   - Reduksi dimensi dengan Principal Component Analysis (PCA).

   - Pembagian dataset dengan fungsi train_test_split dari library sklearn dengan skala 9:1.

   - Melakukan normalisasi atau standarisasi pada variabel numerik dengan StandardScaler method.

## Modeling

Pada tahap ini, kita akan menggunakan tiga algoritma berbeda, yaitu K-Nearest Neighbors (KNN), Random Forest, dan Boosting Algorithm.

1. K-Nearest Neighbors (KNN):

   - Algoritma yang berbasis pada jarak antara data dimana akan mencari K tetangga terdekat dari suatu data baru dan memprediksi nilai target berdasarkan kedekatan tetangga.

   - (+) KNN mudah dipahami dan dapat digunakan untuk masalah regresi.

   - (-) pemilihan nilai K yang tepat menjadi penting dalam performa algoritma ini.

2. Random Forest:

   - Algoritma ensemble yang terdiri dari banyak decision tree sehingga membentuk hutan (forest) dimana setiap pohon dipelajari pada data acak dan kemudian dicari rata²nya atau menggunakan konsep bagging.

   - (+) RF mampu cenderung lebih cepat daripada Boosting karena RF berkonsep bagging sehingga jalannya paralel.

   - (-) Random Forest dapat memakan waktu yang cukup lama dalam proses pelatihan pada dataset yang sangat besar.

3. Boosting Algorithm (misalnya Gradient Boosting, AdaBoost):

   - Algoritma yang membangun model berurutan secara adaptif dengan bentuk seperti barisan.

   - (+) memiliki performa yang tinggi dan mampu menangani data yang rumit. Algoritma ini dapat membantu mengurangi bias dan varian dalam model.

   - (-) Proses pelatihan Boosting Algorithm cenderung lebih lama karna menggunakan konsep boosting dimana dieksekusi secara iteratif kebalik dari bagging di RF.

Dan untuk model terbaik jika tidak dilihat dari metriknya adalah model Random Forest yang dipilih karena karakteristik data yang tidak berkaitan satu sama lain tidak seperti harga emas atau berlian pada data kali ini lebih ke seperti pohon keputusan untuk karakteristik datanya yang selanjutnya baru kita masuk ke evaluasi metrik.

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
