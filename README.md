# Laporan Proyek Machine Learning - Jelvin Krisna Putra 

## Domain Proyek

_Scope_ yang diambil merupakan _scope_ **Ekonomi dan bisnis**. Seperti yang kita ketahui hotel sekarang sudah menjadi peluang bisnis yang menjanjikan, dan kebanyakan pengusaha hotel sudah menerapkan sistem informasi ke dalam layanan mereka, contohnya aplikasi untuk pemesanan hotel hingga _check-in_ bisa melalui gadget pintar di tangan konsumen.

Banyak yang menjadi pertimbangan konsumen dalam memilih merk hingga tempat hotel, namun dalam dunia bisnis, pertimbangan terbesar jatuh ke pada harga sewa dari suatu hotel. menurut Tjiptono (seperti yang dikutip dari [[1]](https://core.ac.uk/download/pdf/268049173.pdf)) dikatakan bahwa ada segmen pembeli yang sangat sensitif terhadap faktor harga (menjadikan harga sebagai satu-satunya pertimbangan membeli produk) dan ada pula yang tidak.

"salah satu tujuan perusahaan itu adalah [sic!]menciptakaan loyalitas pelanggan maka suatu perusahaan yang baik, misalnya penetapan harga yang tepat sesuai dengan pangsa pasar dan kondisi perekonomian masyarakat sekitar dapat menciptakan kepuasan konsumen, konsumen yang merasa puas dapat terciptanya loyalitas pelanggan." [[2]](https://jurnal.darmajaya.ac.id/index.php/jmmd/article/view/989)

Pentingnya harga sewa dalam loyalitas konsumen dan kepuasan pelanggan terkadang membuat banyak perusahaan rela melakukan berbagai macam cara untuk bisa mendapatkan harga sewa yang paling optimal tanpa harus mengorbankan pendapatan perusahaan, "Kepuasan pelanggan merupakan salah satu faktor penting yang harus 
diperhatikan perusahaan karena pelanggan merupakan alasan mengapa suatu perusahaan 
eksis." [[3]](https://www.journal.stieamkop.ac.id/index.php/yume/article/view/3036) karena dengan harga sewa yang tepat akan ada banyak konsumen yang menginap di hotel _Airbnb_ yang akan meningkatkan pendapatan dari _Airbnb_ itu sendiri

Oleh karena itu dalam project kali ini saya sebagai _data science_ diminta oleh perusahaan _Airbnb_ untuk membuat _model machine learning_ yang mampu memprediksi harga sewa yang cocok terhadap suatu hotel airbnb berdasarkan data-data yang ada seperti lokasi, nama, dan data lainnya.

## Business Understanding

### Problem Statements
Adapun permasalahan yang dialami oleh airbnb adalah sebagai berikut:
1. Bagaimana cara menentukan harga sewa suatu unit atau hotel yang optimal dan sesuai dengan kepuasan konsumen?
2. Mengapa harga sewa perlu diprediksi dan disesuaikan dengan kepuasan konsumen?
3. Apa yang diharapkan dari prediksi harga sewa untuk unit dari hotel _Airbnb_ jangka pendek maupun jangka panjangnya?

### Goals
Dari permasalahan di atas dapat diketahui tujuan dari proyek ini adalah:
1. Menentukan harga yang optimal pada suatu hotel berdasarkan data-data lampau yang tersedia dari perusahaan
2. Menentukan harga yang sesuai dan tepat agar tidak tersaingi oleh kompetitor yang lain
3. Memajukan dan mempertahankan perusahaan agar tetap bisa berdiri kokoh dan tahan lama

Beberapa solusi yang diterapkan untuk melatih model adalah dengan menggunakan lebih dari satu algoritma yaitu, *Random Forest, Boosting Algorithm, MLPRegressor, dan SVR* yang penentuan parameter setiap algoritma akan dilakukan dengan *hyperparameter* menggunakan teknik *Random Search Cross Validation* yang hasil terbaiknya kita teruskan ke *Grid Search Cross Validation*, agar menghasilkan parameter yang terbaik karena menentukan parameter pada kasus regresi itu sangat mempengaruhi hasil model dan dapat menghindari *overfitting*, dan terakhir menentukan algoritma mana yang paling akurat dan terbaik untuk menentukan harga sewa, serta mengukur tingkat akurasi model menggunakan metrik *Mean Square Error*.

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
1. Load Data:
   - Membaca file csv dengan _library pandas_ yang ditampilkan ke dalam variabel _hotels_ dengan tipe data dataframe, lalu menampilkan data awal, akhir dan jumlah data yang tersedia baris x kolom
2. Data Cleaning:
   - Karena sudah dalam bentuk dataframe kita bisa menghapus kolom yang tidak relevan atau tidak digunakan dengan memanggil method _drop_ dalam analisis, seperti 'id', 'host_id', 'name', 'host_name', dan 'last_review' yang tidak memberikan pengaruh dalam pemodelan harga se

Tabel 1. Hasil dari pengecekan nilai NaN pada semua kolom
| Feature                         | Missing Values |
|---------------------------------|----------------|
| neighbourhood_group             | 0              |
| neighbourhood                   | 0              |
| latitude                        | 0              |
| longitude                       | 0              |
| room_type                       | 0              |
| price                           | 0              |
| minimum_nights                  | 0              |
| number_of_reviews               | 0              |
| reviews_per_month               | 10052          |
| calculated_host_listings_count  | 0              |
| availability_365                | 0              |
   - Dari tabel 1 bisa dilihat bahwa kolom *review per month* memiliki _NaN value_ pada 10052 record sehingga jika dilakukakn penghapusan data, maka akan mempengaruhi hasil model, sehingga langkah yang dilakukan adalah melakukan input nilai yang *NaN* pada kolom _review per month_ dengan nilai rata-rata dari kolom yang sama.
   - melakukan penghapusan baris _price_ untuk setiap record yang _price_ bernilai 0 karena dalam kolom _price_ yang bernilai 0 hanya berjumlah 11 record sehingga tidak akan terlalu berpengaruh terhadap model yang akan dilatih. 
3. Outlier Detection:

![Outlier](https://raw.githubusercontent.com/krisna31/air-bnb-rent-price-analytics/main/images/1-outlier.png)

Gambar 1. *boxplot minimum nights*
   - Mengidentifikasi outlier pada variabel numerik seperti latitude, longitude, hingga minimum nights dengan menggunakan visualisasi data boxplot dari library *seaborn* bisa dilihat di gambar 1
   - Outliers dihapus dengan menggunakan metode IQR (Interquartile Range), dimana kita harus ?menghitung rentang IQR dengan mengurangi Q1 dari Q3, yaitu IQR = Q3 - Q1, Q3 merupakan quartile ketiga yang didapatkan dari fungsi _quartile(0.75)_ dan 0.25 untuk Q1

![2. Histogram](https://github.com/krisna31/air-bnb-rent-price-analytics/raw/main/images/2-histogram.png)

Gambar 2. *Neighborhood Group Histogram*
   - Selanjutnya kita masuk ke Univariate Analisis dimana data yang dianalisa adalah data-data kategorical seperti *neighborhood_group*, *neighborhood*, dan *room_type* dimana mendapatkan hasil untuk hotel di wilayah *Brooklyn and Manhattan* adalah hotel dengan biaya sewa termahal diantara 4 tempat bisa dilihat pada gambar 2

![3. Neighborhood ](https://github.com/krisna31/air-bnb-rent-price-analytics/raw/main/images/3-neighbourhood.png)

Gambar 3. *Neighborhood*
   - Untuk kolom *neighborhood* dibuang karena jika kita lakukan proses *one hot encoding* pada kolom ini maka akan menghasilkan terlalu banyak kolom baru yang akan berpengaruh ke hasil model bisa dilihat di gambar 3 atau 4.

![4. catplot ](https://github.com/krisna31/air-bnb-rent-price-analytics/raw/main/images/4hh.png)

Gambar 4. *Catplot*
   - Lalu ke Multivariate Analysis dimana ada kata multi berarti kita akan menganalisa banyak varian data yang pada kasus kali ini kita menganalisa data-data numerikal mulai dari *longitude* hingga ke *availability 365* menggunakan *library seaborn* dan *pairplot* serta *correlation matrix with heatmap*, dimana disimpulkan bahwa kesemua data memiliki penyebaran data yang serupa satu sama lain sehingga untuk data *numerikal* tidak ada yang di buang bisa dilihat untuk gambar 5 dan 6.

![6 heatmap](https://github.com/krisna31/air-bnb-rent-price-analytics/raw/main/images/6-heatmap.png)

Gambar 5. *Heatmap Correlation Matrix*

![pairplot](https://github.com/krisna31/air-bnb-rent-price-analytics/raw/main/images/5-spread.jpg)

Gambar 6. *Pairplot*
4. Normalisasi Data:
   - Menggunakan konsep *one hot encoding* untuk mengubah kolom kategorical menjadi numerical dimana jika ada kolom review dengan nilai positif dan negatif maka akan menghasilkan 2 kolom baru dengan nama kolom positif dan negatif dimana nilainya akan bernilai 1 atau 0 sebagai representasi dari nilai pada saat masih dalam 1 kolom.

![7](https://github.com/krisna31/air-bnb-rent-price-analytics/raw/main/images/7.png)

Gambar 7. *Longitude and latitude*
   - Karena tidak ada *feature* yang sangat mempengaruhi harga lihat gambar 7, maka tidak dilakukan Reduksi dimensi dengan Principal Component Analysis (PCA).
   - Pembagian dataset dengan fungsi train_test_split dari library sklearn dengan skala 80:10 dikarenakan jumlah data yang tidak terlalu banyak ada total 27764 data bersih yang tersedia dimana 80%-nya adalah 22211 dan 20%-nya ada 5553
   - Melakukan normalisasi data-data numerical agar bisa mendekati distribusi normal menggunakan *library* dari *StandardScaler* untuk melakukan skalabilitas dari data *numerikal* dengan cara data numerik diubah menjadi memiliki *mean* 0 dan *standar deviasi* 1, dilakukan hanya pada data latih setelah dibagi agar tidak mengotori data test sehingga model bisa benar-benar dinilai apakah bisa menerima data baru atau tidak.

## Modeling

Pada tahap ini, kita akan menggunakan empat algoritma berbeda, yaitu Random Forest, Boosting Algorithm, MLPRegressor, dan SVR
1. _Random Forest_:
   - Algoritma ensemble (gabungan dari algoritma lain menjadi satu) yang terdiri dari banyak _decision tree_ sehingga membentuk _(forest)_ dimana dibagi menjadi 2 lagi yaitu  _bagging_ dan _boosting_ untuk *boosting* akan dijelaskan di poin kedua, *bagging* disini maksudnya adalah setiap model akan melatih dengan sampel yang berbeda dan setiap sampel tidak mempengaruhi hasil dan sampel yang lain, alasan digunakannya algoritma ini adalah karena algoritma ini bisa menyelesaikan masalah regresi dan dari penyebaran data-nya yang tinggi maka algoritma ini yang dipilih dalam kasus kali ini.
   - Adapun kelebihan dari algoritma ini adalah _Random Forest_ mampu cenderung lebih cepat daripada _Boosting_ karena _Random Forest_ menggunakan konsep _bagging_ sehingga jalannya paralel.
   - Kekurangan dari algoritma ini adalah _Random Forest_ dapat memakan waktu yang cukup lama dalam proses pelatihan pada dataset yang sangat besar.
3. Boosting Algorithm (AdaBoost):
   - Algoritma yang membangun model berurutan secara adaptif dengan bentuk seperti barisan.
   - Adapun kelebihan dari algoritma ini adalah memiliki performa yang tinggi dan mampu menangani data yang rumit. Algoritma ini dapat membantu mengurangi bias dan varian dalam model.
   - Kekurangan dari algoritma ini adalah proses pelatihan _Boosting Algorithm_ yang cenderung lebih lama karena menggunakan konsep _boosting_ dimana tiap iterasi akan dieksekusi secara iteratif berkebalikan dari konsep _bagging_ yang terdapat di algoritma _Randlm Forest_
Dan untuk model terbaik jika tidak dilihat dari metriknya adalah model Random Forest yang dipilih karena karakteristik data yang tidak berkaitan satu sama lain tidak seperti harga emas atau berlian pada data kali ini lebih ke seperti pohon keputusan untuk karakteristik datanya yang selanjutnya baru kita masuk ke evaluasi metrik.

## Evaluation

Metrik evaluasi yang akan digunakan untuk menganalisis kinerja model dalam memprediksi harga sewa Airbnb di New York City adalah Mean Squared Error (MSE). Metrik ini sesuai untuk masalah regresi linier.

MSE = $\frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2$

Mean Squared Error (MSE): MSE menghitung error dari rata-rata kuadrat selisih antara nilai prediksi dan nilai sebenarnya. Sederhananya semakin rendah MSE, semakin baik performa model.

Berdasarkan analisis menggunakan metrik evaluasi MSE dan R2, hasil proyek didapatkan sebagai berikut:

|   |   |   |   |   |
|---|---|---|---|---|
|   |   |   |   |   |
|   |   |   |   |   |
|   |   |   |   |   |

![MSE Grafik](https://raw.githubusercontent.com/krisna31/air-bnb-rent-price-analytics/main/MSE-Grafik.jpg)

MSE yang rendah ditunjukkan pada algoritma random forest. namun masih terjadi overfitting dibuktikan dengan MSE rendah hanya terjadi pada data train dan masih high error di data test, namun jika dibandingkan dengan algoritma KNN dan boosting, error pada Random Forest Lah yang paling kecil. Dengan demikian, kita dapat menyimpulkan bahwa model Random Forest yang dipilih menjadi model terbaik.

# Daftar Referensi
[1]	F. Bellia Annishia, E. Prastiyo, J. Dewi Sartika, and J. Timur, “Pengaruh Harga dan Fasilitas Terhadap Keputusan Menginap Tamu di Hotel Best Western Premier The Hive Jakarta,” Jurnal Hospitality dan Pariwisata, vol. 4, no. 1, 2019.
[2]	A. Winata and A. F. Isnawan, “Pengaruh Harga dan Kualitas Jasa Terhadap Loyalitas Pelanggan Hotel Emersia Di Bandar Lampung,” Jurnal Manajemen Magister, vol. 03, no. 02, 2017.
[3]	R. R. S. Oktaviansyah, “Pengaruh Harga, Promosi Dan Service Excellence Terhadap Kepuasan Pelanggan Java Paragon Hotel and Residence,” Jurnal Ilmu Dan Riset Manajemen, 2020.
