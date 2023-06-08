# Laporan Proyek Machine Learning - Jelvin Krisna Putra 

## Domain Proyek

_Scope_ yang diambil merupakan _scope_ **Ekonomi dan bisnis**. Seperti yang kita ketahui hotel sekarang sudah menjadi peluang bisnis yang menjanjikan, dan kebanyakan pengusaha hotel sudah menerapkan sistem informasi ke dalam layanan mereka, contohnya aplikasi untuk pemesanan hotel hingga _check-in_ bisa melalui gadget pintar di tangan konsumen.

Banyak yang menjadi pertimbangan konsumen dalam memilih merk hingga tempat hotel, namun dalam dunia bisnis, pertimbangan terbesar jatuh ke pada harga sewa dari suatu hotel. menurut Tjiptono (seperti yang dikutip dari [[1]](https://core.ac.uk/download/pdf/268049173.pdf)) dikatakan bahwa ada segmen pembeli yang sangat sensitif terhadap faktor harga (menjadikan harga sebagai satu-satunya pertimbangan membeli produk) dan ada pula yang tidak.

"salah satu tujuan perusahaan itu adalah [sic!]menciptakaan loyalitas pelanggan maka suatu perusahaan yang baik, misalnya penetapan harga yang tepat sesuai dengan pangsa pasar dan kondisi perekonomian masyarakat sekitar dapat menciptakan kepuasan konsumen, konsumen yang merasa puas dapat terciptanya loyalitas pelanggan." [[2]](https://jurnal.darmajaya.ac.id/index.php/jmmd/article/view/989)

Pentingnya harga sewa dalam loyalitas konsumen dan kepuasan pelanggan terkadang membuat banyak perusahaan rela melakukan berbagai macam cara untuk bisa mendapatkan harga sewa yang paling optimal tanpa harus mengorbankan pendapatan perusahaan, "Kepuasan pelanggan merupakan salah satu faktor penting yang harus 
diperhatikan perusahaan karena pelanggan merupakan alasan mengapa suatu perusahaan 
eksis." [[3]](https://www.journal.stieamkop.ac.id/index.php/yume/article/view/3036) karena dengan harga sewa yang tepat akan ada banyak konsumen yang menginap di hotel _Airbnb_ yang akan meningkatkan pendapatan dari _Airbnb_ itu sendiri

Oleh karena itu dalam project kali ini **bayangkan** saya sebagai _data science_ diminta oleh perusahaan _Airbnb_ untuk membuat _model machine learning_ yang mampu memprediksi harga sewa yang cocok terhadap suatu hotel airbnb berdasarkan data-data yang ada seperti lokasi, nama, dan data lainnya.

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

Pada tahap ini, kita akan menggunakan empat algoritma berbeda, yaitu _Random Forest_, _Boosting Algorithm_, _Multilayer Perceptron Regressor (MLPRegressor)_, dan _Support Vector Regression (SVR)_ dimana untuk menghindari _overfitting_ tiap algoritma akan dicari parameter terbaiknya dengan teknik _hyperparameter tuning_ (_random search cross validation_ dan _grid search cross validation_)
*Random search cross validation* adalah metode untuk mencari kombinasi parameter optimal dalam model pembelajaran mesin dengan menggunakan teknik cross-validation. Metode ini menguji subset acak dari ruang parameter yang diberikan, bukan mencoba setiap kombinasi parameter secara sistematis.
Sedangkan untuk _grid search cross validatio_ sama-sama menggunakan teknik cross-validation, namun metode ini mencoba setiap kombinasi parameter yang mungkin dalam ruang parameter yang ditentukan sebelumnya, tidak acak namun sistematis.
![Cross Validation Image](https://scikit-learn.org/stable/_images/grid_search_cross_validation.png)
Gambar 8. Cross Validation
(sumber: https://scikit-learn.org/stable/modules/cross_validation.html)
Bisa dilihat dari gambar 8 cross validation bekerja dengan membagi-bagi lagi dataset pelatihan kita menjadi beberapa bagian lagi untuk mencari nilai parameter yang terbaik, dan karena pembagian dataset inilah yang menjadi alasan mengapa kita tidak menggunakan model hasil *hyperparameter tuning* dalam model akhir kita karena kita dianjurkan melakukan pelatihan lagi ke model dengan semua *dataset training*.

```models = pd.DataFrame(index=['train_mse', 'test_mse'], columns=['RandomForest', 'Boosting', 'MLPRegressor', 'SVR'])```
kode di atas dijalankan sebelum kita masuk ke setiap algoritma dimana kita buat _dataframe_ yang akan menampung semua error model agar kita dapat membandingkan errornya satu sama lain
1. _Random Forest_:
   - _Random Forest_ Merupakan Algoritma _ensemble_ yang terdiri dari banyak algoritma yang bergabung menjadi satu sehingga bisa menghasilkan model yang baik, _random forest_ terdiri dari banyak _decision tree_ sehingga membentuk _(forest)_ dimana algoritma ini dibagi menjadi 2 teknik yaitu teknik _bagging_ dan _boosting_ dimana perbedaanya dalah _bagging_ itu tiap model akan melatih sampel yang berbeda dan setiap sampel tidak mempengaruhi hasil dari sampel lainnya sedangkan untuk *boosting* model akan melatih sampel secara iteratif sehingga setiap sampel akan mempengaruhi kualitas model antar satu sama lain, alasan digunakannya algoritma ini adalah karena algoritma ini bisa menyelesaikan masalah regresi dan dari penyebaran data-nya yang tinggi maka algoritma ini yang dipilih dalam kasus kali ini.
   - Adapun kelebihan dari algoritma ini antara lain mampu mengatasi data yang tidak seimbang dan cocok untuk kasus regresi dan klasifikasi inilah yang sekaligus menjadi alasan kenapa algoritma ini masuk menjadi pertimbangan untuk dilatih menjadi model
   - Kekurangan dari algoritma ini adalah lebih kompleks dan membutuhkan lebih banyak sumber daya komputasi karena terdiri dari banyak _decision 
   - Selanjutnya kita akan menentukan parameter yang digunakan dalam algoritma _random forest_ pencarian menggunakan _RandomizedSearchCV_ dan dilanjutkan lebih spesifik dengan _GridSearchCV_ yang diambil dari _sklearn.model_selection_, parameter tebakan diambil secara random dengan _library numpy_ yang dibuat alias menjadi _np_ dimana _function arange_ itu untuk mengenerate angka random sesuai dengan parameter yang diberi contohnya pada baris ```'n_estimators': np.arange(25, 200, 25),``` angka yang akan dihasilkan adalah list ```[ 25  50  75 100 125 150 175]``` dan parameter lain dilakukan hal yang serupa, setelah ditentukan untuk setiap tebakan kita membuat _object_ dari _random search_ dengan ```random_search_rf = RandomizedSearchCV()``` dengan paramater ```estimator=rf``` _rf_ merupakan object dari ```RandomForestRegressor()```, ```param_distributions=param_dist,``` _param distribution_ merupakan kamus _(dictionary)_ yang berisi daftar _parameter_ dan daerah nilai yang akan dieksplorasi selama pencarian parameter, ```n_iter=20``` dibuat menjadi 20, yang berarti akan diuji 20 kombinasi acak dari parameter yang diberikan,akan menghasilkan _100 fitting_, _random state_ dibuat 123 agar setiap pelatihan model bisa menghasilkan model yang serupa  dan parameter _n_jobs_ ini mengontrol jumlah pekerjaan paralel yang akan dieksekusi saat pencarian parameter. Dalam hal ini, ```n_jobs=-1``` berarti menggunakan semua inti (cores) yang tersedia pada komputer untuk menjalankan pekerjaan secara paralel, dan _verbose=2_ untuk menentukan level debug agar proses pelatihan bisa diatmpilkan di output, lalu untuk menjalankan _random search cv_ dengan memanggil fungsi ```random_search_rf.fit(X_train, y_train)```
   - Kemudian hasil dari parameter terbaik menurut search _random_search_rf_ akan digunakan untuk melatih model dengan algoritma yang sama namun parameter dicari dengan _Grid Search Cross Validation_ dengan cara membuat variabel _param_grid_ yang diinisialisasi dengan _random_search_rf_ kemudian membuat ```grid_search_rf = GridSearchCV()``` dengan parameter yang relatif sama dengan _random search cross validation_
   - Setelah mendapatkan parameter yang terbaik kita kemudian menggunakan parameter terbaik tadi untuk digunakan dalam algoritma dengan menggunakan _**kwargs_ seperti kode di samping ```RF = RandomForestRegressor(**best_params_grid)``` dan memsnggil fungsi fit untuk melakukan pelatihan ke algoritma, kita melatih ulang model dengan seluruh dataset *training* yang sudah di-split, itu karena _Cross Validation_ hanya mencari parameter terbaik dengan membagi dataset menjadi beberapa potongan sehingga kita tidak dianjurkan menggunakan model yang dihasilkan langsung oleh algoritma *random search* dan *grid search* seperti yang sudah dijelaskan di poin *modelling* paragraf awal. Menghitung ```mean_squared_error``` dengan kode ```models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)``` dimana besaran _MSE_ dimasukkan ke dalam satu list yang sama bernama _models_ sehingga bisa dibandingkan dengan 3 algoritma lainnya.
2. Boosting Algorithm (AdaBoost):
   - Boosting adalah metode _ensemble learning_ yang menggabungkan beberapa model "lemah" _(weak learners)_ secara sekuensial untuk memperkuat model secara. Algoritma boosting terdiri dari beberapa jenis dan yang digunakan adalah Adaboost, untuk perbedaan *boosting* dan *bagging* bisa dibaca lagi di *random forest algorithm*.
   - Adapun kelebihan dari algoritma ini adalah dapat menangani berbagai tipe data dan masalah seperti regresi dan klasifikasi dan mampu menangani data yang tidak seimbang sekaligus menjadi alasan mengapa algoritma ini dipilih untuk menjadi algoritma yang akan kita gunakan sebagai komparasi algoritma lain.
   - Kekurangan dari algoritma ini adalah Rentan terhadap overfitting jika salah dalam menentukan paramater yang tepat sehingga cukup kompleks.
   - teknik pencarian parameter untuk algoritma ini akan menggunakan teknik yang sama dengan algoritma random forest yaitu *random search cross validation* dan *grid search cross validation*, dengan penyesuaian parameter disesuaikan dengan algoritma boosting, dimana parameter untuk algoritma ini ada 3 mulai dari parameter n_estimators diisi dengan angka generate dari 30 hingga 80 dengan penambahan setiap angka ditambah 10, lalu learning_rate diisi nilai desimal dengan range 0.01 hingga 1.0 dengan 10 angka totalnya dan loss yang diisi dengan *linear*, *square*, dan *exponential* , setelah didapatkan parameter terbaiknya parameter tersebut kemudian digunakan untuk dicari kembali menggunakan *grid search cross validation*  dengan cara yang serupa dengan algoritma pertama *random forest*.
   - Pelatihan model akhir akan dilakukan dengan parameter terbaik dari *grid search cross validation* dengan _**kwargs_, selanjutnya model akhir akan dihitung mean square errornya dengan cara memanggil fungsi ```mean_squared_error(y_pred=boosting.predict(X_train)```
3. Multi Layer Perceptron Regressor (MLPRegressor)
   - Model neural network dengan lapisan tersembunyi yang digunakan untuk masalah regresi, secara garis besar terdiri atas *layer input, hidden layer, dan output layer*, beberapa parameter penting dalam MLPRegressor adalah *hidden_layer_sizes* yang mengatur jumlah lapisan dan jumlah unit di setiap lapisan tersembunyi, *activation* fungsi aktivasi yang digunakan di setiap unit tersembunyi, *learning rate* tingkat pembelajaran yang mengontrol seberapa cepat model belajar dari data, dan *solver* optimizer yang digunakan untuk menemukan bobot terbaik dalam jaringan.
   - Kelebihan dari algoritma ini adalah mampu menangani masalah regresi yang kompleks dengan baik, fleksibel dan dapat membuat model hubungan non-linear, dapat mengatasi data yang berukuran besar, dan sekaligus menjadi alasan mengapa dipilih menjadi algoritma yang akan dijadikan komparasi dengan algoritma lain.
   - kekurangan dari algoritma ini adalah waktu komputasi yang lebih lama, pemilihan parameter yang harus tepat dan sulit untuk memberikan interpretasi yang langsung pada hasil model karena menggunakan layer tersembunyi.
   - param yang diisi ada hidden layer sizes yang diisi dengan list of tuple ```[(25,), (50,), (100,), (150,)]```, activation diisi _relu_ solver diisi _adam, lbfgs_, alpha 0.0001, 0.001, 0.01, dan *learning rate* diisi *contant* dan *adaptive* yang kemudian dimasukkan ke dalam fungsi *random search cross validation* yang parameter terbaiknya dicari lagi lebih spesifik oleh *grid search cross validation* sama seperti algoritma-algoritma sebelumnya, dan akhirnya menghasilkan parameter terbaik.
   - Pelatihan model akhir akan dilakukan dengan parameter terbaik dari *grid search cross validation* dengan _**kwargs_, selanjutnya model akhir akan dihitung mean square errornya dengan cara memanggil fungsi ```mean_squared_error(y_pred=boosting.predict(X_train)```
4. Support Vector Regressor (SVR)
   - versi regresi dari metode Support Vector Machine (SVM). Parameter-parameter yang umum digunakan dalam SVR adalah:
C: Parameter yang mengontrol trade-off antara penalti kesalahan dan margin maksimal.
kernel: Jenis kernel yang digunakan dalam fungsi pemetaan non-linear.
epsilon: Membentuk margin toleransi untuk kesalahan prediksi.
   - Mampu menangani masalah regresi dengan baik, termasuk data yang memiliki banyak fitur.
Efektif dalam mengatasi masalah dengan dimensi tinggi.
Memiliki toleransi terhadap outlier.
   - Memerlukan tuning parameter yang cermat untuk kinerja yang optimal.
Sensitif terhadap skala data.
Tidak memberikan interpretasi yang langsung terhadap hubungan antara fitur dan target.
   - Parameter yang dipilih ada *kernel* dengan isi *linear, rbf, *dan* poly,* selanjutnya *C* diisi *0.1, 0.5, 1, 2, 3, 4,* ada *gamma* dengan *scale* atau *auto* dan *epsilon* yang diisi angka desimal 0.01 hingga 1 dengan total angka ada 10, yang kemudian dimasukkan ke dalam fungsi *random search cross validation* yang parameter terbaiknya dicari lagi lebih spesifik oleh *grid search cross validation* sama seperti algoritma-algoritma sebelumnya, dan akhirnya menghasilkan parameter terbaik.
   - Pelatihan model akhir akan dilakukan dengan parameter terbaik dari *grid search cross validation* dengan _**kwargs_, selanjutnya model akhir akan dihitung mean square errornya dengan cara memanggil fungsi ```mean_squared_error(y_pred=boosting.predict(X_train)```

Untuk model terbaik akan dilihat dari model yang memiliki *MSE* terkecil.

## Evaluation

Metrik evaluasi yang akan digunakan untuk menganalisis kinerja model dalam memprediksi harga sewa Airbnb di New York City adalah Mean Squared Error (MSE). Metrik ini sesuai untuk masalah regresi.

MSE = $\frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2$
Dari rumus di atas, *Mean Square Error* dihitung dari rata-rata kuadrat selisih antara nilai prediksi dan nilai sebenarnya. Sederhananya semakin rendah MSE, semakin baik performa model.

Tabel 2. Perbandingan *MSE* setiap algoritma
|                |   train    |   test     |
|----------------|------------|------------|
| RF             | 1.406154   | 2.130259   |
| Boosting       | 2.377423   | 2.426624   |
| MLPRegressor   | 2.037612   | 2.158412   |
| SVR            | 2.428922   | 2.508062   |

Dapat dilihat pada tabel 2 error terendah untuk data train dimiliki oleh algoritma *random forest* dan tertinggi ada pada algoritma *SVR* dan untuk data test algoritma *random forest* memiliki error terendah dan tertinggi ada pada algoritma *SVR*, untuk kasus ini dan data cleaning yang penulis lakukan serta cara memilih parameter bisa dilihat algoritma random forest yang terbaik namun tidak menutup kemungkinan bahwa di tangan *data science* lain dengan dataset yang sama algoritma lain menjadi pemenangnya.


![Grafik MSE](https://raw.githubusercontent.com/krisna31/air-bnb-rent-price-analytics/main/images/mse%20graph.png)
Gambar 9. Grafik Batang *MSE* setiap algoritma

Tabel 3. Perbandingan Nilai sebenarnya dan prediksi tiap model
| y_true | prediksi_RF | prediksi_Boosting | prediksi_MLPRegressor | prediksi_SVR |
|--------|-------------|-------------------|-----------------------|--------------|
| 200 | 192.8   | 194.2          | 192.4        | 158.4 |

tabel 3 menunjukkan prediksi dari algoritma random forest, boosting, dan MLPRegressor yang bisa mendekati nilai sebenarnya untuk contoh satu data.
Sehingga dari tabel 2, tabel 3, dan gambar 9 bisa disimpulkan bahwa model yang kita buat dengan batas toleransi MSE 3% di kedua dataset berhasil dibuat dengan algoritma terbaik ada di *Random Forest Algorithm* untuk proyek kali ini.

# Daftar Referensi
[1]	F. Bellia Annishia, E. Prastiyo, J. Dewi Sartika, and J. Timur, “Pengaruh Harga dan Fasilitas Terhadap Keputusan Menginap Tamu di Hotel Best Western Premier The Hive Jakarta,” Jurnal Hospitality dan Pariwisata, vol. 4, no. 1, 2019.
[2]	A. Winata and A. F. Isnawan, “Pengaruh Harga dan Kualitas Jasa Terhadap Loyalitas Pelanggan Hotel Emersia Di Bandar Lampung,” Jurnal Manajemen Magister, vol. 03, no. 02, 2017.
[3]	R. R. S. Oktaviansyah, “Pengaruh Harga, Promosi Dan Service Excellence Terhadap Kepuasan Pelanggan Java Paragon Hotel and Residence,” Jurnal Ilmu Dan Riset Manajemen, 2020.
