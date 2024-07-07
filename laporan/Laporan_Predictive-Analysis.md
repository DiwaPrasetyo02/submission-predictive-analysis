
# Laporan Proyek Machine Learning - Diwa Prasetyo


## Domain

Kalangan anak muda terlalu fomo untuk berinvestasi saham, dimana investor muda mengalami kerugian. Kerugian terbesar yang dialami oleh seorang investor pemula ialah pada jenjang usia muda.


Permasalahan FOMO investasi saham di kalangan anak muda merupakan isu penting yang perlu diatasi untuk melindungi investor muda dan mendorong budaya investasi yang sehat dan berkelanjutan. Model machine learning memiliki potensi besar untuk berkontribusi dalam mengatasi permasalahan ini dengan mengembangkan alat edukasi, rekomendasi investasi, deteksi penipuan, dan peningkatan minat baca. Dengan penelitian dan pengembangan yang berkelanjutan, model machine learning dapat menjadi alat yang berharga untuk memberdayakan investor muda dan membangun masa depan keuangan yang lebih cerah bagi mereka. Dengan penggunaan model machine learning yang akan memberikan rekomendasi prediksi harga maka calon investor muda lebih terpersonalisasi.

## Business Understanding
### Problem Statement
Bagaimana cara untuk membantu anak muda yang belum memiliki pengetahuan dan pengalaman investasi saham untuk memilih saham yang tepat, sehingga mereka dapat berinvestasi dengan bijak dan meminimalisir potensi kerugian?
### Goals
Meminimalisir kerugian finansial akibat investasi saham yang tidak tepat.

### Solution
Menyediakan sistem rekomendasi investasi saham yang terpersonalisasi dengan Mengembangkan model machine learning yang dapat menganalisis profil risiko, tujuan investasi, dan toleransi risiko investor muda untuk memberikan rekomendasi saham yang sesuai dengan kebutuhan.
## Data Understanding
Dataset yang digunakan adalah salah satu saham dengan kode [LPKR](https://docs.google.com/spreadsheets/d/1jg5f474SCONnOBmMtTeNw4Mc6RGIBipo/edit?usp=sharing&ouid=105045882344846833587&rtpof=true&sd=true)
* Date   : Waktu dari setiap entry Data
* Open   : Harga buka saham pada waktu perdagangan dalam suatu periode
* High   : Harga tertinggi saham dalam suatu periode
* Low    : Harga terendah saham dalam suatu periode
* Close  : Harga penutupan saham dalam perdagangan saham dalam suatu periode
* Volume : Jumlah saham yang diperdagangkan dalam suatu periode

Adapun Insight yang didapat sebagai berikut:
* Tren harga cenderung menurun
Dari grafik terlihat bahwa harga penutupan saham mengalami tren penurunan yang cukup signifikan sejak tahun 2014 hingga 2024.
Harga saham mencapai puncaknya sekitar tahun 2014-2015, kemudian mengalami penurunan yang tajam dan stabil pada tahun-tahun berikutnya.
![Gambar Grafik](https://github.com/DiwaPrasetyo02/submission-predictive-analysis/blob/main/Screenshot%202024-06-23%20140711.png?raw=true)
* Visualisasi grafik harga tertinggi dan terendah 
Pada sekitar tahun 2020, terlihat penurunan signifikan pada kedua harga tertinggi dan terendah, menunjukkan dampak yang mungkin berasal dari kejadian besar seperti pandemi COVID-19. Dampak ini terlihat di hampir semua sektor dan pasar saham secara global.
![Gambar Grafik](https://github.com/DiwaPrasetyo02/submission-predictive-analysis/blob/main/Screenshot%202024-06-23%20140841.png?raw=true)
* Tren Volume saham cenderung menurun dan terdapat fluktuasi di tahun 2020
Volume perdagangan mencapai puncaknya pada tahun 2022, dengan lonjakan yang jauh lebih tinggi dibandingkan periode lainnya.
Ini mungkin disebabkan oleh peristiwa besar seperti perubahan kebijakan perusahaan, merger dan akuisisi, atau peristiwa global yang mempengaruhi pasar saham. ![Gambar Grafik](https://github.com/DiwaPrasetyo02/submission-predictive-analysis/blob/main/Screenshot%202024-06-23%20140859.png?raw=true)
## Data Preparation
Mengecek missing value
* Missing Value adalah nilai yang hilang dari suatu baris, missing value perlu diatasi untuk mencegah data menjadi bias saat dilatih.
Mengubah format type data Date
* Perubahan ini diperlukan karena default Date di Python adalah Object sehingga harus mengubahnya ke Date Time agar Python bisa memahami data tersebut.

Mengecek outlier dan menangani outlier
* Outlier adalah nilai abnormal yang terdapat pada data. Akan tetapi di Stock atau saham hal ini wajar saja terjadi. Seperti tiba-tiba harga naik drastis atau turun drastis. Tetapi tidak ada salahnya  melihat outlier.

Melakukan normalisasi data yaitu min-max scalling
* Min-Max Scalling dilakukan agar data lebih normal dan berada di ambang 0 dan 1.
## Modeling
Model yang digunakan adalah Long Short Term Memory (LSTM). LSTM dapat digunakan untuk berbagai jenis data time series, baik itu univariat (satu variabel) maupun multivariat (beberapa variabel). LSTM memberikan fleksibilitas dalam aplikasi di berbagai domain. Dalam pelatihan model time series forecasting, masalah vanishing gradient sering terjadi pada RNN tradisional. LSTM mengatasi masalah tersebut dengan mekanisme gating yang memungkinkan gradien tetap stabil selama pelatihan, sehingga dapat menangkap pola jangka panjang dalam data.
* Pada tahap modelling awalnya akan dibuat sebuah fungsi data time series. Dengan scaled_data dan membuat dataset yang cocok untuk model machine learning dengan membaginya menjadi pasangan X dan y, di mana X adalah sekumpulan time steps dan y adalah nilai yang sesuai yang harus diprediksi setelah time steps tersebut. 
* Setelah itu bagi data menjadi data train : data test dengan pembagian 80 : 20. Dan  menambahkan reshape sebagai inputan model yang akan  buat.
* Kemudian bangun model machine learning dengan LSTM yaitu Long Short Term Memory yang telah dijelaskan di awal. 
* Selanjutnya  tambahkan Early Stopping seperti Callback yang digunakan untuk menghentikan pelatihan model jika nilai val_loss tidak membaik setelah sejumlah epochs.
* Terakhir, train model dengan epoch sebanyak 100x, batch size 32 dan gunakan early stopping yang sudah dikonfigurasi sebelumnya.

## Evaluasi

### Mean Absolute Error (MAE)
Pada proyek ini, metrik evaluasi yang digunakan adalah Mean Absolute Error (MAE). MAE adalah salah satu metrik evaluasi yang umum digunakan untuk mengukur performa model regresi. MAE menghitung rata-rata dari nilai absolut selisih antara nilai yang diprediksi oleh model dan nilai aktual.

Penjelasan Mengenai Metrik yang Digunakan
Mean Absolute Error (MAE) memberikan gambaran langsung tentang rata-rata kesalahan prediksi dalam unit yang sama dengan variabel yang diprediksi, tanpa memperhitungkan arah kesalahan (positif atau negatif).

MAE memberikan nilai kesalahan rata-rata yang mudah dipahami oleh semua pemangku kepentingan, termasuk mereka yang mungkin tidak memiliki latar belakang teknis yang kuat. MAE diukur dalam unit yang sama dengan variabel yang diprediksi (harga saham), sehingga memberikan interpretasi langsung terhadap seberapa jauh prediksi menyimpang dari nilai sebenarnya.

Hasil Penerapan Metrik Evaluasi
Dari hasil evaluasi model prediksi harga saham, nilai MAE yang diperoleh adalah 0.0116752153262496. Ini menunjukkan bahwa, rata-rata kesalahan prediksi model  adalah ser 0.0117 dalam satuan harga saham yang digunakan (misalnya, dalam dolar atau rupiah, tergantung pada dataset).

Score atau Nilai yang Didapatkan: Nilai MAE sebesar 0.0116752153262496.

#### Interpretasi Hasil Metrik Sesuai dengan Konteks Data, Problem Statement, Goals, dan Solusi

* Konteks Data: Dataset yang digunakan adalah data harga saham yang mencakup beberapa fitur seperti harga penutupan, harga tertinggi, harga terendah, dan volume perdagangan.
Data ini digunakan untuk memprediksi harga penutupan saham di masa mendatang.

* Problem Statement:
Tujuan proyek ini adalah untuk membangun model yang dapat memprediksi harga penutupan saham dengan akurasi yang tinggi.

* Goals:
Goal dari proyek ini adalah untuk menghasilkan prediksi harga saham yang akurat sehingga dapat digunakan untuk pengambilan keputusan di pasar saham.

* Hasil Evaluasi:
Nilai MAE sebesar 0.0116752153262496 menunjukkan bahwa model memiliki akurasi yang baik dalam memprediksi harga saham di masa mendatang.
Kesalahan rata-rata ini cukup kecil, menunjukkan bahwa prediksi model mendekati nilai aktual dan performa model cukup baik untuk tujuan prediksi harga saham.

* Apakah Proyek Ini Berhasil?:
Berdasarkan nilai MAE yang rendah, proyek ini dapat dianggap berhasil. Model telah mencapai tujuan yang diinginkan, yaitu menghasilkan prediksi harga saham yang akurat.

* Apakah Hasil Evaluasi Sudah Mampu Mencapai Goals yang Diinginkan?:
Ya, hasil evaluasi menunjukkan bahwa model mampu mencapai goals yang diinginkan, yaitu menghasilkan prediksi harga saham yang dapat diandalkan.

* Apakah Sudah Mampu Menyelesaikan Problem yang Diangkat?:
Ya, model telah mampu menyelesaikan problem yang diangkat dengan menyediakan prediksi harga saham yang akurat, yang sangat berguna untuk pengambilan keputusan di pasar saham.
Dengan nilai MAE yang rendah,  dapat menyimpulkan bahwa model prediksi harga saham ini memiliki performa yang baik, memberikan prediksi yang cukup akurat dan dapat diandalkan dalam konteks data dan problem statement yang ada. Model ini dapat digunakan untuk membantu pengambilan keputusan di pasar saham dengan tingkat kepercayaan yang cukup tinggi.

### References

Rahmawati, A., & Surya, T. (2023). Analisis Investasi Saham. *Owner*. 5(2), 123-134. https://owner.polgan.ac.id/index.php/owner/article/download/1619/949/8692

Santoso, D. (2022, March 28). Geliat Kaum Muda Berinvestasi. *Kompas.id*. https://www.kompas.id/baca/telaah/2022/03/28/geliat-kaum-muda-berinvestasi

CNBC. (2023, November 13). Access ASEAN: Indonesia. *CNBC*. https://www.cnbc.com/video/2023/11/13/access-asean--indonesia.html






