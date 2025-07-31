# Dokumen Persyaratan Produk MVP - AI Bias Detector

## 1. Pendahuluan
    *   **Tujuan:** Dokumen ini merinci persyaratan untuk Produk Minimum yang Layak (MVP) dari AI Bias Detector, sebuah pustaka Python yang dirancang untuk mendeteksi dan membantu memitigasi bias dalam model dan dataset machine learning.
    *   **Pernyataan Masalah:** Seiring semakin meluasnya penggunaan sistem AI, memastikan keadilan dan kesetaraan menjadi sangat penting. Bias tersembunyi dalam data dan model dapat menyebabkan hasil yang diskriminatif, yang bertujuan untuk dideteksi dan dibantu oleh pustaka ini.
    *   **Visi:** Menjadikan AI Bias Detector sebagai pustaka Python dasar yang mirip Scikit-learn untuk ilmuwan data dan insinyur ML yang berfokus pada pembuatan dan penerapan sistem AI yang adil.

## 2. Tujuan
    *   **Tujuan Utama:** Mengembangkan pustaka Python yang mampu menganalisis dataset dan model machine learning yang terlatih untuk mendeteksi bias tersembunyi spesifik terkait gender, ras, dan usia.
    *   **Tujuan Sekunder:** Memberikan rekomendasi yang jelas dan dapat ditindaklanjuti untuk menyeimbangkan bias yang teridentifikasi.
    *   **Tujuan Teknis:** Membuat pustaka dengan API yang familiar dan mudah diintegrasikan bagi pengguna yang terbiasa dengan Scikit-learn.

## 3. Target Audiens
    *   Ilmuwan Data (Data Scientists)
    *   Insinyur Machine Learning (Machine Learning Engineers)
    *   Peneliti AI (AI Researchers)
    *   Pengembang yang bekerja dengan model machine learning

## 4. Fitur Utama (MVP)
    *   **4.1. Analisis Bias Dataset:**
        *   **Input:** Menerima format dataset umum, terutama Pandas DataFrames.
        *   **Fungsionalitas:**
            *   Mengidentifikasi dan mengukur representasi atribut sensitif (gender, ras, usia) dalam dataset.
            *   Menghitung ukuran statistik untuk mendeteksi potensi bias dalam distribusi data di berbagai atribut ini.
        *   **Output:** Laporan yang merinci metrik bias dataset dan distribusi atribut.
    *   **4.2. Analisis Bias Model:**
        *   **Input:** Menerima model machine learning yang terlatih, dengan fokus awal pada model yang kompatibel dengan API Scikit-learn.
        *   **Fungsionalitas:**
            *   Mengevaluasi metrik kinerja model (misalnya, akurasi, presisi, recall) di berbagai kelompok demografis yang ditentukan oleh atribut sensitif.
            *   Mendeteksi disparitas dalam prediksi model atau tingkat kesalahan yang berkorelasi dengan atribut tersebut.
        *   **Output:** Laporan yang menyoroti disparitas kinerja model dan metrik bias.
    *   **4.3. Rekomendasi Mitigasi Bias:**
        *   **Fungsionalitas:** Berdasarkan hasil analisis, menyarankan strategi yang sesuai untuk mitigasi bias. Ini termasuk:
            *   Teknik penyesuaian ulang data (misalnya, oversampling kelompok minoritas, undersampling kelompok mayoritas).
            *   Teknik pembobotan ulang data (re-weighting).
            *   Panduan penerapan teknik keadilan algoritmik (misalnya, metode pra-pemrosesan, dalam-pemrosesan, pasca-pemrosesan).
        *   **Output:** Rekomendasi yang dapat ditindaklanjuti dengan panduan implementasinya.

## 5. Non-Tujuan (untuk MVP)
    *   **Koreksi Bias Otomatis:** MVP akan berfokus pada deteksi dan rekomendasi, bukan pada koreksi bias secara otomatis.
    *   **Jenis Bias Komprehensif:** MVP akan secara spesifik menangani bias gender, ras, dan usia. Jenis bias lain (misalnya, status sosial ekonomi, afiliasi politik) berada di luar cakupan untuk rilis awal ini.
    *   **Interpretasi Model Tingkat Lanjut:** Analisis mendalam tentang akar penyebab bias dalam algoritma model bukanlah fokus utama untuk MVP.
    *   **Antarmuka Pengguna Grafis (GUI):** Pustaka akan dirancang untuk penggunaan terprogram melalui kode Python, bukan melalui antarmuka visual.

## 6. Pertimbangan Masa Depan
    *   Perluasan untuk mencakup berbagai jenis bias.
    *   Integrasi dengan kerangka kerja ML populer lainnya seperti TensorFlow dan PyTorch.
    *   Pengembangan modul koreksi bias otomatis.
    *   Penyertaan alat visualisasi untuk analisis bias dan kemajuan mitigasi.
    *   Dukungan untuk pemantauan bias secara real-time pada model yang diterapkan.
