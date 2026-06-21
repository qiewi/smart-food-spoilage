# PRD — Analisis Performa Model Klasifikasi Kesegaran Makanan

**Tugas Akhir — Smart Canteen | Klasifikasi Fresh vs Spoiled berbasis Sensor Gas MQ**

Dokumen ini adalah rencana kerja (Product Requirements Document) + langkah teknis
sebelum implementasi machine learning. Tujuannya memastikan setiap keputusan
metodologis sudah jelas, dapat dipertanggungjawabkan di sidang, dan bebas dari
kesalahan umum seperti data leakage.

---

## 1. Tujuan & Ruang Lingkup

**Tujuan:** Membandingkan empat model klasifikasi klasik untuk menentukan model
terbaik dalam mengklasifikasikan makanan sebagai `fresh` atau `spoiled` berdasarkan
pembacaan sensor.

**Model yang dibandingkan:**

1. Logistic Regression (bukan Linear Regression — ini tugas klasifikasi)
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Random Forest

**Catatan konsep penting:** Keempat model ini *tidak* dilatih per-epoch. Epoch
adalah konsep neural network. Padanan "tuning latihan" di sini adalah
**hyperparameter tuning** menggunakan `GridSearchCV` + **cross-validation**.

**Dataset:** Hasil eksperimen sendiri (file `*_relabeled.csv`), kolom:
`elapsed, mq2, mq135, mq4, humidity, tempC, food_type, label`.

---

## 2. Definisi Sukses (Success Criteria)

Sesuai NFR di proposal:

| Kriteria | Target | Sumber |
|---|---|---|
| Akurasi model terbaik | > 85% | NFR01 |
| Prioritas metrik | **Recall kelas `spoiled`** setinggi mungkin | NFR01 (kurangi false negative) |
| Waktu inferensi | < 2 detik / sampel | NFR02 |

> **Kenapa recall `spoiled` jadi prioritas, bukan akurasi?**
> False negative = makanan **busuk diklasifikasi fresh** = makanan tidak aman
> tersaji ke konsumen. Ini risiko keamanan pangan paling berbahaya, jauh lebih
> serius daripada false positive (makanan fresh dibuang). Maka model dinilai
> terutama dari kemampuannya menangkap semua sampel spoiled.

---

## 3. Keputusan Desain (WAJIB diputuskan sebelum coding)

Tiga keputusan ini mengubah seluruh pipeline. Tetapkan dulu, catat alasannya di bab metodologi.

### 3.1 Strategi Split Data — paling kritis

Ini sumber kesalahan #1 pada penelitian sensor time-series.

**Masalah:** Data sensor selama satu proses pembusukan sangat berkorelasi antar
waktu (pembacaan menit ke-10 mirip menit ke-11). Kalau di-split **acak** 80/20,
sampel test akan "bocor" karena tetangga waktunya ada di train. Akibatnya akurasi
terlihat 99%+ padahal model sebenarnya cuma menghafal, **tidak** belajar pola yang
bisa digeneralisasi. Inilah kenapa angka 99% di literatur (Stephan dkk.) harus
disikapi hati-hati.

**KEPUTUSAN: pakai `StratifiedGroupKFold` (group split berbasis run).**

Karena setiap makanan diambil datanya **minimal 3x run/trial terpisah**, kita punya
kondisi ideal untuk strategi paling kuat: split berbasis grup.

- **1 run/trial = 1 grup.** Penanda grup disimpan di kolom `run_id`.
- **Aturan utama:** satu run utuh hanya boleh berada di train ATAU di test, tidak
  boleh terpecah ke keduanya.
- Tiap fold melatih di sebagian run dan menguji di run yang **belum pernah dilihat**.
- `Stratified` menjaga proporsi `fresh : spoiled` tetap seimbang di tiap fold.

**Kenapa ini yang dipilih:**

| | Random split | `StratifiedGroupKFold` (run-based) |
|---|---|---|
| Pembacaan menit berdekatan dari run sama | Bisa terpisah train/test → **bocor** | Selalu bersama → aman |
| Akurasi yang dilaporkan | Over-optimistis (menghafal) | Jujur, mencerminkan generalisasi |
| Pertanyaan yang dijawab | "Bisa tebak titik tengah run yang sama?" | "Bisa generalisasi ke run baru?" ← relevan |

Skema ini meniru kondisi nyata: model dilatih pada eksperimen lama, lalu dipakai
pada makanan baru yang baru digoreng/dimasak hari berikutnya.

**Konsekuensi yang harus diterima & dilaporkan:**
- Dengan ~3 run per makanan, jumlah grup masih sedikit → estimasi performa punya
  **variansi cukup besar** (skor antar fold bisa naik-turun). Ini wajar.
- **Wajib laporkan rata-rata ± standar deviasi antar fold**, bukan satu angka tunggal.
- Sebut jumlah run sebagai keterbatasan di bab evaluasi. Menambah run (mis. ke 5)
  akan menstabilkan estimasi — sebutkan sebagai saran pengembangan.
- Angka akurasi mungkin terlihat lebih rendah dari random split — itu **bukan
  kegagalan**, itu angka yang jujur dan justru lebih dihargai penguji.

### 3.2 Cakupan Fitur — humidity & tempC ikut atau tidak?

Ada ketidaksesuaian yang perlu kamu putuskan: Batasan Masalah proposal (poin 2)
menyatakan **tanpa** suhu/kelembapan, tapi hardware (DHT22) sebenarnya merekam
`humidity` dan `tempC`.

- **Opsi 1 — Murni gas (sesuai proposal):** fitur = `mq2, mq135, mq4`. Konsisten
  dengan batasan masalah tertulis.
- **Opsi 2 — Gas + lingkungan:** fitur = `mq2, mq135, mq4, humidity, tempC`.
  Literatur (Stephan dkk.) menunjukkan suhu/kelembapan menambah akurasi.

**Rekomendasi:** Jalankan **keduanya** sebagai ablation study (bandingkan performa
dengan vs tanpa fitur lingkungan). Ini justru memperkaya analisis tugas akhir.
Kalau harus pilih satu untuk konsistensi proposal, pakai Opsi 1 dan revisi batasan
masalah bila ingin memasukkan lingkungan.

### 3.3 Satu model gabungan atau per-makanan?

- **Gabungan + `food_type` sebagai fitur (one-hot):** satu model untuk semua
  makanan. Lebih praktis untuk deployment di kantin (1 model). Direkomendasikan
  untuk dataset kecil.
- **Per-makanan (4 model terpisah):** akurasi bisa lebih tinggi per komoditas tapi
  data tiap model jadi makin sedikit dan lebih rawan overfit.

**Rekomendasi:** Mulai dari model gabungan dengan `food_type` di-one-hot encode.

---

## 4. Pipeline — Langkah demi Langkah (CRISP-DM)

### FASE 1 — Data Understanding

1. **Gabungkan semua CSV** hasil relabel menjadi satu DataFrame. Saat menggabung,
   **tambahkan kolom `run_id`** dari nama file tiap CSV (1 file = 1 trial = 1 run).
   Kolom ini WAJIB ada — dipakai sebagai `groups` saat cross-validation, dan
   **bukan** sebagai fitur model.
2. **Inspeksi awal:** `df.shape`, `df.info()`, `df.describe()`, cek `dtypes`.
3. **Cek missing values:** DHT22 kadang mengembalikan NaN → kolom `humidity`/`tempC`
   bisa kosong. Hitung jumlahnya.
4. **Cek keseimbangan kelas** per makanan: berapa rasio fresh : spoiled? Hampir
   pasti tidak seimbang (durasi spoiled biasanya lebih panjang). Catat ini.
5. **Visualisasi trajektori sensor** terhadap waktu (`elapsed`) per makanan —
   untuk melihat apakah ada titik di mana gas naik tajam (konfirmasi label masuk akal).

### FASE 2 — Data Preparation

6. **DROP kolom `elapsed` dan `run_id` dari fitur.** Keduanya bukan fitur:
   `elapsed` adalah metadata waktu (kalau dipakai → model menebak label dari waktu
   saja, leakage telak), dan `run_id` hanya penanda grup untuk CV (kalau dipakai →
   model menghafal identitas run). `run_id` tetap **disimpan terpisah** untuk
   dipakai sebagai argumen `groups=` saat cross-validation.
7. **Tangani missing values** pada `humidity`/`tempC` (jika Opsi 2 dipakai):
   imputasi median, atau drop baris jika jumlahnya sedikit. Putuskan & catat.
8. **Encode `food_type`** dengan one-hot (`pd.get_dummies` atau `OneHotEncoder`)
   bila pakai model gabungan.
9. **Encode label** target: `fresh` → 0, `spoiled` → 1 (positive class = spoiled,
   karena itu yang ingin kita deteksi).
10. **Tangani outlier / warm-up:** buang data beberapa menit pertama tiap run kalau
    sensor masih warm-up dan nilainya tidak stabil.
11. **Feature scaling** — penting untuk **KNN** dan **Logistic Regression**
    (berbasis jarak/gradien). Decision Tree & Random Forest tidak butuh, tapi tidak
    masalah jika diskalakan. **WAJIB** scaling dilakukan **di dalam pipeline / di
    dalam tiap fold CV**, bukan di seluruh data sekaligus, agar tidak bocor
    statistik test ke train. Gunakan `StandardScaler`.
12. **Split data dengan `StratifiedGroupKFold`** (lihat 3.1), pakai `run_id` sebagai
    `groups`. Tidak ada train_test_split acak — evaluasi sepenuhnya lewat cross
    validation berbasis grup agar tiap run diuji sebagai data yang belum dilihat.

> **Catatan (opsi pengembangan, TIDAK dipakai sekarang):** kalau ternyata variasi
> baseline antar-run bikin performa tidak stabil, teknik *baseline correction*
> (mis. rasio `R/R0`) bisa ditambahkan di tahap ini sebagai eksperimen lanjutan.
> Untuk versi awal ini kita pakai nilai sensor apa adanya dulu.

### FASE 3 — Modelling

13. **Bangun `Pipeline`** untuk tiap model: `StandardScaler` → classifier.
    Scaler di dalam pipeline = aman dari leakage saat cross-validation.
14. **Hyperparameter tuning** dengan `GridSearchCV` (atau `RandomizedSearchCV`),
    scoring difokuskan ke metrik prioritas (`recall` atau `f1` untuk kelas spoiled).
    Hyperparameter kunci per model:

    | Model | Hyperparameter utama |
    |---|---|
    | Logistic Regression | `C` (regularisasi), `penalty`, `class_weight='balanced'` |
    | Decision Tree | `max_depth`, `min_samples_leaf`, `class_weight='balanced'` |
    | KNN | `n_neighbors`, `weights` (uniform/distance), `metric` |
    | Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight` |

    > `class_weight='balanced'` membantu mengatasi kelas yang tidak seimbang —
    > relevan karena `spoiled` mungkin lebih sedikit/banyak dari `fresh`.

15. **Cross-validation dengan `StratifiedGroupKFold`** (`n_splits=3`, sejumlah run;
    naikkan kalau run bertambah), `groups=run_id` — menggantikan konsep "epoch".
    `GridSearchCV` menerima `cv=StratifiedGroupKFold(...)` dan `groups` diteruskan
    lewat `.fit(X, y, groups=run_id)`. Catat skor **rata-rata ± standar deviasi**
    antar fold — bukan satu angka tunggal.

### FASE 4 — Evaluation

16. **Evaluasi via cross-validation berbasis grup.** Karena tiap run jadi test
    secara bergiliran, kumpulkan prediksi out-of-fold (`cross_val_predict` dengan
    `StratifiedGroupKFold`) lalu hitung metrik di seluruh prediksi tersebut:
    Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC. Setiap sampel
    diprediksi saat run-nya berada di fold test → tidak ada leakage.
17. **Fokus utama:** Recall & F1 untuk kelas `spoiled`. Tampilkan
    `classification_report` per kelas, bukan cuma akurasi global.
18. **Tabel perbandingan** keempat model berdampingan → ini inti hasil tugas akhir.
19. **Confusion matrix** tiap model — soroti jumlah false negative (spoiled→fresh).
20. **Pilih model terbaik** berdasarkan kombinasi: recall spoiled tertinggi, F1
    kompetitif, dan kestabilan CV (std kecil). Bukan semata akurasi tertinggi.
21. **Feature importance** (untuk Decision Tree & Random Forest) → sensor mana
    paling berkontribusi? Ini analisis bagus untuk pembahasan.

---

## 5. Struktur Kode yang Disarankan

```
project/
├── data/
│   └── relabeled/              # CSV hasil relabel
├── notebooks/
│   └── analisis_model.ipynb    # eksplorasi + eksperimen
├── src/
│   ├── load_data.py            # gabung & bersihkan CSV
│   ├── preprocess.py           # encoding, scaling, split
│   ├── train.py                # pipeline + GridSearchCV 4 model
│   └── evaluate.py             # metrik + confusion matrix + plot
└── results/
    ├── comparison_table.csv
    └── figures/
```

---

## 6. Checklist Anti-Kesalahan (cek sebelum lapor hasil)

- [ ] Kolom `elapsed` **dan** `run_id` sudah di-drop dari fitur?
- [ ] `run_id` disimpan terpisah & dipakai sebagai `groups` di CV?
- [ ] Scaling dilakukan di dalam pipeline/CV fold, bukan di seluruh data?
- [ ] CV pakai `StratifiedGroupKFold` (bukan KFold/random split)?
- [ ] Metrik yang dilaporkan menyertakan recall/F1 per kelas, bukan cuma akurasi?
- [ ] `class_weight='balanced'` dipertimbangkan untuk kelas tak seimbang?
- [ ] Hasil CV dilaporkan dengan rata-rata ± std antar fold (bukan satu angka)?
- [ ] Jumlah run disebut eksplisit sebagai keterbatasan?

---

## 7. Output yang Diharapkan

1. Tabel perbandingan 4 model (Accuracy, Precision, Recall, F1, AUC).
2. Confusion matrix tiap model.
3. Grafik feature importance (tree-based).
4. Rekomendasi model terbaik + justifikasi berbasis recall spoiled.
5. (Opsional) Ablation study: dengan vs tanpa fitur lingkungan.
