# Evaluasi Internal Test — Nasi Putih (gas-only)

Rotasi 3-fold; data uji tiap fold **dipangkas ekor (trailing-trim) → train:test = 70:30**
(ekor tiap trial dibuang urut waktu, agar uji tetap kontigu temporal & memuat 2 kelas).
Metrik dari prediksi **out-of-fold 3-fold digabung** (lintas-trial), sumber sama dengan
`confusion_matrices_internal.png` & `roc_curves_internal.png` → angka tabel = angka gambar.
Positif = spoiled; threshold 0.5.

## Grouped

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.579 | 0.572 | 0.574 | 0.571 | 0.569 |
| Decision Tree | 0.636 | 0.627 | 0.631 | 0.627 | 0.691 |
| KNN | 0.638 | 0.619 | 0.613 | 0.615 | 0.651 |
| Random Forest | 0.569 | 0.567 | 0.570 | 0.564 | 0.651 |

## Random Split

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.620 | 0.601 | 0.598 | 0.598 | 0.743 |
| Decision Tree | 0.883 | 0.879 | 0.877 | 0.878 | 0.936 |
| KNN | 0.894 | 0.893 | 0.885 | 0.888 | 0.957 |
| Random Forest | 0.899 | 0.897 | 0.892 | 0.894 | 0.963 |
