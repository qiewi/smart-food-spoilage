# Evaluasi Validation Test 50/50 — Nasi Putih (gas-only)

Metrik dari probabilitas **3-fold dirata-rata per baris** (soft-ensemble),
sumber sama dengan `confusion_matrices_validation.png` & `roc_curves_validation.png`
→ angka tabel = angka gambar. Validasi = trial 0407, 50/50. Positif = spoiled; threshold 0.5.

## Grouped

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.732 | 0.826 | 0.732 | 0.712 | 0.971 |
| Decision Tree | 0.565 | 0.565 | 0.565 | 0.565 | 0.652 |
| KNN | 0.590 | 0.590 | 0.590 | 0.590 | 0.711 |
| Random Forest | 0.694 | 0.699 | 0.694 | 0.692 | 0.801 |

## Random Split

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.718 | 0.818 | 0.718 | 0.694 | 0.940 |
| Decision Tree | 0.628 | 0.629 | 0.628 | 0.628 | 0.701 |
| KNN | 0.569 | 0.570 | 0.569 | 0.569 | 0.646 |
| Random Forest | 0.691 | 0.695 | 0.691 | 0.689 | 0.745 |
