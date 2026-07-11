# Evaluasi Internal Test — Nasi Putih (gas-only)

Metrik dari prediksi **out-of-fold (OOF) 3-fold digabung** (lintas-trial),
sumber sama dengan `confusion_matrices_internal.png` & `roc_curves_internal.png`
→ angka tabel = angka gambar. Positif = spoiled; threshold 0.5.

## Grouped

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.639 | 0.611 | 0.618 | 0.613 | 0.653 |
| Decision Tree | 0.688 | 0.660 | 0.668 | 0.663 | 0.750 |
| KNN | 0.690 | 0.651 | 0.641 | 0.645 | 0.694 |
| Random Forest | 0.631 | 0.609 | 0.617 | 0.609 | 0.721 |

## Random Split

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.674 | 0.635 | 0.629 | 0.631 | 0.798 |
| Decision Tree | 0.900 | 0.890 | 0.887 | 0.888 | 0.945 |
| KNN | 0.909 | 0.903 | 0.893 | 0.898 | 0.965 |
| Random Forest | 0.914 | 0.906 | 0.900 | 0.903 | 0.971 |
