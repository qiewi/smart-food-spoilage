"""Confusion matrices on validation/ — 4 models x 3 splits (grid figure).

Uses the already-saved dashboard models (dashboard/backend/models) so the matrices are
consistent with the reported validation metrics. Splits: Grouped, Stratified K-Fold
(random), Stratified per Trial. Labels: 0=fresh, 1=spoiled.
Output: results/figures/confusion_matrices.png + printed acc/recall(spoiled) table.
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

from src import config
from src.load_data import load_relabeled
from src.preprocess import build_xy, prepare_df

warnings.filterwarnings("ignore")
MODELS_DIR = config.PROJECT_ROOT / "dashboard" / "backend" / "models"
SPLITS = [("Grouped", "grouped"),
          ("Stratified K-Fold", "stratifiedkfold"),
          ("Stratified per Trial", "stratifiedpertrial")]
MODELS = [("Logistic Regression", "logistic_regression"),
          ("Decision Tree", "decision_tree"),
          ("KNN", "knn"),
          ("Random Forest", "random_forest")]
CLASSES = ["fresh", "spoiled"]

val_df = prepare_df(load_relabeled(config.PROJECT_ROOT / "validation"))
Xval, yval, _ = build_xy(val_df)
yv = yval.to_numpy()
print(f"VALID: {len(yv)} baris | fresh={int((yv == 0).sum())} spoiled={int((yv == 1).sum())}\n")

fig, axes = plt.subplots(len(MODELS), len(SPLITS), figsize=(10.5, 12.5))
print(f"{'Model':22s} {'Split':22s} {'acc':>6s} {'rec(spoiled)':>13s}")
print("-" * 66)
for i, (mname, mslug) in enumerate(MODELS):
    for j, (sname, sslug) in enumerate(SPLITS):
        pipe = joblib.load(MODELS_DIR / f"{sslug}__{mslug}.joblib")
        yp = pipe.predict(Xval)
        cm = confusion_matrix(yv, yp, labels=[0, 1])
        acc = accuracy_score(yv, yp)
        rec_sp = recall_score(yv, yp, pos_label=config.POSITIVE_LABEL, zero_division=0)
        print(f"{mname:22s} {sname:22s} {acc:6.3f} {rec_sp:13.3f}")

        ax = axes[i, j]
        ax.imshow(cm, cmap="Greens", vmin=0, vmax=cm.max())
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, f"{v}", ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if v > cm.max() * 0.6 else "#111")
        ax.set_xticks([0, 1]); ax.set_xticklabels(CLASSES, fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(CLASSES, fontsize=8)
        ax.set_xlabel(f"Prediksi\nacc={acc:.2f}  rec(sp)={rec_sp:.2f}", fontsize=8)
        if i == 0:
            ax.set_title(sname, fontsize=11, fontweight="bold", pad=10)
        if j == 0:
            ax.set_ylabel(f"{mname}\n\nAktual", fontsize=9, fontweight="bold")

fig.suptitle("Confusion Matrix pada data validation — 4 model × 3 metode split",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(config.FIGURES_DIR / "confusion_matrices.png", dpi=150)
print("\nFigure: results/figures/confusion_matrices.png")
