"""Confusion matrices on the REBALANCED (50/50) validation — 4 models x 3 splits.

Same grid as confusion_matrices.py but validation is undersampled to 50/50 (spoiled cut to
match fresh, seed=RANDOM_STATE). Uses the saved gas+env dashboard models. Labels 0=fresh,
1=spoiled. Subtitle shows accuracy + recall-macro (overall).
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
SPLITS = [("Grouped", "grouped"), ("Stratified K-Fold", "stratifiedkfold"),
          ("Stratified per Trial", "stratifiedpertrial")]
MODELS = [("Logistic Regression", "logistic_regression"), ("Decision Tree", "decision_tree"),
          ("KNN", "knn"), ("Random Forest", "random_forest")]
CLASSES = ["fresh", "spoiled"]

val_df = prepare_df(load_relabeled(config.PROJECT_ROOT / "validation"))
Xval, yval, _ = build_xy(val_df)
yv = yval.to_numpy()

# undersample spoiled -> 50/50
rng = np.random.default_rng(config.RANDOM_STATE)
fi, si = np.where(yv == 0)[0], np.where(yv == 1)[0]
keep = np.sort(np.concatenate([fi, rng.choice(si, size=len(fi), replace=False)]))
Xb, yb = Xval.iloc[keep], yv[keep]
print(f"Validation 50/50: {len(yb)} baris (fresh {(yb==0).sum()} / spoiled {(yb==1).sum()})\n")

fig, axes = plt.subplots(len(MODELS), len(SPLITS), figsize=(10.5, 12.5))
for i, (mname, mslug) in enumerate(MODELS):
    for j, (sname, sslug) in enumerate(SPLITS):
        pipe = joblib.load(MODELS_DIR / f"{sslug}__{mslug}.joblib")
        yp = pipe.predict(Xb)
        cm = confusion_matrix(yb, yp, labels=[0, 1])
        acc = accuracy_score(yb, yp)
        recm = recall_score(yb, yp, average="macro", zero_division=0)
        ax = axes[i, j]
        ax.imshow(cm, cmap="Greens", vmin=0, vmax=cm.max())
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, f"{v}", ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if v > cm.max() * 0.6 else "#111")
        ax.set_xticks([0, 1]); ax.set_xticklabels(CLASSES, fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(CLASSES, fontsize=8)
        ax.set_xlabel(f"Prediksi\nacc={acc:.2f}  rec-macro={recm:.2f}", fontsize=8)
        if i == 0:
            ax.set_title(sname, fontsize=11, fontweight="bold", pad=10)
        if j == 0:
            ax.set_ylabel(f"{mname}\n\nAktual", fontsize=9, fontweight="bold")

fig.suptitle("Confusion Matrix — validation REBALANCED 50/50 (gas+env) — 4 model × 3 split",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(config.FIGURES_DIR / "confusion_matrices_5050.png", dpi=150)
print("Figure: results/figures/confusion_matrices_5050.png")
