"""GAS-ONLY confusion matrices on each split's INTERNAL 30% test — the training-time claim.

Counterpart of confusion_matrices_5050_gasonly.py (validation): here every model is
evaluated on its own split's held-out 30% of training/ ("internal test", what the
training phase claims). Random Split holds out random rows (leaky: same trials seen in
train); Grouped holds out whole trials, then each held-out trial is trimmed
temporal-contiguously (trailing rows dropped) so train:test = 70:30 — per the
supervisor's requirement; both classes remain since labels are time-thresholded.
Models are the saved dashboard pipelines — fit on the identical 70% partitions
(same RANDOM_STATE), so no retraining.
Layout matches the validation figure: 2 splits (rows) x 4 models (cols), landscape.

Output: results/figures/confusion_matrices_internal_gasonly.png
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from src import config
config.NUMERIC_FEATURES = ["mq2", "mq135", "mq4"]                 # GAS ONLY
config.FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES

from src.load_data import load_relabeled
from src.preprocess import build_xy, prepare_df

warnings.filterwarnings("ignore")
RS = config.RANDOM_STATE
MODELS_DIR = config.PROJECT_ROOT / "dashboard" / "backend" / "models"
MODELS = [("Logistic Regression", "logistic_regression"), ("Decision Tree", "decision_tree"),
          ("KNN", "knn"), ("Random Forest", "random_forest")]
CLASSES = ["fresh", "spoiled"]

prepared = prepare_df(load_relabeled(config.PROJECT_ROOT / "training")).reset_index(drop=True)
X, y, groups = build_xy(prepared)
y_arr, run_arr, elapsed = y.to_numpy(), groups.to_numpy(), prepared["elapsed_sec"].to_numpy()

# identical partitions to export_models.py (same seed -> same 70/30)
skf_tr, skf_te = next(StratifiedShuffleSplit(1, test_size=0.30, random_state=RS).split(X, y))
g_tr, g_te = next(StratifiedGroupKFold(3, shuffle=True, random_state=RS).split(X, y, groups))
# trim each held-out trial's trailing rows so train:test = 70:30 (keep beginning -> both classes)
target = int(round((0.30 / 0.70) * len(g_tr)))
kf = min(1.0, target / len(g_te))
kept = []
for r in np.unique(run_arr[g_te]):
    rows = g_te[run_arr[g_te] == r]
    rows = rows[np.argsort(elapsed[rows])]
    kept.extend(rows[:max(1, int(round(len(rows) * kf)))])
g_te = np.array(sorted(kept))
SPLITS = [("Grouped", "grouped", g_te), ("Random Split", "stratifiedkfold", skf_te)]
for sname, _, te in SPLITS:
    yt = y_arr[te]
    print(f"  [{sname:12s}] internal test: {len(te)} baris "
          f"(fresh {(yt==0).sum()} / spoiled {(yt==1).sum()})")

fig, axes = plt.subplots(len(SPLITS), len(MODELS), figsize=(14, 7))
for i, (sname, sslug, te) in enumerate(SPLITS):
    Xt, yt = X.iloc[te], y_arr[te]
    for j, (mname, mslug) in enumerate(MODELS):
        pipe = joblib.load(MODELS_DIR / f"{sslug}__{mslug}.joblib")
        yp = pipe.predict(Xt)
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        acc = accuracy_score(yt, yp)
        recm = recall_score(yt, yp, average="macro", zero_division=0)
        ax = axes[i, j]
        ax.imshow(cm, cmap="Greens", vmin=0, vmax=cm.max())
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, f"{v}", ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if v > cm.max() * 0.6 else "#111")
        ax.set_xticks([0, 1]); ax.set_xticklabels(CLASSES, fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(CLASSES, fontsize=8)
        ax.set_xlabel(f"Prediksi\nacc={acc:.2f}  rec-macro={recm:.2f}", fontsize=8)
        if i == 0:
            ax.set_title(mname, fontsize=11, fontweight="bold", pad=10)
        if j == 0:
            ax.set_ylabel(f"{sname}\n\nAktual", fontsize=9, fontweight="bold")

fig.suptitle("Confusion Matrix — internal test saat training — 2 split × 4 model",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(config.FIGURES_DIR / "confusion_matrices_internal_gasonly.png", dpi=150)
print("\nFigure: results/figures/confusion_matrices_internal_gasonly.png")
