"""ROC curves on each split's INTERNAL 30% test — the training-time claim (GAS-ONLY).

Counterpart of roc_curves_by_split.py (validation): every model's ROC is computed on its
own split's held-out 30% of training/. Random Split holds out random rows (leaky: the
same trials were seen in training); Grouped holds out whole trials, then each held-out
trial is trimmed temporal-contiguously (trailing rows dropped) so train:test = 70:30 —
per the supervisor's requirement. Models are the saved dashboard pipelines (fit on the
identical 70% partitions, same RANDOM_STATE) — no retrain.
Layout matches the validation figure: 2 panels (Grouped | Random Split), 4 models each.

Output: results/figures/roc_curves_internal_by_split.png
        results/roc_curves_internal_auc.csv
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from src import config
config.NUMERIC_FEATURES = ["mq2", "mq135", "mq4"]                 # GAS ONLY
config.FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES

from src.load_data import load_relabeled
from src.preprocess import build_xy, prepare_df

warnings.filterwarnings("ignore")
RS = config.RANDOM_STATE
MODELS_DIR = config.PROJECT_ROOT / "dashboard" / "backend" / "models"
MODELS = [("Logistic Regression", "logistic_regression", "#10b981"),
          ("Decision Tree", "decision_tree", "#0ea5e9"),
          ("KNN", "knn", "#8b5cf6"),
          ("Random Forest", "random_forest", "#f59e0b")]

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

fig, axes = plt.subplots(1, len(SPLITS), figsize=(10.5, 5.2))
rows = []
for j, (sname, sslug, te) in enumerate(SPLITS):
    Xt, yt = X.iloc[te], y_arr[te]
    print(f"[{sname}] internal test: {len(te)} baris "
          f"(fresh {(yt==0).sum()} / spoiled {(yt==1).sum()})")
    ax = axes[j]
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#999", label="Acak (AUC 0.50)")
    for mname, mslug, color in MODELS:
        pipe = joblib.load(MODELS_DIR / f"{sslug}__{mslug}.joblib")
        proba = pipe.predict_proba(Xt)[:, 1]
        fpr, tpr, _ = roc_curve(yt, proba)
        auc = roc_auc_score(yt, proba)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{mname} (AUC {auc:.3f})")
        rows.append({"split": sname, "model": mname, "roc_auc": round(float(auc), 4)})
        print(f"  {mname:22s} AUC={auc:.3f}")
    ax.set_title(sname, fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    if j == 0:
        ax.set_ylabel("True Positive Rate")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, ls=":", lw=0.6, color="#ddd")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    print()

fig.suptitle("Kurva ROC — internal test saat training",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(config.FIGURES_DIR / "roc_curves_internal_by_split.png", dpi=150)
pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "roc_curves_internal_auc.csv", index=False)
print("Figure: results/figures/roc_curves_internal_by_split.png")
print("CSV   : results/roc_curves_internal_auc.csv")
