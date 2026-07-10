"""ROC curves — each model, per each split (GAS-ONLY, validation REBALANCED 50/50).

2 panels (one per split): Grouped and Random Split. Each panel
overlays the 4 models' ROC curves with their AUC in the legend. Uses the saved dashboard
models (trained on each split's 70%) and the identical 50/50 rebalanced validation, so the
AUCs match the report tables / dashboard exactly. Labels 0=fresh, 1=spoiled (positive).

Output: results/figures/roc_curves_by_split.png  +  results/roc_curves_by_split_auc.csv
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, roc_curve

from src import config
from src.load_data import load_relabeled
from src.preprocess import build_xy, prepare_df

warnings.filterwarnings("ignore")
MODELS_DIR = config.PROJECT_ROOT / "dashboard" / "backend" / "models"
SPLITS = [("Grouped", "grouped"), ("Random Split", "stratifiedkfold")]
MODELS = [("Logistic Regression", "logistic_regression", "#10b981"),
          ("Decision Tree", "decision_tree", "#0ea5e9"),
          ("KNN", "knn", "#8b5cf6"),
          ("Random Forest", "random_forest", "#f59e0b")]

# ---------- data: identical 50/50 rebalanced validation ----------
val_df = prepare_df(load_relabeled(config.PROJECT_ROOT / "validation"))
Xval, yval, _ = build_xy(val_df)
yv = yval.to_numpy()
rng = np.random.default_rng(config.RANDOM_STATE)
fi, si = np.where(yv == 0)[0], np.where(yv == 1)[0]
keep = np.sort(np.concatenate([fi, rng.choice(si, size=len(fi), replace=False)]))
Xb, yb = Xval.iloc[keep], yv[keep]
print(f"Validation 50/50: {len(yb)} baris (fresh {(yb==0).sum()} / spoiled {(yb==1).sum()})\n")

# ---------- plot ----------
fig, axes = plt.subplots(1, len(SPLITS), figsize=(10.5, 5.2))
rows = []
for j, (sname, sslug) in enumerate(SPLITS):
    ax = axes[j]
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#999", label="Acak (AUC 0.50)")
    for mname, mslug, color in MODELS:
        pipe = joblib.load(MODELS_DIR / f"{sslug}__{mslug}.joblib")
        proba = pipe.predict_proba(Xb)[:, 1]
        fpr, tpr, _ = roc_curve(yb, proba)
        auc = roc_auc_score(yb, proba)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{mname} (AUC {auc:.3f})")
        rows.append({"split": sname, "model": mname, "roc_auc": round(float(auc), 4)})
        print(f"  [{sname:20s}] {mname:22s} AUC={auc:.3f}")
    ax.set_title(sname, fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    if j == 0:
        ax.set_ylabel("True Positive Rate")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, ls=":", lw=0.6, color="#ddd")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    print()

fig.suptitle("Kurva ROC — 4 model per split (validasi)",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
out_png = config.FIGURES_DIR / "roc_curves_by_split.png"
fig.savefig(out_png, dpi=150)

out_csv = config.RESULTS_DIR / "roc_curves_by_split_auc.csv"
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"Figure: results/figures/roc_curves_by_split.png")
print(f"CSV   : results/roc_curves_by_split_auc.csv")
