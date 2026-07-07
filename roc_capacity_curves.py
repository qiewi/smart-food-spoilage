"""ROC-vs-capacity curves ("epoch sampai mentok") for the 4 models — standalone analysis.

The lecturer asked to see, per model, how ROC behaves as we increase the iterative/
capacity parameter until it saturates. Only LR (via SGD) has literal epochs; DT/RF/KNN
use their natural capacity knob, swept over the CITED grid ranges (src.train._model_specs):
  - Logistic Regression -> epoch (SGDClassifier, log-loss)
  - Random Forest       -> n_estimators  [cited: 50,100,200,500,1000]
  - Decision Tree       -> max_depth     [cited: 3,5,7,9,15,None]
  - KNN                 -> n_neighbors k [cited: 1,3,5,7,9,11]

For each value we train on ALL training/ data and report ROC-AUC on the training data
(in-sample) and on validation/ (honest). The train-vs-validation gap = the leakage story.
Output: figure results/figures/roc_capacity_curves.png + results/roc_capacity_curves.csv
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

from src import config
from src.load_data import load_relabeled
from src.preprocess import build_xy, make_preprocessor, prepare_df

warnings.filterwarnings("ignore")
RS = config.RANDOM_STATE

# ---------- data (preprocess once; trees are scale-invariant, SGD/KNN need scaling) ----------
prepared = prepare_df(load_relabeled(config.PROJECT_ROOT / "training")).reset_index(drop=True)
X, y, _ = build_xy(prepared)
y_arr = y.to_numpy()
val_df = prepare_df(load_relabeled(config.PROJECT_ROOT / "validation"))
Xval, yval, _ = build_xy(val_df)
yval_arr = yval.to_numpy()
prep = make_preprocessor()
Xt = prep.fit_transform(X)          # fit on TRAIN only
Xvt = prep.transform(Xval)
print(f"TRAIN {Xt.shape} | VALID {Xvt.shape}\n")


def auc(model_proba_tr, model_proba_val):
    return roc_auc_score(y_arr, model_proba_tr), roc_auc_score(yval_arr, model_proba_val)


rows = []

# ---------- Logistic Regression: ROC vs epoch (SGD) ----------
EPOCHS = list(range(1, 41))
cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_arr)
sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                    class_weight={0: cw[0], 1: cw[1]}, random_state=RS)
rng = np.random.default_rng(RS)
lr_tr, lr_val = [], []
for ep in EPOCHS:
    order = rng.permutation(len(y_arr))
    sgd.partial_fit(Xt[order], y_arr[order], classes=np.array([0, 1]))
    a_tr, a_val = auc(sgd.predict_proba(Xt)[:, 1], sgd.predict_proba(Xvt)[:, 1])
    lr_tr.append(a_tr)
    lr_val.append(a_val)
    rows.append({"model": "Logistic Regression", "param": "epoch", "value": ep,
                 "train_auc": a_tr, "val_auc": a_val})

# ---------- Random Forest: ROC vs n_estimators ----------
N_EST = [10, 25, 50, 100, 200, 350, 500, 750, 1000]           # cited: 50,100,200,500,1000
rf_tr, rf_val = [], []
for n in N_EST:
    m = RandomForestClassifier(n_estimators=n, random_state=RS, n_jobs=-1).fit(Xt, y_arr)
    a_tr, a_val = auc(m.predict_proba(Xt)[:, 1], m.predict_proba(Xvt)[:, 1])
    rf_tr.append(a_tr)
    rf_val.append(a_val)
    rows.append({"model": "Random Forest", "param": "n_estimators", "value": n,
                 "train_auc": a_tr, "val_auc": a_val})

# ---------- Decision Tree: ROC vs max_depth (25 = None/unlimited) ----------
DEPTHS = [1, 2, 3, 5, 7, 9, 11, 15, 25]                        # cited: 3,5,7,9,15,None
dt_tr, dt_val = [], []
for d in DEPTHS:
    m = DecisionTreeClassifier(max_depth=(None if d == 25 else d), random_state=RS).fit(Xt, y_arr)
    a_tr, a_val = auc(m.predict_proba(Xt)[:, 1], m.predict_proba(Xvt)[:, 1])
    dt_tr.append(a_tr)
    dt_val.append(a_val)
    rows.append({"model": "Decision Tree", "param": "max_depth", "value": ("None" if d == 25 else d),
                 "train_auc": a_tr, "val_auc": a_val})

# ---------- KNN: ROC vs k ----------
KS = [1, 3, 5, 7, 9, 11, 15, 21, 31]                          # cited: 1,3,5,7,9,11
kn_tr, kn_val = [], []
for k in KS:
    m = KNeighborsClassifier(n_neighbors=k).fit(Xt, y_arr)
    a_tr, a_val = auc(m.predict_proba(Xt)[:, 1], m.predict_proba(Xvt)[:, 1])
    kn_tr.append(a_tr)
    kn_val.append(a_val)
    rows.append({"model": "KNN", "param": "k", "value": k, "train_auc": a_tr, "val_auc": a_val})

pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "roc_capacity_curves.csv", index=False)

# ---------- figure ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
GREEN, GRAY = "#10b981", "#94a3b8"


def panel(ax, xvals, tr, val, title, xlabel, logx=False, xticklabels=None):
    ax.plot(xvals, val, "o-", color=GREEN, label="Validasi (jujur)")
    ax.plot(xvals, tr, "s--", color=GRAY, label="Training (in-sample)")
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.30, 1.02)
    if logx:
        ax.set_xscale("log")
    if xticklabels is not None:
        ax.set_xticks(xvals)
        ax.set_xticklabels(xticklabels)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")


panel(axes[0, 0], EPOCHS, lr_tr, lr_val, "Logistic Regression (SGD) — Epoch", "Epoch")
panel(axes[0, 1], N_EST, rf_tr, rf_val, "Random Forest — Jumlah Pohon", "n_estimators", logx=True)
panel(axes[1, 0], DEPTHS, dt_tr, dt_val, "Decision Tree — Kedalaman", "max_depth",
      xticklabels=[str(d) if d != 25 else "None" for d in DEPTHS])
panel(axes[1, 1], KS, kn_tr, kn_val, "KNN — Jumlah Tetangga (k)", "k")
fig.suptitle("Kurva ROC-AUC vs parameter iteratif/kapasitas (sampai jenuh) — validasi vs training",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "roc_capacity_curves.png", dpi=150)


def summarize(name, xvals, val, labels=None):
    best_i = int(np.argmax(val))
    xb = labels[best_i] if labels else xvals[best_i]
    print(f"  {name:22s} val-AUC puncak {max(val):.3f} @ {xb} | akhir {val[-1]:.3f}")


print("Ringkasan (ROC-AUC validasi):")
summarize("LR (epoch)", EPOCHS, lr_val)
summarize("RF (n_estimators)", N_EST, rf_val)
summarize("DT (max_depth)", DEPTHS, dt_val, [str(d) if d != 25 else "None" for d in DEPTHS])
summarize("KNN (k)", KS, kn_val)
print("\nFigure: results/figures/roc_capacity_curves.png | CSV: results/roc_capacity_curves.csv")
