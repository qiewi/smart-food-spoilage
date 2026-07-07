"""GAS-ONLY ROC-vs-capacity curves per split — ROC measured on REBALANCED 50/50 validation.

Same as roc_capacity_by_split_5050.py but features overridden to mq2/mq135/mq4 + food_type
(NO humidity/tempC). Uses current (new) labels.
Output: results/figures/roc_capacity_by_split_5050_gasonly.png + .csv
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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

from src import config
config.NUMERIC_FEATURES = ["mq2", "mq135", "mq4"]                 # GAS ONLY
config.FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES

from src.load_data import load_relabeled
from src.preprocess import build_xy, make_preprocessor, prepare_df

warnings.filterwarnings("ignore")
RS = config.RANDOM_STATE

prepared = prepare_df(load_relabeled(config.PROJECT_ROOT / "training")).reset_index(drop=True)
X, y, groups = build_xy(prepared)
y_arr, run_arr, elapsed = y.to_numpy(), groups.to_numpy(), prepared["elapsed_sec"].to_numpy()
val_df = prepare_df(load_relabeled(config.PROJECT_ROOT / "validation"))
Xval, yval, _ = build_xy(val_df)
yval_arr = yval.to_numpy()

# rebalance validation to 50/50
rng0 = np.random.default_rng(RS)
fi, si = np.where(yval_arr == 0)[0], np.where(yval_arr == 1)[0]
keep = np.sort(np.concatenate([fi, rng0.choice(si, size=len(fi), replace=False)]))
Xval, yval_arr = Xval.iloc[keep], yval_arr[keep]
print(f"Fitur (GAS-ONLY): {config.FEATURES} | Validation 50/50: {len(yval_arr)} baris\n")

N = len(X)
idx = np.arange(N)
skf_tr, _ = next(StratifiedShuffleSplit(1, test_size=0.30, random_state=RS).split(X, y))
g_tr, _ = next(StratifiedGroupKFold(3, shuffle=True, random_state=RS).split(X, y, groups))

SPLITS = {"Grouped": g_tr, "Random Split": skf_tr}
SPLIT_COLOR = {"Grouped": "#10b981", "Random Split": "#f59e0b"}

EPOCHS = list(range(1, 41))
N_EST = [10, 25, 50, 100, 200, 350, 500, 750, 1000]
DEPTHS = [1, 2, 3, 5, 7, 9, 11, 15, 25]
KS = [1, 3, 5, 7, 9, 11, 15, 21, 31]
rows = []
curves = {m: {} for m in ["Logistic Regression", "Random Forest", "Decision Tree", "KNN"]}

for split_name, tr_idx in SPLITS.items():
    prep = make_preprocessor()
    Xtr = prep.fit_transform(X.iloc[tr_idx])
    Xvt = prep.transform(Xval)
    ytr = y_arr[tr_idx]

    def rec(model, param, value, proba_val):
        a = roc_auc_score(yval_arr, proba_val)
        rows.append({"split": split_name, "model": model, "param": param, "value": value, "val_auc": a})
        return a

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=ytr)
    sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                        class_weight={0: cw[0], 1: cw[1]}, random_state=RS)
    rng = np.random.default_rng(RS)
    lr = []
    for ep in EPOCHS:
        order = rng.permutation(len(ytr))
        sgd.partial_fit(Xtr[order], ytr[order], classes=np.array([0, 1]))
        lr.append(rec("Logistic Regression", "epoch", ep, sgd.predict_proba(Xvt)[:, 1]))
    curves["Logistic Regression"][split_name] = (EPOCHS, lr)

    curves["Random Forest"][split_name] = (N_EST, [rec("Random Forest", "n_estimators", n,
        RandomForestClassifier(n_estimators=n, random_state=RS, n_jobs=-1).fit(Xtr, ytr)
        .predict_proba(Xvt)[:, 1]) for n in N_EST])

    curves["Decision Tree"][split_name] = (DEPTHS, [rec("Decision Tree", "max_depth",
        ("None" if d == 25 else d), DecisionTreeClassifier(max_depth=(None if d == 25 else d),
        random_state=RS).fit(Xtr, ytr).predict_proba(Xvt)[:, 1]) for d in DEPTHS])

    curves["KNN"][split_name] = (KS, [rec("KNN", "k", k,
        KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr).predict_proba(Xvt)[:, 1]) for k in KS])
    print(f"  [selesai] {split_name}")

pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "roc_capacity_by_split_5050_gasonly.csv", index=False)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
PANELS = [("Logistic Regression", "Epoch", axes[0, 0], False, None),
          ("Random Forest", "n_estimators", axes[0, 1], True, None),
          ("Decision Tree", "max_depth", axes[1, 0], False,
           [str(d) if d != 25 else "None" for d in DEPTHS]),
          ("KNN", "k (jumlah tetangga)", axes[1, 1], False, None)]
for model, xlabel, ax, logx, xticklab in PANELS:
    for split_name, (xv, yvv) in curves[model].items():
        ax.plot(xv, yvv, "o-", color=SPLIT_COLOR[split_name], label=split_name, markersize=4)
    ax.set_title(model, fontweight="bold", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ROC-AUC (validation 50/50)")
    ax.set_ylim(0.30, 1.02)
    if logx:
        ax.set_xscale("log")
    if xticklab is not None:
        ax.set_xticks(DEPTHS)
        ax.set_xticklabels(xticklab)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
fig.suptitle("GAS-ONLY — ROC-AUC pada validation 50/50 vs parameter kapasitas — per split (train 70%)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "roc_capacity_by_split_5050_gasonly.png", dpi=150)
print("\nFigure: results/figures/roc_capacity_by_split_5050_gasonly.png")
