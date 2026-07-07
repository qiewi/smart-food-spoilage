"""ROC-vs-capacity curves per split — ROC measured on REBALANCED 50/50 validation.

Same as roc_capacity_by_split.py but the validation set is undersampled to 50/50 (spoiled
cut to match fresh, seed=RANDOM_STATE) before computing ROC-AUC. Since AUC is
prevalence-invariant, curves should be nearly identical to the full-validation version.
Output: results/figures/roc_capacity_by_split_5050.png + results/roc_capacity_by_split_5050.csv
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
from sklearn.model_selection import (StratifiedGroupKFold,
                                     StratifiedShuffleSplit, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

from src import config
from src.load_data import load_relabeled
from src.preprocess import build_xy, make_preprocessor, prepare_df

warnings.filterwarnings("ignore")
RS = config.RANDOM_STATE

# ---------- data ----------
prepared = prepare_df(load_relabeled(config.PROJECT_ROOT / "training")).reset_index(drop=True)
X, y, groups = build_xy(prepared)
y_arr, run_arr, elapsed = y.to_numpy(), groups.to_numpy(), prepared["elapsed_sec"].to_numpy()
val_df = prepare_df(load_relabeled(config.PROJECT_ROOT / "validation"))
Xval, yval, _ = build_xy(val_df)
yval_arr = yval.to_numpy()

# ---- rebalance validation to 50/50 (undersample spoiled) ----
rng0 = np.random.default_rng(RS)
fi, si = np.where(yval_arr == 0)[0], np.where(yval_arr == 1)[0]
keep = np.sort(np.concatenate([fi, rng0.choice(si, size=len(fi), replace=False)]))
Xval, yval_arr = Xval.iloc[keep], yval_arr[keep]
print(f"Validation 50/50: {len(yval_arr)} baris (fresh {(yval_arr==0).sum()} / spoiled {(yval_arr==1).sum()})\n")

N = len(X)
idx = np.arange(N)

# ---------- 70/30 train indices per split ----------
skf_tr, _ = next(StratifiedShuffleSplit(1, test_size=0.30, random_state=RS).split(X, y))

strat = (y.astype(str) + "_" + groups.astype(str)).to_numpy()
vc = pd.Series(strat).value_counts()
bad = vc[vc < 2].index.tolist()
if bad:
    m = np.isin(strat, bad)
    strat[m] = y.astype(str).to_numpy()[m]
    if (pd.Series(strat).value_counts() < 2).any():
        strat = y.astype(str).to_numpy()
spt_tr, _ = train_test_split(idx, test_size=0.30, random_state=RS, stratify=strat)

g_tr, g_te = next(StratifiedGroupKFold(3, shuffle=True, random_state=RS).split(X, y, groups))
target_test = int(round((0.30 / 0.70) * len(g_tr)))
keep_frac = min(1.0, target_test / len(g_te))
kept = []
for r in np.unique(run_arr[g_te]):
    rowsr = g_te[run_arr[g_te] == r]
    rowsr = rowsr[np.argsort(elapsed[rowsr])]
    kept.extend(rowsr[:max(1, int(round(len(rowsr) * keep_frac)))])

SPLITS = {"Stratified K-Fold": skf_tr, "Grouped": g_tr, "Stratified per Trial": spt_tr}
SPLIT_COLOR = {"Stratified K-Fold": "#f59e0b", "Grouped": "#10b981",
               "Stratified per Trial": "#6366f1"}

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

pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "roc_capacity_by_split_5050.csv", index=False)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
PANELS = [("Logistic Regression", "Epoch", axes[0, 0], False, None),
          ("Random Forest", "n_estimators", axes[0, 1], True, None),
          ("Decision Tree", "max_depth", axes[1, 0], False,
           [str(d) if d != 25 else "None" for d in DEPTHS]),
          ("KNN", "k (jumlah tetangga)", axes[1, 1], False, None)]
for model, xlabel, ax, logx, xticklab in PANELS:
    for split_name, (xv, yv) in curves[model].items():
        ax.plot(xv, yv, "o-", color=SPLIT_COLOR[split_name], label=split_name, markersize=4)
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
fig.suptitle("ROC-AUC pada validation 50/50 vs parameter kapasitas — per metode split (train 70%)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "roc_capacity_by_split_5050.png", dpi=150)
print("\nFigure: results/figures/roc_capacity_by_split_5050.png | CSV: results/roc_capacity_by_split_5050.csv")
