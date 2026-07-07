"""GAS-ONLY comparison: retune 3 splits x 4 models on gas-only features, evaluate on
validation/ (full 19/81) and rebalanced (50/50). Standalone — does not touch dashboard.

Splits mirror the dashboard: Stratified K-Fold & Grouped tune on ALL data (via their CV,
refit all); Stratified per Trial trains on its 70% holdout. Features overridden to
mq2/mq135/mq4 + food_type (NO humidity/tempC).
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline

from src import config
config.NUMERIC_FEATURES = ["mq2", "mq135", "mq4"]                 # GAS ONLY
config.FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES

from src.load_data import load_relabeled
from src.preprocess import build_xy, make_preprocessor, prepare_df
from src.train import REFIT_METRIC, SCORING, _model_specs

warnings.filterwarnings("ignore")
RS = config.RANDOM_STATE
RF_N_ITER, N_JOBS = 40, -1
SPECS = _model_specs()
COLS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "recall_spoiled", "roc_auc"]

prepared = prepare_df(load_relabeled(config.PROJECT_ROOT / "training")).reset_index(drop=True)
X, y, groups = build_xy(prepared)
y_arr = y.to_numpy()
val_df = prepare_df(load_relabeled(config.PROJECT_ROOT / "validation"))
Xval, yval, _ = build_xy(val_df)
yv = yval.to_numpy()
print(f"Fitur (GAS-ONLY): {config.FEATURES}\n")

# rebalanced 50/50 validation
rng = np.random.default_rng(RS)
fi, si = np.where(yv == 0)[0], np.where(yv == 1)[0]
keep = np.sort(np.concatenate([fi, rng.choice(si, size=len(fi), replace=False)]))
Xb, yb = Xval.iloc[keep], yv[keep]

# per-Trial 70% train (stratify by label x trial)
strat = (y.astype(str) + "_" + groups.astype(str)).to_numpy()
vc = pd.Series(strat).value_counts()
bad = vc[vc < 2].index.tolist()
if bad:
    m = np.isin(strat, bad)
    strat[m] = y.astype(str).to_numpy()[m]
spt_tr, _ = train_test_split(np.arange(len(X)), test_size=0.30, random_state=RS, stratify=strat)
Xtr70, ytr70 = X.iloc[spt_tr], y.iloc[spt_tr]

CONFIGS = {
    "Grouped": (StratifiedGroupKFold(3, shuffle=True, random_state=RS), X, y, groups),
    "Stratified K-Fold": (StratifiedKFold(3, shuffle=True, random_state=RS), X, y, None),
    "Stratified per Trial": (StratifiedKFold(3, shuffle=True, random_state=RS), Xtr70, ytr70, None),
}


def tune(est, grid, kind, cv, xf, yf, grp):
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", clone(est))])
    try:
        pipe.set_params(clf__n_jobs=1)
    except ValueError:
        pass
    s = (RandomizedSearchCV(pipe, grid, n_iter=RF_N_ITER, scoring=SCORING, refit=REFIT_METRIC,
                            cv=cv, random_state=RS, n_jobs=N_JOBS) if kind == "random"
         else GridSearchCV(pipe, grid, scoring=SCORING, refit=REFIT_METRIC, cv=cv, n_jobs=N_JOBS))
    s.fit(xf, yf, groups=grp) if grp is not None else s.fit(xf, yf)
    return s.best_estimator_


def metrics(ytrue, yp, pr):
    return {"accuracy": accuracy_score(ytrue, yp),
            "precision_macro": precision_score(ytrue, yp, average="macro", zero_division=0),
            "recall_macro": recall_score(ytrue, yp, average="macro", zero_division=0),
            "f1_macro": f1_score(ytrue, yp, average="macro", zero_division=0),
            "recall_spoiled": recall_score(ytrue, yp, pos_label=1, zero_division=0),
            "roc_auc": roc_auc_score(ytrue, pr)}


full, bal = {}, {}
for sname, (cv, xf, yf, grp) in CONFIGS.items():
    for name, est, grid, kind in SPECS:
        best = tune(est, grid, kind, cv, xf, yf, grp)
        full[(sname, name)] = metrics(yv, best.predict(Xval), best.predict_proba(Xval)[:, 1])
        bal[(sname, name)] = metrics(yb, best.predict(Xb), best.predict_proba(Xb)[:, 1])
    print(f"  [selesai] {sname}")

pd.set_option("display.width", 200)
MODEL_NAMES = [s[0] for s in SPECS]
for tag, d in [("VALIDATION ASLI (19/81)", full), ("VALIDATION REBALANCED (50/50)", bal)]:
    print("\n" + "#" * 92 + f"\n# GAS-ONLY — {tag}\n" + "#" * 92)
    for sname in CONFIGS:
        tbl = pd.DataFrame({m: d[(sname, m)] for m in MODEL_NAMES}).T[COLS]
        print(f"\n--- {sname} ---")
        print(tbl.round(3).to_string())

pd.DataFrame(bal).T.to_csv(config.RESULTS_DIR / "gasonly_rebalanced_tables.csv")
print("\nTersimpan: results/gasonly_rebalanced_tables.csv")
