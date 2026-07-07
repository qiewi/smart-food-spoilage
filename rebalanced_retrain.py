"""Retrain (gas+env, NEW labels) 3 splits x 4 models, evaluate on validation 50/50.

Fresh training on the current training/ labels (Nasi Putih now 6h). Splits mirror the
dashboard: Stratified K-Fold & Grouped tune on ALL data (via their CV, refit all);
Stratified per Trial trains on its 70% holdout. Features = gas + environment (5 numeric +
food_type). Validation undersampled to 50/50 (seed=RANDOM_STATE). Full (19/81) also shown.
Metrics macro (recall = overall, not spoiled-only).
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
from src.load_data import load_relabeled
from src.preprocess import build_xy, make_preprocessor, prepare_df
from src.train import REFIT_METRIC, SCORING, _model_specs

warnings.filterwarnings("ignore")
RS, RF_N_ITER, N_JOBS = config.RANDOM_STATE, 40, -1
SPECS = _model_specs()
MODEL_NAMES = [s[0] for s in SPECS]
COLS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"]

prepared = prepare_df(load_relabeled(config.PROJECT_ROOT / "training")).reset_index(drop=True)
X, y, groups = build_xy(prepared)
val_df = prepare_df(load_relabeled(config.PROJECT_ROOT / "validation"))
Xval, yval, _ = build_xy(val_df)
yv = yval.to_numpy()
print(f"Fitur (GAS+ENV): {config.FEATURES}")
print(f"TRAIN {len(X)} baris (fresh {(y==0).sum()} / spoiled {(y==1).sum()}) "
      f"| VALID {len(yv)} baris (fresh {(yv==0).sum()} / spoiled {(yv==1).sum()})\n")

# rebalanced 50/50 validation
rng = np.random.default_rng(RS)
fi, si = np.where(yv == 0)[0], np.where(yv == 1)[0]
keep = np.sort(np.concatenate([fi, rng.choice(si, size=len(fi), replace=False)]))
Xb, yb = Xval.iloc[keep], yv[keep]

# per-Trial 70% train
strat = (y.astype(str) + "_" + groups.astype(str)).to_numpy()
vc = pd.Series(strat).value_counts()
bad = vc[vc < 2].index.tolist()
if bad:
    mm = np.isin(strat, bad)
    strat[mm] = y.astype(str).to_numpy()[mm]
spt_tr, _ = train_test_split(np.arange(len(X)), test_size=0.30, random_state=RS, stratify=strat)

CONFIGS = {
    "Grouped": (StratifiedGroupKFold(3, shuffle=True, random_state=RS), X, y, groups),
    "Stratified K-Fold": (StratifiedKFold(3, shuffle=True, random_state=RS), X, y, None),
    "Stratified per Trial": (StratifiedKFold(3, shuffle=True, random_state=RS),
                             X.iloc[spt_tr], y.iloc[spt_tr], None),
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


def met(ytrue, yp, pr):
    return {"accuracy": accuracy_score(ytrue, yp),
            "precision_macro": precision_score(ytrue, yp, average="macro", zero_division=0),
            "recall_macro": recall_score(ytrue, yp, average="macro", zero_division=0),
            "f1_macro": f1_score(ytrue, yp, average="macro", zero_division=0),
            "roc_auc": roc_auc_score(ytrue, pr)}


full, bal = {}, {}
for sname, (cv, xf, yf, grp) in CONFIGS.items():
    for name, est, grid, kind in SPECS:
        best = tune(est, grid, kind, cv, xf, yf, grp)
        full[(sname, name)] = met(yv, best.predict(Xval), best.predict_proba(Xval)[:, 1])
        bal[(sname, name)] = met(yb, best.predict(Xb), best.predict_proba(Xb)[:, 1])
    print(f"  [selesai] {sname}")

pd.set_option("display.width", 200)
for tag, d in [("VALIDATION 50/50", bal), ("VALIDATION ASLI (label baru)", full)]:
    print("\n" + "#" * 92 + f"\n# GAS+ENV (label baru) — {tag}\n" + "#" * 92)
    for sname in CONFIGS:
        tbl = pd.DataFrame({m: d[(sname, m)] for m in MODEL_NAMES}).T[COLS]
        print(f"\n--- {sname} ---")
        print(tbl.round(3).to_string())

pd.DataFrame(bal).T.to_csv(config.RESULTS_DIR / "rebalanced_retrain_5050.csv")
print("\nTersimpan: results/rebalanced_retrain_5050.csv")
