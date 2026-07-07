"""GAS-ONLY: same models as the gas-only internal test (train 70% per split, new labels),
evaluated on VALIDATION (full 19/81 and rebalanced 50/50), per split x model.

Mirrors gasenv_internal_to_validation.py but features overridden to mq2/mq135/mq4 +
food_type. Metrics macro (recall = overall).
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedGroupKFold, StratifiedKFold,
                                     StratifiedShuffleSplit, train_test_split)
from sklearn.pipeline import Pipeline

from src import config
config.NUMERIC_FEATURES = ["mq2", "mq135", "mq4"]                 # GAS ONLY
config.FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES

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
idx = np.arange(len(X))
val_df = prepare_df(load_relabeled(config.PROJECT_ROOT / "validation"))
Xval, yval, _ = build_xy(val_df)
yv = yval.to_numpy()
print(f"Fitur (GAS-ONLY): {config.FEATURES}")
print(f"VALID {len(yv)} baris (fresh {(yv==0).sum()} / spoiled {(yv==1).sum()})\n")

rng = np.random.default_rng(RS)
fi, si = np.where(yv == 0)[0], np.where(yv == 1)[0]
keepb = np.sort(np.concatenate([fi, rng.choice(si, size=len(fi), replace=False)]))
Xb, yb = Xval.iloc[keepb], yv[keepb]

skf_tr, _ = next(StratifiedShuffleSplit(1, test_size=0.30, random_state=RS).split(X, y))
strat = (y.astype(str) + "_" + groups.astype(str)).to_numpy()
vc = pd.Series(strat).value_counts()
badk = vc[vc < 2].index.tolist()
if badk:
    mm = np.isin(strat, badk)
    strat[mm] = y.astype(str).to_numpy()[mm]
pt_tr, _ = train_test_split(idx, test_size=0.30, random_state=RS, stratify=strat)
g_tr, _ = next(StratifiedGroupKFold(3, shuffle=True, random_state=RS).split(X, y, groups))
TRAIN = {"Grouped": g_tr, "Stratified K-Fold": skf_tr, "Stratified per Trial": pt_tr}


def tune(est, grid, kind, xf, yf):
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", clone(est))])
    try:
        pipe.set_params(clf__n_jobs=1)
    except ValueError:
        pass
    cv = StratifiedKFold(3, shuffle=True, random_state=RS)
    s = (RandomizedSearchCV(pipe, grid, n_iter=RF_N_ITER, scoring=SCORING, refit=REFIT_METRIC,
                            cv=cv, random_state=RS, n_jobs=N_JOBS) if kind == "random"
         else GridSearchCV(pipe, grid, scoring=SCORING, refit=REFIT_METRIC, cv=cv, n_jobs=N_JOBS))
    s.fit(xf, yf)
    return s.best_estimator_


def met(ytrue, yp, pr):
    return {"accuracy": accuracy_score(ytrue, yp),
            "precision_macro": precision_score(ytrue, yp, average="macro", zero_division=0),
            "recall_macro": recall_score(ytrue, yp, average="macro", zero_division=0),
            "f1_macro": f1_score(ytrue, yp, average="macro", zero_division=0),
            "roc_auc": roc_auc_score(ytrue, pr)}


full, bal = {}, {}
for sname, tr in TRAIN.items():
    Xtr, ytr = X.iloc[tr], y.iloc[tr]
    for name, est, grid, kind in SPECS:
        best = tune(est, grid, kind, Xtr, ytr)
        full[(sname, name)] = met(yv, best.predict(Xval), best.predict_proba(Xval)[:, 1])
        bal[(sname, name)] = met(yb, best.predict(Xb), best.predict_proba(Xb)[:, 1])
    print(f"  [selesai] {sname}")

pd.set_option("display.width", 200)
for tag, d in [("VALIDATION 50/50", bal), ("VALIDATION ASLI (19/81)", full)]:
    print("\n" + "#" * 92 + f"\n# GAS-ONLY (label baru, model dari train-70%) — {tag}\n" + "#" * 92)
    for sname in TRAIN:
        tbl = pd.DataFrame({m: d[(sname, m)] for m in MODEL_NAMES}).T[COLS]
        print(f"\n--- {sname} ---")
        print(tbl.round(3).to_string())

pd.DataFrame(bal).T.to_csv(config.RESULTS_DIR / "gasonly_validation_5050_newlabel.csv")
pd.DataFrame(full).T.to_csv(config.RESULTS_DIR / "gasonly_validation_full_newlabel.csv")
print("\nTersimpan: results/gasonly_validation_5050_newlabel.csv & gasonly_validation_full_newlabel.csv")
