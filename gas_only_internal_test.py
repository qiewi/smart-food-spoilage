"""GAS-ONLY internal test: train on the split's 70%, evaluate on its own 30% test.

Three 70/30 splits of training/ (NOT validation): Stratified K-Fold (stratified rows),
Grouped (whole runs, test trimmed trailing to 30% keeping both classes), Stratified per
Trial (stratified by label x trial). Each model tuned (gas-only, cited grids) on the 70%
train, evaluated on the 30% internal test. Metrics macro (recall = overall, not spoiled-only).
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
y_arr, run_arr, elapsed = y.to_numpy(), groups.to_numpy(), prepared["elapsed_sec"].to_numpy()
idx = np.arange(len(X))
print(f"Fitur (GAS-ONLY): {config.FEATURES}\n")

# ---- 70/30 train/test per split ----
skf_tr, skf_te = next(StratifiedShuffleSplit(1, test_size=0.30, random_state=RS).split(X, y))

strat = (y.astype(str) + "_" + groups.astype(str)).to_numpy()
vc = pd.Series(strat).value_counts()
badk = vc[vc < 2].index.tolist()
if badk:
    mm = np.isin(strat, badk)
    strat[mm] = y.astype(str).to_numpy()[mm]
pt_tr, pt_te = train_test_split(idx, test_size=0.30, random_state=RS, stratify=strat)

g_tr, g_te = next(StratifiedGroupKFold(3, shuffle=True, random_state=RS).split(X, y, groups))
target = int(round((0.30 / 0.70) * len(g_tr)))
kf = min(1.0, target / len(g_te))
kept = []
for r in np.unique(run_arr[g_te]):
    rows = g_te[run_arr[g_te] == r]
    rows = rows[np.argsort(elapsed[rows])]
    kept.extend(rows[:max(1, int(round(len(rows) * kf)))])   # keep beginning -> both classes
g_te = np.array(sorted(kept))

SPLITS = {"Grouped": (g_tr, g_te),
          "Stratified K-Fold": (skf_tr, skf_te),
          "Stratified per Trial": (pt_tr, pt_te)}

print("Komposisi TEST internal (fresh/spoiled):")
for s, (tr, te) in SPLITS.items():
    yte = y_arr[te]
    print(f"  {s:22s} train {len(tr):5d} / test {len(te):5d}  "
          f"(test fresh {int((yte==0).sum())} / spoiled {int((yte==1).sum())})")
print()


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


res = {}
for sname, (tr, te) in SPLITS.items():
    Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
    for name, est, grid, kind in SPECS:
        best = tune(est, grid, kind, Xtr, ytr)
        res[(sname, name)] = met(yte, best.predict(Xte), best.predict_proba(Xte)[:, 1])
    print(f"  [selesai] {sname}")

pd.set_option("display.width", 200)
print("\n" + "#" * 92 + "\n# GAS-ONLY — UJI INTERNAL (train 70% -> test 30% dari data training)\n" + "#" * 92)
for sname in SPLITS:
    tbl = pd.DataFrame({m: res[(sname, m)] for m in MODEL_NAMES}).T[COLS]
    print(f"\n--- {sname} ---")
    print(tbl.round(3).to_string())

pd.DataFrame(res).T.to_csv(config.RESULTS_DIR / "gasonly_internal_test_tables.csv")
print("\nTersimpan: results/gasonly_internal_test_tables.csv")
