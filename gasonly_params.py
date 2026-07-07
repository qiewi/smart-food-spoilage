"""Print the tuned hyperparameters (best_params_) for the GAS-ONLY models used in the
validation-50/50 evaluation: tuned on each split's 70% train (new labels), cited grids.
Deterministic (same seed) -> identical to gasonly_internal_to_validation.py's models.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
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

prepared = prepare_df(load_relabeled(config.PROJECT_ROOT / "training")).reset_index(drop=True)
X, y, groups = build_xy(prepared)
idx = np.arange(len(X))

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


def best_params(est, grid, kind, xf, yf):
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
    return {k.replace("clf__", ""): v for k, v in s.best_params_.items()}


print("PARAMETER TERPILIH (GAS-ONLY, tuned di 70% train tiap split) — label baru\n")
for sname, tr in TRAIN.items():
    print("=" * 88)
    print(f"[{sname}]")
    print("=" * 88)
    for name, est, grid, kind in SPECS:
        bp = best_params(est, grid, kind, X.iloc[tr], y.iloc[tr])
        print(f"  {name:22s}: {bp}")
    print()
