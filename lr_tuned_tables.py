"""Rebuild the 4 comparison tables with LR = tuned LogisticRegression (max_iter, NOT epoch).

Only Logistic Regression is recomputed; DT/KNN/RF rows are reused from the saved
split_strategy_*.csv (already tuned per strategy in the previous run) -> fast & cool.

LR uses the cited grid (Hasan 2020 + sklearn): C, penalty, class_weight, solver=liblinear.
The split's CV drives the search, exactly like DT/KNN/RF.

Tables:
  1. Training (internal) comparison - RANDOM split  (Stratified K-Fold)
  2. Training (internal) comparison - MONTE CARLO split
  3. Training (internal) comparison - GROUPED split
  4. VALIDATION comparison - every model x every split (12 rows)
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold, StratifiedShuffleSplit)
from sklearn.pipeline import Pipeline

from src import config
from src.load_data import load_relabeled
from src.preprocess import build_xy, make_preprocessor, prepare_df
from src.train import REFIT_METRIC, SCORING

warnings.filterwarnings("ignore")
RS = config.RANDOM_STATE
METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "mcc", "roc_auc"]
OLD_LR = "Logistic Regression (SGD, 5 epoch)"
NEW_LR = "Logistic Regression"
ORDER = [NEW_LR, "Decision Tree", "KNN", "Random Forest"]

# Cited LR grid (same as src.train._model_specs); max_iter-based convex solver, no epochs.
LR_EST = LogisticRegression(solver="liblinear", max_iter=1000)
LR_GRID = {"clf__C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
           "clf__penalty": ["l1", "l2"],
           "clf__class_weight": [None, "balanced"]}


def macro_metrics(yt, yp, proba):
    return {"accuracy": accuracy_score(yt, yp),
            "precision_macro": precision_score(yt, yp, average="macro", zero_division=0),
            "recall_macro": recall_score(yt, yp, average="macro", zero_division=0),
            "f1_macro": f1_score(yt, yp, average="macro", zero_division=0),
            "mcc": matthews_corrcoef(yt, yp),
            "roc_auc": roc_auc_score(yt, proba)}


# ---------- data ----------
prepared = prepare_df(load_relabeled(config.PROJECT_ROOT / "training")).reset_index(drop=True)
X, y, groups = build_xy(prepared)
y_arr, run_arr, elapsed = y.to_numpy(), groups.to_numpy(), prepared["elapsed_sec"].to_numpy()
val_df = prepare_df(load_relabeled(config.PROJECT_ROOT / "validation"))
Xval, yval, _ = build_xy(val_df)


def grouped_trimmed_splits():
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RS)
    out = []
    for tr, te in sgkf.split(X, y, groups):
        target = (0.30 / 0.70) * len(tr)
        keep = min(1.0, target / len(te))
        kept = []
        for r in np.unique(run_arr[te]):
            rows = te[run_arr[te] == r]
            rows = rows[np.argsort(elapsed[rows])]
            kept.extend(rows[:max(1, int(round(len(rows) * keep)))])
        out.append((tr, np.array(sorted(kept))))
    return out


SPLITS = {
    "StratifiedKFold": list(StratifiedKFold(3, shuffle=True, random_state=RS).split(X, y)),
    "MonteCarlo": list(StratifiedShuffleSplit(20, test_size=0.3, random_state=RS).split(X, y)),
    "Grouped": grouped_trimmed_splits(),
}
TUNE_CV = {
    "StratifiedKFold": (StratifiedKFold(3, shuffle=True, random_state=RS), None),
    "MonteCarlo": (StratifiedShuffleSplit(5, test_size=0.3, random_state=RS), None),
    "Grouped": (StratifiedGroupKFold(3, shuffle=True, random_state=RS), groups),
}
CSV = {"StratifiedKFold": "split_strategy_1_StratifiedKFold.csv",
       "MonteCarlo": "split_strategy_2_MonteCarloCV.csv",
       "Grouped": "split_strategy_3_GroupedRotation.csv"}


# ---------- tune LR + evaluate per strategy ----------
lr_internal, lr_valid, lr_params = {}, {}, {}
for s, splits in SPLITS.items():
    cv, grp = TUNE_CV[s]
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", clone(LR_EST))])
    gs = GridSearchCV(pipe, LR_GRID, scoring=SCORING, refit=REFIT_METRIC, cv=cv, n_jobs=4)
    gs.fit(X, y, groups=grp) if grp is not None else gs.fit(X, y)
    best = gs.best_estimator_
    lr_params[s] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    rows = []
    for tr, te in splits:
        p = clone(best)
        p.fit(X.iloc[tr], y.iloc[tr])
        rows.append(macro_metrics(y.iloc[te], p.predict(X.iloc[te]),
                                  p.predict_proba(X.iloc[te])[:, 1]))
    lr_internal[s] = pd.DataFrame(rows).mean()
    lr_valid[s] = macro_metrics(yval, best.predict(Xval), best.predict_proba(Xval)[:, 1])


# ---------- assemble tables: reuse DT/KNN/RF from CSV, swap LR row ----------
def rebuild_internal(s):
    df = pd.read_csv(config.RESULTS_DIR / CSV[s], index_col=0).drop(index=OLD_LR)
    df.loc[NEW_LR] = lr_internal[s]
    return df.reindex(ORDER)[METRICS]


internal = {s: rebuild_internal(s) for s in SPLITS}
val_df_csv = pd.read_csv(config.RESULTS_DIR / "split_strategy_validation.csv",
                         index_col=[0, 1])
val_df_csv = val_df_csv[val_df_csv.index.get_level_values("model") != OLD_LR]
for s in SPLITS:
    val_df_csv.loc[(s, NEW_LR), :] = lr_valid[s]
val_tbl = val_df_csv.reindex(pd.MultiIndex.from_product([list(SPLITS), ORDER],
                             names=["split", "model"]))[METRICS]

# ---------- print + save ----------
pd.set_option("display.width", 200)
titles = {"StratifiedKFold": "TABEL 1 - Perbandingan performa TRAINING antar model | RANDOM split (Stratified K-Fold)",
          "MonteCarlo": "TABEL 2 - Perbandingan performa TRAINING antar model | MONTE CARLO split",
          "Grouped": "TABEL 3 - Perbandingan performa TRAINING antar model | GROUPED split"}
for s in SPLITS:
    print("=" * 96 + f"\n{titles[s]}\n" + "=" * 96)
    print(internal[s].round(3).to_string())
    internal[s].to_csv(config.RESULTS_DIR / f"tbl_{s}_internal.csv")
    print()

print("=" * 96 + "\nTABEL 4 - Perbandingan performa antar model dari tiap split | DATA VALIDATION\n" + "=" * 96)
print(val_tbl.round(3).to_string())
val_tbl.to_csv(config.RESULTS_DIR / "tbl_validation.csv")

print("\n" + "=" * 96 + "\nParameter LR terpilih per strategi (grid bersitasi, berbasis max_iter - bukan epoch)\n" + "=" * 96)
for s in SPLITS:
    print(f"  {s:16s}: {lr_params[s]}")
