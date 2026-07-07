"""GAS-ONLY variant of the 4 comparison tables (drop humidity & tempC).

Purpose: let the user SEE how the tables change if humidity/tempC are removed
(features = mq2, mq135, mq4 + food_type). Does NOT touch src/config.py on disk or
any existing result file -- it overrides the feature list at runtime only.

All 4 models are tuned per strategy from the cited grids in src.train._model_specs()
(LR = LogisticRegression, max_iter). Same 3 splits as split_strategies.py.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedGroupKFold, StratifiedKFold,
                                     StratifiedShuffleSplit)
from sklearn.pipeline import Pipeline

from src import config

# --- override feature set to GAS-ONLY (runtime only; build_xy/make_preprocessor read these) ---
config.NUMERIC_FEATURES = ["mq2", "mq135", "mq4"]
config.FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES

from src.load_data import load_relabeled
from src.preprocess import build_xy, make_preprocessor, prepare_df
from src.train import REFIT_METRIC, SCORING, _model_specs

warnings.filterwarnings("ignore")
RS = config.RANDOM_STATE
RF_N_ITER = 40
# Full-core: search parallelizes across all cores; each RF stays single-threaded
# (clf__n_jobs=1 below) so cores aren't double-booked.
N_JOBS = -1
METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "mcc", "roc_auc"]
SPECS = _model_specs()                       # all 4 models, cited grids
ORDER = [s[0] for s in SPECS]                # LR, DT, KNN, RF


def macro_metrics(yt, yp, proba):
    return {"accuracy": accuracy_score(yt, yp),
            "precision_macro": precision_score(yt, yp, average="macro", zero_division=0),
            "recall_macro": recall_score(yt, yp, average="macro", zero_division=0),
            "f1_macro": f1_score(yt, yp, average="macro", zero_division=0),
            "mcc": matthews_corrcoef(yt, yp),
            "roc_auc": roc_auc_score(yt, proba)}


# ---------- data (now gas-only via config override) ----------
prepared = prepare_df(load_relabeled(config.PROJECT_ROOT / "training")).reset_index(drop=True)
X, y, groups = build_xy(prepared)
y_arr, run_arr, elapsed = y.to_numpy(), groups.to_numpy(), prepared["elapsed_sec"].to_numpy()
val_df = prepare_df(load_relabeled(config.PROJECT_ROOT / "validation"))
Xval, yval, _ = build_xy(val_df)
print(f"FITUR DIPAKAI: {list(X.columns)}")
print(f"TRAIN {len(X)} baris / {groups.nunique()} run | VALID {len(Xval)} baris\n")


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


def tune(est, grid, kind, cv, grp):
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", clone(est))])
    try:
        pipe.set_params(clf__n_jobs=1)
    except ValueError:
        pass
    if kind == "random":
        s = RandomizedSearchCV(pipe, grid, n_iter=RF_N_ITER, scoring=SCORING,
                               refit=REFIT_METRIC, cv=cv, random_state=RS, n_jobs=N_JOBS)
    else:
        s = GridSearchCV(pipe, grid, scoring=SCORING, refit=REFIT_METRIC, cv=cv, n_jobs=N_JOBS)
    s.fit(X, y, groups=grp) if grp is not None else s.fit(X, y)
    return s.best_estimator_, s.best_params_


def internal_scores(splits, template):
    rows = []
    for tr, te in splits:
        p = clone(template)
        p.fit(X.iloc[tr], y.iloc[tr])
        rows.append(macro_metrics(y.iloc[te], p.predict(X.iloc[te]),
                                  p.predict_proba(X.iloc[te])[:, 1]))
    d = pd.DataFrame(rows)
    return d.mean(), d.std()


pd.set_option("display.width", 200)
val_rows, chosen = {}, {}
internal_tbls = {}
for s, splits in SPLITS.items():
    cv, grp = TUNE_CV[s]
    i_mean = {}
    for name, est, grid, kind in SPECS:
        best, params = tune(est, grid, kind, cv, grp)
        chosen[(s, name)] = {k.replace("clf__", ""): v for k, v in params.items()}
        im, _ = internal_scores(splits, clone(best))
        i_mean[name] = im
        val_rows[(s, name)] = macro_metrics(yval, best.predict(Xval), best.predict_proba(Xval)[:, 1])
    internal_tbls[s] = pd.DataFrame(i_mean).T[METRICS].reindex(ORDER)

titles = {"StratifiedKFold": "TABEL 1 (GAS-ONLY) - TRAINING | RANDOM split (Stratified K-Fold)",
          "MonteCarlo": "TABEL 2 (GAS-ONLY) - TRAINING | MONTE CARLO split",
          "Grouped": "TABEL 3 (GAS-ONLY) - TRAINING | GROUPED split"}
for s in SPLITS:
    print("=" * 92 + f"\n{titles[s]}\n" + "=" * 92)
    print(internal_tbls[s].round(3).to_string())
    internal_tbls[s].to_csv(config.RESULTS_DIR / f"gasonly_{s}_internal.csv")
    print()

val_tbl = pd.DataFrame(val_rows).T[METRICS]
val_tbl.index = pd.MultiIndex.from_tuples(val_tbl.index, names=["split", "model"])
val_tbl = val_tbl.reindex(pd.MultiIndex.from_product([list(SPLITS), ORDER], names=["split", "model"]))
val_tbl.to_csv(config.RESULTS_DIR / "gasonly_validation.csv")
print("=" * 92 + "\nTABEL 4 (GAS-ONLY) - DATA VALIDATION | tiap model x tiap split\n" + "=" * 92)
print(val_tbl.round(3).to_string())

print("\n" + "=" * 92 + "\nParameter terpilih (GAS-ONLY) per strategi\n" + "=" * 92)
for s in SPLITS:
    print(f"\n[{s}]")
    for name in ORDER:
        print(f"  {name:22s}: {chosen[(s, name)]}")
