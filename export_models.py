"""Train & persist the (split x model) pipelines + metadata for the dashboard — GAS-ONLY.

Matches the report tables (gasonly_internal_to_validation.py): every split trains on its
70% train partition and is tuned via an inner StratifiedKFold, then evaluated on the
REBALANCED 50/50 validation. 2 splits: Grouped (honest, per-run) and Random Split (random
rows, leaky). Features = gas only (mq2, mq135, mq4 + food_type) — environmental features excluded
(entangled with the controlled-temperature labeling; not deployable). No MCC.

train_metrics = internal inner-CV score (from GridSearch) for the dashboard train-vs-val chart.
Output: dashboard/backend/models/{split}__{model}.joblib  +  metadata.json
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedGroupKFold, StratifiedKFold,
                                     StratifiedShuffleSplit)
from sklearn.pipeline import Pipeline

from src import config
config.NUMERIC_FEATURES = ["mq2", "mq135", "mq4"]                 # GAS ONLY
config.FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES

from src.load_data import load_relabeled
from src.preprocess import build_xy, make_preprocessor, prepare_df
from src.train import REFIT_METRIC, SCORING, _model_specs

warnings.filterwarnings("ignore")
RS, RF_N_ITER, N_JOBS = config.RANDOM_STATE, 40, -1
OUT = config.PROJECT_ROOT / "dashboard" / "backend" / "models"
OUT.mkdir(parents=True, exist_ok=True)

SPECS = _model_specs()
MODEL_SLUG = {"Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree",
              "KNN": "knn", "Random Forest": "random_forest"}
SPLIT_SLUG = {"Grouped": "grouped", "Random Split": "stratifiedkfold"}


def val_metrics(yt, yp, proba):              # no MCC
    return {"accuracy": float(accuracy_score(yt, yp)),
            "precision_macro": float(precision_score(yt, yp, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(yt, yp, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(yt, proba))}


# ---------- data ----------
prepared = prepare_df(load_relabeled(config.PROJECT_ROOT / "training")).reset_index(drop=True)
X, y, groups = build_xy(prepared)
val_df = prepare_df(load_relabeled(config.PROJECT_ROOT / "validation"))
Xval, yval, _ = build_xy(val_df)
yv = yval.to_numpy()
foods = sorted(prepared["food_type"].unique().tolist())

# rebalanced 50/50 validation
rng = np.random.default_rng(RS)
fi, si = np.where(yv == 0)[0], np.where(yv == 1)[0]
keep = np.sort(np.concatenate([fi, rng.choice(si, size=len(fi), replace=False)]))
Xb, yb = Xval.iloc[keep], yv[keep]

# 70% train indices per split (same as the report tables)
skf_tr, _ = next(StratifiedShuffleSplit(1, test_size=0.30, random_state=RS).split(X, y))
g_tr, _ = next(StratifiedGroupKFold(3, shuffle=True, random_state=RS).split(X, y, groups))
TRAIN = {"Grouped": g_tr, "Random Split": skf_tr}
print(f"Fitur (GAS-ONLY): {config.FEATURES} | makanan: {foods} | validasi 50/50: {len(yb)} baris\n")


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
    cvr, bi = s.cv_results_, s.best_index_
    train_m = {"accuracy": float(cvr["mean_test_accuracy"][bi]),
               "precision_macro": float(cvr["mean_test_precision"][bi]),
               "recall_macro": float(cvr["mean_test_recall"][bi]),
               "f1_macro": float(cvr["mean_test_f1"][bi]),
               "roc_auc": float(cvr["mean_test_roc_auc"][bi])}
    return s.best_estimator_, s.best_params_, train_m


meta = {"splits": list(TRAIN), "models": [s[0] for s in SPECS], "food_types": foods,
        "features": config.FEATURES, "label_map": config.LABEL_MAP,
        "metrics_note": "validation rebalanced 50/50; models trained on each split's 70%",
        "metrics": {}, "train_metrics": {}, "best_params": {}}

for split, tr in TRAIN.items():
    meta["metrics"][split], meta["train_metrics"][split], meta["best_params"][split] = {}, {}, {}
    for name, est, grid, kind in SPECS:
        best, raw, train_m = tune(est, grid, kind, X.iloc[tr], y.iloc[tr])
        joblib.dump(best, OUT / f"{SPLIT_SLUG[split]}__{MODEL_SLUG[name]}.joblib")
        m = val_metrics(yb, best.predict(Xb), best.predict_proba(Xb)[:, 1])
        meta["metrics"][split][name] = m
        meta["train_metrics"][split][name] = train_m
        meta["best_params"][split][name] = {k.replace("clf__", ""): v for k, v in raw.items()}
        print(f"  [{split:20s}] {name:22s} val50/50 acc={m['accuracy']:.3f} auc={m['roc_auc']:.3f}")

(OUT / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
print(f"\n{len(TRAIN) * len(SPECS)} model GAS-ONLY (match tabel laporan) + metadata.json di {OUT}")
