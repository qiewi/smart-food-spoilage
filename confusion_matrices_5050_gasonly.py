"""GAS-ONLY confusion matrices on the REBALANCED (50/50) validation — 4 models x 2 splits.

Gas-only models have no saved artifact, so we retrain them (train 70% per split, tuned via
inner StratifiedKFold, same as the gas-only validation eval; deterministic -> identical
models), then draw confusion matrices on the 50/50 validation. Labels 0=fresh, 1=spoiled.
Output: results/figures/confusion_matrices_5050_gasonly.png
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
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
SPECS = _model_specs()
MODELS = [s[0] for s in SPECS]
CLASSES = ["fresh", "spoiled"]

prepared = prepare_df(load_relabeled(config.PROJECT_ROOT / "training")).reset_index(drop=True)
X, y, groups = build_xy(prepared)
idx = np.arange(len(X))
val_df = prepare_df(load_relabeled(config.PROJECT_ROOT / "validation"))
Xval, yval, _ = build_xy(val_df)
yv = yval.to_numpy()

rng = np.random.default_rng(RS)
fi, si = np.where(yv == 0)[0], np.where(yv == 1)[0]
keep = np.sort(np.concatenate([fi, rng.choice(si, size=len(fi), replace=False)]))
Xb, yb = Xval.iloc[keep], yv[keep]
print(f"GAS-ONLY | Validation 50/50: {len(yb)} baris (fresh {(yb==0).sum()} / spoiled {(yb==1).sum()})\n")

skf_tr, _ = next(StratifiedShuffleSplit(1, test_size=0.30, random_state=RS).split(X, y))
g_tr, _ = next(StratifiedGroupKFold(3, shuffle=True, random_state=RS).split(X, y, groups))
TRAIN = {"Grouped": g_tr, "Random Split": skf_tr}


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


# train all models per split, keep fitted estimators
fitted = {}
for sname, tr in TRAIN.items():
    for name, est, grid, kind in SPECS:
        fitted[(name, sname)] = tune(est, grid, kind, X.iloc[tr], y.iloc[tr])
    print(f"  [selesai] {sname}")

SPLIT_ORDER = ["Grouped", "Random Split"]
fig, axes = plt.subplots(len(SPLIT_ORDER), len(MODELS), figsize=(14, 7))
for i, sname in enumerate(SPLIT_ORDER):
    for j, mname in enumerate(MODELS):
        pipe = fitted[(mname, sname)]
        yp = pipe.predict(Xb)
        cm = confusion_matrix(yb, yp, labels=[0, 1])
        acc = accuracy_score(yb, yp)
        recm = recall_score(yb, yp, average="macro", zero_division=0)
        ax = axes[i, j]
        ax.imshow(cm, cmap="Greens", vmin=0, vmax=cm.max())
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, f"{v}", ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if v > cm.max() * 0.6 else "#111")
        ax.set_xticks([0, 1]); ax.set_xticklabels(CLASSES, fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(CLASSES, fontsize=8)
        ax.set_xlabel(f"Prediksi\nacc={acc:.2f}  rec-macro={recm:.2f}", fontsize=8)
        if i == 0:
            ax.set_title(mname, fontsize=11, fontweight="bold", pad=10)
        if j == 0:
            ax.set_ylabel(f"{sname}\n\nAktual", fontsize=9, fontweight="bold")

fig.suptitle("Confusion Matrix — validation REBALANCED 50/50 (GAS-ONLY) — 2 split × 4 model",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(config.FIGURES_DIR / "confusion_matrices_5050_gasonly.png", dpi=150)
print("\nFigure: results/figures/confusion_matrices_5050_gasonly.png")
