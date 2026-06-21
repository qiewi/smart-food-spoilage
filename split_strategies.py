"""Compare three data-splitting strategies (TUNED per strategy) + test on validation/.

Strategies (each used both to TUNE hyperparameters and to report the internal score):
  1. Stratified K-Fold        : StratifiedKFold(3) - rows shuffled, exclusive folds.
  2. Monte Carlo CV           : StratifiedShuffleSplit(20, test=0.3) - repeated random
                                70/30 -> internal mean +/- std (instability).
  3. Grouped split + rotation : StratifiedGroupKFold(3); each held-out test run trimmed
                                temporal-contiguous (drop trailing/spoiled rows) so test
                                ~30% AND both classes remain.

DT / KNN / RF are tuned per strategy with the CITED grids in src.train._model_specs()
(Hasan 2020; Sarno 2023; arXiv:2310.14629 + scikit-learn). The split's CV drives the
search, so a leaky (row-based) CV selects OVERFIT params (deep trees, small k) that then
collapse on validation -- the real danger of leakage. Logistic Regression = SGDClassifier
(log loss) trained 5 epochs: a linear baseline that cannot memorize (kept fixed).

Validation table: 4 models x 3 strategies = 12 rows. Learning curve for the SGD-LR.
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedGroupKFold, StratifiedKFold,
                                     StratifiedShuffleSplit)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

from src import config
from src.load_data import load_relabeled
from src.preprocess import build_xy, make_preprocessor, prepare_df
from src.train import REFIT_METRIC, SCORING, _model_specs

warnings.filterwarnings("ignore")
RS = config.RANDOM_STATE
N_EPOCHS = 5
RF_N_ITER = 40
# Cap parallelism to avoid maxing every core (the laptop overheated at n_jobs=-1).
# 4 candidates in parallel, each RF single-threaded -> ~4 cores busy, much cooler.
N_JOBS = 4
METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "mcc", "roc_auc"]
LR_NAME = "Logistic Regression (SGD, 5 epoch)"
# cited grids for the tunable (memorizing) models; LR handled separately as SGD.
SPECS = {n: (e, g, k) for n, e, g, k in _model_specs() if n != "Logistic Regression"}
ORDER = [LR_NAME, "Decision Tree", "KNN", "Random Forest"]


def sgd_lr():
    return SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                         class_weight="balanced", max_iter=N_EPOCHS, tol=None,
                         random_state=RS)


def macro_metrics(yt, yp, proba):
    return {
        "accuracy": accuracy_score(yt, yp),
        "precision_macro": precision_score(yt, yp, average="macro", zero_division=0),
        "recall_macro": recall_score(yt, yp, average="macro", zero_division=0),
        "f1_macro": f1_score(yt, yp, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(yt, yp),
        "roc_auc": roc_auc_score(yt, proba),
    }


# ---------- load data ----------
prepared = prepare_df(load_relabeled(config.PROJECT_ROOT / "training")).reset_index(drop=True)
X, y, groups = build_xy(prepared)
y_arr, run_arr, elapsed = y.to_numpy(), groups.to_numpy(), prepared["elapsed_sec"].to_numpy()
N = len(X)
val_df = prepare_df(load_relabeled(config.PROJECT_ROOT / "validation"))
Xval, yval, _ = build_xy(val_df)
print(f"TRAIN: {N} baris / {groups.nunique()} run | VALID: {len(Xval)} baris "
      f"({sorted(val_df.food_type.unique())})\n")


# ---------- split generators (positional indices) for the INTERNAL score ----------
def grouped_trimmed_splits():
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RS)
    splits = []
    for tr, te in sgkf.split(X, y, groups):
        target = (0.30 / 0.70) * len(tr)
        keep_frac = min(1.0, target / len(te))
        kept = []
        for r in np.unique(run_arr[te]):
            rows = te[run_arr[te] == r]
            rows = rows[np.argsort(elapsed[rows])]
            k = max(1, int(round(len(rows) * keep_frac)))
            kept.extend(rows[:k])
        te_trim = np.array(sorted(kept))
        assert set(y_arr[te_trim]) == {0, 1}, "grouped test fold lost a class!"
        splits.append((tr, te_trim))
    return splits


STRATEGIES = {
    "1_StratifiedKFold": list(StratifiedKFold(3, shuffle=True, random_state=RS).split(X, y)),
    "2_MonteCarloCV": list(StratifiedShuffleSplit(20, test_size=0.3, random_state=RS).split(X, y)),
    "3_GroupedRotation": grouped_trimmed_splits(),
}
# CV used to TUNE under each strategy (groups required only for the grouped one).
TUNE_CV = {
    "1_StratifiedKFold": (StratifiedKFold(3, shuffle=True, random_state=RS), None),
    "2_MonteCarloCV": (StratifiedShuffleSplit(5, test_size=0.3, random_state=RS), None),
    "3_GroupedRotation": (StratifiedGroupKFold(3, shuffle=True, random_state=RS), groups),
}
LABELS = {"1_StratifiedKFold": "Stratified K-Fold (k=3)",
          "2_MonteCarloCV": "Monte Carlo CV (20x, 70/30) - mean +/- std",
          "3_GroupedRotation": "Grouped split + rotation (70/30, trailing-trim)"}
SHORT = {"1_StratifiedKFold": "StratifiedKFold", "2_MonteCarloCV": "MonteCarlo",
         "3_GroupedRotation": "Grouped"}


def tune(est, grid, kind, cv, grp):
    """Tune via the strategy CV using the cited grid. Returns best_estimator_, best_params_."""
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", clone(est))])
    try:  # force single-threaded base estimator (RF/KNN) to avoid core oversubscription
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
    """Re-evaluate the tuned params under the strategy's own splits (train fold -> test fold)."""
    rows = []
    for tr, te in splits:
        pipe = clone(template)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        rows.append(macro_metrics(y.iloc[te], pipe.predict(X.iloc[te]),
                                  pipe.predict_proba(X.iloc[te])[:, 1]))
    d = pd.DataFrame(rows)
    return d.mean(), d.std()


# ---------- main: tune per strategy, score internal + validation ----------
pd.set_option("display.width", 200)
val_rows, chosen = {}, {}
for sname, splits in STRATEGIES.items():
    cv, grp = TUNE_CV[sname]
    i_mean, i_std = {}, {}
    for mname in ORDER:
        if mname == LR_NAME:
            best_full = Pipeline([("prep", make_preprocessor()), ("clf", sgd_lr())]).fit(X, y)
            template = Pipeline([("prep", make_preprocessor()), ("clf", sgd_lr())])
            chosen[(SHORT[sname], mname)] = "fixed: SGD log_loss, 5 epoch, alpha=1e-4"
        else:
            est, grid, kind = SPECS[mname]
            best_full, params = tune(est, grid, kind, cv, grp)
            template = clone(best_full)
            chosen[(SHORT[sname], mname)] = {k.replace("clf__", ""): v for k, v in params.items()}
        im, istd = internal_scores(splits, template)
        i_mean[mname], i_std[mname] = im, istd
        val_rows[(SHORT[sname], mname)] = macro_metrics(
            yval, best_full.predict(Xval), best_full.predict_proba(Xval)[:, 1])

    itbl = pd.DataFrame(i_mean).T[METRICS]
    istdt = pd.DataFrame(i_std).T[METRICS]
    itbl.to_csv(config.RESULTS_DIR / f"split_strategy_{sname}.csv")
    print("=" * 96)
    print(f"STRATEGI: {LABELS[sname]}  | n_split={len(splits)}  (internal test, TUNED)")
    print("=" * 96)
    if sname == "2_MonteCarloCV":
        print((itbl.round(3).astype(str) + " ± " + istdt.round(3).astype(str)).to_string())
    else:
        print(itbl.round(3).to_string())
    print()

# ---------- 12-row validation table ----------
val_tbl = pd.DataFrame(val_rows).T[METRICS]
val_tbl.index = pd.MultiIndex.from_tuples(val_tbl.index, names=["split", "model"])
val_tbl.to_csv(config.RESULTS_DIR / "split_strategy_validation.csv")
print("=" * 96)
print("TES AKHIR pada DATA VALIDATION (Ayam + Nasi Putih) - 4 model x 3 split = 12 baris")
print("(tiap model di-tune via CV strateginya, lalu dilatih ulang di SELURUH training/)")
print("=" * 96)
print(val_tbl.round(3).to_string())

# ---------- hyperparameter chosen per strategy (the leakage evidence) ----------
print("\n" + "=" * 96)
print("PARAMETER TERPILIH per strategi (dari grid bersitasi) - bukti CV acak memilih model overfit")
print("=" * 96)
for sname in STRATEGIES:
    print(f"\n[{SHORT[sname]}]")
    for mname in ORDER:
        print(f"  {mname:34s}: {chosen[(SHORT[sname], mname)]}")

# ---------- SGD-LR learning curve over 5 epochs ----------
prep = make_preprocessor()
Xtr_t, Xval_t = prep.fit_transform(X), prep.transform(Xval)
_cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_arr)
sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                    class_weight={0: _cw[0], 1: _cw[1]}, random_state=RS)
rng = np.random.default_rng(RS)
tr_acc, va_acc = [], []
for ep in range(1, N_EPOCHS + 1):
    order = rng.permutation(len(y_arr))
    sgd.partial_fit(Xtr_t[order], y_arr[order], classes=np.array([0, 1]))
    tr_acc.append(accuracy_score(y_arr, sgd.predict(Xtr_t)))
    va_acc.append(accuracy_score(yval.to_numpy(), sgd.predict(Xval_t)))

fig, ax = plt.subplots(figsize=(7, 4.5))
ep = range(1, N_EPOCHS + 1)
ax.plot(ep, tr_acc, "o-", label="Train accuracy")
ax.plot(ep, va_acc, "s-", label="Validation accuracy")
ax.set_xticks(list(ep))
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.set_title("Learning Curve - Logistic Regression (SGD), 5 epoch", fontweight="bold")
ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "learning_curve_sgd_lr.png", dpi=150)
print(f"\nLearning curve: results/figures/learning_curve_sgd_lr.png")
print(f"  train acc/epoch: {[round(a,3) for a in tr_acc]}")
print(f"  val   acc/epoch: {[round(a,3) for a in va_acc]}")
