"""Latih & simpan 8 pipeline (split x model) GAS-ONLY Nasi Putih + metadata.json untuk dashboard.

Fitur gas-only [mq2, mq135, mq4] — TANPA food_type/humidity/tempC (single-food). MODEL DEPLOY:
tiap (split, model) dilatih pada **SELURUH data training** (bukan 70%), tuned via CV split-nya
sendiri lalu di-refit pada 100% data → satu model tunggal yang benar-benar dipakai. "Split" di
sini menentukan **CV tuning**: Grouped = StratifiedGroupKFold (jujur, per-trial), Random Split =
StratifiedKFold (baris acak, rawan bocor pada pemilihan hyperparameter). Refit F1. Dievaluasi
pada validasi REBALANCED 50/50. train_metrics = skor inner-CV terbaik (chart train-vs-val).

Catatan kejujuran: angka di sini = performa **model akhir (deployed)**, berbeda dari
results/tabel_validation_test.md yang memakai **evaluasi metodologi** (ensemble rotasi 3-fold).
Dua tujuan berbeda: tabel = perbandingan metodologi/kebocoran; dashboard = model yang dipakai.

Mengikuti pola export_models.py (root) tetapi memakai modul gas-only analisis-nasi-putih/src.
Output: analisis-nasi-putih/dashboard/backend/models/{split}__{model}.joblib + metadata.json

Jalankan:  cd analisis-nasi-putih && python export_dashboard_models.py
"""

import json
import warnings

import joblib
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedGroupKFold, StratifiedKFold)
from sklearn.pipeline import Pipeline

from src import config, data, evaluate
from src.models import model_specs

warnings.filterwarnings("ignore")
RS, RF_N_ITER, N_JOBS = config.RANDOM_STATE, config.RF_N_ITER, -1
OUT = config.BASE / "dashboard" / "backend" / "models"
OUT.mkdir(parents=True, exist_ok=True)

SPECS = model_specs()
MODEL_SLUG = {"Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree",
              "KNN": "knn", "Random Forest": "random_forest"}
SPLIT_SLUG = {"Grouped": "grouped", "Random Split": "stratifiedkfold"}
# "Split" = strategi CV untuk pemilihan hyperparameter (model tetap di-refit pada seluruh data).
SEARCH_CV = {"Grouped": StratifiedGroupKFold(config.N_SPLITS, shuffle=True, random_state=RS),
             "Random Split": StratifiedKFold(config.N_SPLITS, shuffle=True, random_state=RS)}


def val_metrics(yt, yp, proba):              # no MCC
    return {"accuracy": float(accuracy_score(yt, yp)),
            "precision_macro": float(precision_score(yt, yp, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(yt, yp, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(yt, proba))}


# ---------- data (gas-only, cleaning + label encode) ----------
tr = data.encode_labels(data.clean(data.load(config.TRAIN_DIR)))
X, y, groups = data.build_xy(tr)
val = data.encode_labels(data.clean(data.load(config.VAL_DIR)))
Xval, yval, _ = data.build_xy(val)
Xb, yb = evaluate.rebalance_5050(Xval, yval.to_numpy())      # validasi 50/50 (650:650)
print(f"Fitur (GAS-ONLY): {config.FEATURES} | makanan: ['Nasi Putih'] | "
      f"latih: {len(y)} baris (SELURUH data) | validasi 50/50: {len(yb)} baris\n")


def tune(est, grid, kind, cv, gp):
    """Tune via CV split-nya pada SELURUH data, refit pada seluruh data (best_estimator_)."""
    pipe = Pipeline([("prep", data.make_preprocessor()), ("clf", clone(est))])
    try:
        pipe.set_params(clf__n_jobs=1)
    except ValueError:
        pass
    s = (RandomizedSearchCV(pipe, grid, n_iter=RF_N_ITER, scoring=config.SCORING,
                            refit=config.REFIT_METRIC, cv=cv, random_state=RS, n_jobs=N_JOBS)
         if kind == "random"
         else GridSearchCV(pipe, grid, scoring=config.SCORING, refit=config.REFIT_METRIC,
                           cv=cv, n_jobs=N_JOBS))
    s.fit(X, y, groups=gp)
    cvr, bi = s.cv_results_, s.best_index_
    train_m = {"accuracy": float(cvr["mean_test_accuracy"][bi]),
               "precision_macro": float(cvr["mean_test_precision"][bi]),
               "recall_macro": float(cvr["mean_test_recall"][bi]),
               "f1_macro": float(cvr["mean_test_f1"][bi]),
               "roc_auc": float(cvr["mean_test_roc_auc"][bi])}
    return s.best_estimator_, s.best_params_, train_m


meta = {"splits": list(SEARCH_CV), "models": [s[0] for s in SPECS], "food_types": ["Nasi Putih"],
        "features": config.FEATURES, "label_map": config.LABEL_MAP,
        "metrics_note": ("gas-only Nasi Putih; MODEL DEPLOY dilatih pada SELURUH data training, "
                         "tuning via CV split-nya; validasi rebalanced 50/50 (trial 0407)"),
        "metrics": {}, "train_metrics": {}, "best_params": {}}

for split, cv in SEARCH_CV.items():
    meta["metrics"][split], meta["train_metrics"][split], meta["best_params"][split] = {}, {}, {}
    gp = groups if split == "Grouped" else None
    for name, est, grid, kind in SPECS:
        best, raw, train_m = tune(est, grid, kind, cv, gp)
        joblib.dump(best, OUT / f"{SPLIT_SLUG[split]}__{MODEL_SLUG[name]}.joblib")
        m = val_metrics(yb, best.predict(Xb), best.predict_proba(Xb)[:, 1])
        meta["metrics"][split][name] = m
        meta["train_metrics"][split][name] = train_m
        meta["best_params"][split][name] = {k.replace("clf__", ""): v for k, v in raw.items()}
        print(f"  [{split:20s}] {name:22s} val50/50 acc={m['accuracy']:.3f} auc={m['roc_auc']:.3f}")

(OUT / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
print(f"\n{len(SEARCH_CV) * len(SPECS)} model GAS-ONLY Nasi Putih + metadata.json di {OUT}")
