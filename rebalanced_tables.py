"""Full metric tables on the REBALANCED (50/50) validation, per split x model.

Undersamples spoiled to match fresh (seed = RANDOM_STATE), then predicts with each saved
model and reports the same metric suite as the original validation tables (no MCC):
accuracy, precision_macro, recall_macro, f1_macro, recall_spoiled, roc_auc.
"""

import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src import config
from src.load_data import load_relabeled
from src.preprocess import build_xy, prepare_df

warnings.filterwarnings("ignore")
MODELS_DIR = config.PROJECT_ROOT / "dashboard" / "backend" / "models"
SPLITS = [("Grouped", "grouped"), ("Stratified K-Fold", "stratifiedkfold"),
          ("Stratified per Trial", "stratifiedpertrial")]
MODELS = [("Logistic Regression", "logistic_regression"), ("Decision Tree", "decision_tree"),
          ("KNN", "knn"), ("Random Forest", "random_forest")]
COLS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "recall_spoiled", "roc_auc"]

val_df = prepare_df(load_relabeled(config.PROJECT_ROOT / "validation"))
Xval, yval, _ = build_xy(val_df)
yv = yval.to_numpy()

rng = np.random.default_rng(config.RANDOM_STATE)
fresh_i, spoil_i = np.where(yv == 0)[0], np.where(yv == 1)[0]
keep = np.sort(np.concatenate([fresh_i, rng.choice(spoil_i, size=len(fresh_i), replace=False)]))
Xb, yb = Xval.iloc[keep], yv[keep]
print(f"Validation REBALANCED: {len(yb)} baris (fresh {(yb==0).sum()} / spoiled {(yb==1).sum()} = 50/50)\n")


def metrics(ytrue, yp, pr):
    return {"accuracy": accuracy_score(ytrue, yp),
            "precision_macro": precision_score(ytrue, yp, average="macro", zero_division=0),
            "recall_macro": recall_score(ytrue, yp, average="macro", zero_division=0),
            "f1_macro": f1_score(ytrue, yp, average="macro", zero_division=0),
            "recall_spoiled": recall_score(ytrue, yp, pos_label=1, zero_division=0),
            "roc_auc": roc_auc_score(ytrue, pr)}


pd.set_option("display.width", 200)
combined = {}
for sname, sslug in SPLITS:
    rows = {}
    for mname, mslug in MODELS:
        pipe = joblib.load(MODELS_DIR / f"{sslug}__{mslug}.joblib")
        rows[mname] = metrics(yb, pipe.predict(Xb), pipe.predict_proba(Xb)[:, 1])
        combined[(sname, mname)] = rows[mname]
    tbl = pd.DataFrame(rows).T[COLS]
    print("=" * 92)
    print(f"REBALANCED 50/50 — {sname}")
    print("=" * 92)
    print(tbl.round(3).to_string())
    print()

full = pd.DataFrame(combined).T[COLS]
full.index = pd.MultiIndex.from_tuples(full.index, names=["split", "model"])
full.to_csv(config.RESULTS_DIR / "rebalanced_validation_tables.csv")
print("Tersimpan: results/rebalanced_validation_tables.csv")
