"""Recompute dashboard metadata 'metrics' on the REBALANCED 50/50 validation (no retrain).

Loads the existing gas-only models, evaluates each on the 50/50 validation (undersample
spoiled, seed=RANDOM_STATE), and overwrites metadata['metrics']. Keeps train_metrics +
best_params. Fixes the misleading imbalanced-accuracy shown on the dashboard cards.
"""

import json
import warnings

import numpy as np
import joblib
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src import config
config.NUMERIC_FEATURES = ["mq2", "mq135", "mq4"]                 # GAS ONLY
config.FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES

from src.load_data import load_relabeled
from src.preprocess import build_xy, prepare_df

warnings.filterwarnings("ignore")
OUT = config.PROJECT_ROOT / "dashboard" / "backend" / "models"
MODEL_SLUG = {"Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree",
              "KNN": "knn", "Random Forest": "random_forest"}
SPLIT_SLUG = {"StratifiedKFold": "stratifiedkfold", "MonteCarlo": "montecarlo",
              "Grouped": "grouped", "Stratified Split per Trial": "stratifiedpertrial"}

meta = json.loads((OUT / "metadata.json").read_text())
val_df = prepare_df(load_relabeled(config.PROJECT_ROOT / "validation"))
Xval, yval, _ = build_xy(val_df)
yv = yval.to_numpy()
rng = np.random.default_rng(config.RANDOM_STATE)
fi, si = np.where(yv == 0)[0], np.where(yv == 1)[0]
keep = np.sort(np.concatenate([fi, rng.choice(si, size=len(fi), replace=False)]))
Xb, yb = Xval.iloc[keep], yv[keep]
print(f"Validation 50/50: {len(yb)} baris (fresh {(yb==0).sum()} / spoiled {(yb==1).sum()})\n")


def met(yt, yp, pr):
    return {"accuracy": float(accuracy_score(yt, yp)),
            "precision_macro": float(precision_score(yt, yp, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(yt, yp, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(yt, pr))}


for split in meta["splits"]:
    for model in meta["models"]:
        pipe = joblib.load(OUT / f"{SPLIT_SLUG[split]}__{MODEL_SLUG[model]}.joblib")
        m = met(yb, pipe.predict(Xb), pipe.predict_proba(Xb)[:, 1])
        meta["metrics"][split][model] = m
        print(f"  {split:26s} {model:22s} acc={m['accuracy']:.3f} auc={m['roc_auc']:.3f}")

meta["metrics_note"] = "validation rebalanced 50/50"
(OUT / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
print("\nmetadata.json diperbarui -> metrics = validation 50/50")
