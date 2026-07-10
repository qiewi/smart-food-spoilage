"""Evaluate the saved dashboard models on validation/, DOUBLY balanced.

Validation = Ayam 0430 + Nasi 0408 (relabeled). Two newer trials were REJECTED and
removed from validation/ after inspection: Ikan 0603 (inverted gas pattern vs every
training trial) and Telur 0601 (recorded with a faulty sensor; gas decreases over time).
The eval subset is balanced on BOTH axes: per food, sample M fresh + M spoiled
(M = min count over every food x class) -> equal share per food and 50:50 fresh/spoiled.
Models = the saved (split x model) pipelines trained on each split's 70% of training/.
Undersampling seeded with RANDOM_STATE.

Output: results/validation_balanced_<n>food.csv + console table
"""

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src import config
config.NUMERIC_FEATURES = ["mq2", "mq135", "mq4"]                 # GAS ONLY
config.FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES

from src.load_data import load_relabeled
from src.preprocess import build_xy, prepare_df

warnings.filterwarnings("ignore")
RS = config.RANDOM_STATE
MODELS_DIR = config.PROJECT_ROOT / "dashboard" / "backend" / "models"
SPLITS = [("Grouped", "grouped"), ("Random Split", "stratifiedkfold")]
MODELS = [("Logistic Regression", "logistic_regression"), ("Decision Tree", "decision_tree"),
          ("KNN", "knn"), ("Random Forest", "random_forest")]

# ---------- balanced subset: equal share per food, 50:50 per class ----------
prep = prepare_df(load_relabeled(config.PROJECT_ROOT / "validation")).reset_index(drop=True)
counts = prep.groupby(["food_type", "label"]).size()
m = int(counts.min())
rng = np.random.default_rng(RS)
keep = []
for (food, lab), n in counts.items():
    idx = prep.index[(prep["food_type"] == food) & (prep["label"] == lab)].to_numpy()
    keep.append(rng.choice(idx, size=m, replace=False) if n > m else idx)
sub = prep.loc[np.sort(np.concatenate(keep))]
Xb, yb, _ = build_xy(sub)
yb = yb.to_numpy()
print(f"Set evaluasi seimbang: {len(yb)} baris | {m} per (komoditas x kelas) | "
      f"fresh {(yb==0).sum()} : spoiled {(yb==1).sum()}")
print(sub.groupby("food_type").size().to_string(), "\n")

# ---------- evaluate the 8 saved pipelines ----------
rows = []
for sname, sslug in SPLITS:
    for mname, mslug in MODELS:
        pipe = joblib.load(MODELS_DIR / f"{sslug}__{mslug}.joblib")
        yp = pipe.predict(Xb)
        proba = pipe.predict_proba(Xb)[:, 1]
        rows.append({
            "split": sname, "model": mname,
            "accuracy": accuracy_score(yb, yp),
            "precision_macro": precision_score(yb, yp, average="macro", zero_division=0),
            "recall_macro": recall_score(yb, yp, average="macro", zero_division=0),
            "f1_macro": f1_score(yb, yp, average="macro", zero_division=0),
            "roc_auc": roc_auc_score(yb, proba),
        })

df = pd.DataFrame(rows)
for sname, _ in SPLITS:
    part = df[df["split"] == sname].drop(columns="split").set_index("model")
    print(f"=== {sname} ===")
    print(part.round(3).to_string(), "\n")

n_food = sub["food_type"].nunique()
out = config.RESULTS_DIR / f"validation_balanced_{n_food}food.csv"
df.round(4).to_csv(out, index=False)
print(f"CSV: results/{out.name}")
