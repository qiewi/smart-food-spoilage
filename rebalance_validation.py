"""Sanity check: rebalance validation/ to 50/50 (undersample spoiled) and re-evaluate.

Point: prevalence-invariant metrics (recall_macro, ROC-AUC) should barely change if our
conclusions are real; accuracy will shift (it was inflated by the 81% spoiled majority).
Uses the saved dashboard models. Splits: Grouped, Stratified K-Fold, Stratified per Trial.
"""

import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, f1_score, recall_score, roc_auc_score)

from src import config
from src.load_data import load_relabeled
from src.preprocess import build_xy, prepare_df

warnings.filterwarnings("ignore")
MODELS_DIR = config.PROJECT_ROOT / "dashboard" / "backend" / "models"
SPLITS = [("Grouped", "grouped"), ("Stratified K-Fold", "stratifiedkfold"),
          ("Stratified per Trial", "stratifiedpertrial")]
MODELS = [("Logistic Regression", "logistic_regression"), ("Decision Tree", "decision_tree"),
          ("KNN", "knn"), ("Random Forest", "random_forest")]

val_df = prepare_df(load_relabeled(config.PROJECT_ROOT / "validation"))
Xval, yval, _ = build_xy(val_df)
yv = yval.to_numpy()

# ---- undersample spoiled -> 50/50 ----
rng = np.random.default_rng(config.RANDOM_STATE)
fresh_i = np.where(yv == 0)[0]
spoil_i = np.where(yv == 1)[0]
keep = np.sort(np.concatenate([fresh_i, rng.choice(spoil_i, size=len(fresh_i), replace=False)]))
Xb, yb = Xval.iloc[keep], yv[keep]
print(f"ASLI      : {len(yv)} baris (fresh {len(fresh_i)}, spoiled {len(spoil_i)} = 19/81)")
print(f"REBALANCED: {len(yb)} baris (fresh {(yb==0).sum()}, spoiled {(yb==1).sum()} = 50/50)\n")


def met(X, ytrue, pipe):
    yp = pipe.predict(X)
    pr = pipe.predict_proba(X)[:, 1]
    return (accuracy_score(ytrue, yp),
            recall_score(ytrue, yp, average="macro", zero_division=0),
            recall_score(ytrue, yp, pos_label=1, zero_division=0),
            f1_score(ytrue, yp, average="macro", zero_division=0),
            roc_auc_score(ytrue, pr))


rows = []
for mname, mslug in MODELS:
    for sname, sslug in SPLITS:
        pipe = joblib.load(MODELS_DIR / f"{sslug}__{mslug}.joblib")
        a0, rm0, rs0, f0, au0 = met(Xval, yv, pipe)
        a1, rm1, rs1, f1_, au1 = met(Xb, yb, pipe)
        rows.append({"model": mname, "split": sname,
                     "acc_asli": a0, "acc_50/50": a1,
                     "recMacro_asli": rm0, "recMacro_50/50": rm1,
                     "auc_asli": au0, "auc_50/50": au1})

df = pd.DataFrame(rows)
df.to_csv(config.RESULTS_DIR / "rebalance_validation.csv", index=False)
pd.set_option("display.width", 200)
print("Perbandingan validation ASLI (19/81) vs REBALANCED (50/50):")
print(df.round(3).to_string(index=False))
print("\nRata-rata perubahan absolut: "
      f"accuracy {abs(df['acc_asli']-df['acc_50/50']).mean():.3f} | "
      f"recall_macro {abs(df['recMacro_asli']-df['recMacro_50/50']).mean():.3f} | "
      f"ROC-AUC {abs(df['auc_asli']-df['auc_50/50']).mean():.3f}")
