"""Internal-test metrics per split (Tabel V.3) — GAS-ONLY, from the saved dashboard models.

Each split's 30% held-out portion of training/: Random Split = random rows; Grouped =
whole held-out trials trimmed temporal-contiguously (trailing rows dropped) so
train:test = 70:30 (supervisor requirement; both classes remain). Models = the saved
(split x model) pipelines (fit on the identical 70%, same RANDOM_STATE) — no retraining.

Output: results/internal_test_2split.csv + console table
"""

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from src import config
config.NUMERIC_FEATURES = ["mq2", "mq135", "mq4"]                 # GAS ONLY
config.FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES

from src.load_data import load_relabeled
from src.preprocess import build_xy, prepare_df

warnings.filterwarnings("ignore")
RS = config.RANDOM_STATE
MD = config.PROJECT_ROOT / "dashboard" / "backend" / "models"
MODELS = [("Logistic Regression", "logistic_regression"), ("Decision Tree", "decision_tree"),
          ("KNN", "knn"), ("Random Forest", "random_forest")]

prepared = prepare_df(load_relabeled(config.PROJECT_ROOT / "training")).reset_index(drop=True)
X, y, groups = build_xy(prepared)
ya, run_arr, elapsed = y.to_numpy(), groups.to_numpy(), prepared["elapsed_sec"].to_numpy()

_, skf_te = next(StratifiedShuffleSplit(1, test_size=0.30, random_state=RS).split(X, y))
g_tr, g_te = next(StratifiedGroupKFold(3, shuffle=True, random_state=RS).split(X, y, groups))
# trim each held-out trial's trailing rows so train:test = 70:30
target = int(round((0.30 / 0.70) * len(g_tr)))
kf = min(1.0, target / len(g_te))
kept = []
for r in np.unique(run_arr[g_te]):
    rows = g_te[run_arr[g_te] == r]
    rows = rows[np.argsort(elapsed[rows])]
    kept.extend(rows[:max(1, int(round(len(rows) * kf)))])
g_te = np.array(sorted(kept))
print(f"Internal test: Grouped {len(g_te)} baris, Random Split {len(skf_te)} baris\n")

rows_out = []
for sname, sslug, te in [("Grouped", "grouped", g_te), ("Random Split", "stratifiedkfold", skf_te)]:
    Xt, yt = X.iloc[te], ya[te]
    for mname, mslug in MODELS:
        p = joblib.load(MD / f"{sslug}__{mslug}.joblib")
        yp = p.predict(Xt)
        pr = p.predict_proba(Xt)[:, list(p.classes_).index(1)]
        rows_out.append({
            "split": sname, "model": mname,
            "accuracy": accuracy_score(yt, yp),
            "precision_macro": precision_score(yt, yp, average="macro", zero_division=0),
            "recall_macro": recall_score(yt, yp, average="macro", zero_division=0),
            "f1_macro": f1_score(yt, yp, average="macro", zero_division=0),
            "roc_auc": roc_auc_score(yt, pr),
        })

df = pd.DataFrame(rows_out)
df.round(4).to_csv(config.RESULTS_DIR / "internal_test_2split.csv", index=False)
print(df.round(3).to_string(index=False))
print("\nCSV: results/internal_test_2split.csv")
