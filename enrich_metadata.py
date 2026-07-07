"""Add training (internal) metrics to dashboard metadata.json from results/tbl_*.csv.

The 12 models already have validation metrics in metadata.json. This adds the
matching internal/training metrics (same tuning) so the dashboard can show the
training->validation gap (the leakage story). No model retraining.
"""

import json

import pandas as pd

from src import config

OUT = config.PROJECT_ROOT / "dashboard" / "backend" / "models"
meta = json.loads((OUT / "metadata.json").read_text())

FILES = {
    "StratifiedKFold": "tbl_StratifiedKFold_internal.csv",
    "MonteCarlo": "tbl_MonteCarlo_internal.csv",
    "Grouped": "tbl_Grouped_internal.csv",
}
COLS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"]

meta["train_metrics"] = {}
for split, fn in FILES.items():
    df = pd.read_csv(config.RESULTS_DIR / fn, index_col=0)
    meta["train_metrics"][split] = {
        model: {c: float(df.loc[model, c]) for c in COLS} for model in meta["models"]
    }

(OUT / "metadata.json").write_text(json.dumps(meta, indent=2))
print("metadata.json diperkaya dengan train_metrics untuk", list(FILES))
