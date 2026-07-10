"""Central configuration for the freshness-classification pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "relabeled"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Opsi 2: gas sensors + environment (DHT22)
NUMERIC_FEATURES = ["mq2", "mq135", "mq4", "humidity", "tempC"]
# Gas sensors only (used for per-food feature importance).
MQ_SENSORS = ["mq2", "mq135", "mq4"]
# Combined model: food_type as one-hot feature (PRD section 3.3)
CATEGORICAL_FEATURES = ["food_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns that exist in the dataframe but must never be model features.
# `elapsed`/`elapsed_sec` leak the label (label was derived from elapsed),
# `run_id` is the CV grouping key, `label` is the target.
NON_FEATURE_COLUMNS = ["elapsed", "elapsed_sec", "run_id", "label"]

LABEL_COLUMN = "label"
GROUP_COLUMN = "run_id"
# Positive class = spoiled (the event we must not miss).
LABEL_MAP = {"fresh": 0, "spoiled": 1}
POSITIVE_LABEL = 1

RANDOM_STATE = 42
# CV folds for StratifiedGroupKFold = minimum runs per food type; raise when
# more runs are collected.
N_SPLITS = 3

# Drop the first few minutes of EACH run (relative to its own start) while the
# sensors warm up. Recording begins when the food is cooked and the sensor powers on.
WARMUP_MINUTES = 5

# Per-run baseline correction (R/R0): divide each MQ sensor by its own baseline
# to cancel the large run-to-run baseline drift. R0 = mean over the first
# BASELINE_MINUTES after warm-up (each run's own fresh-state reading).
BASELINE_CORRECTION = False
BASELINE_MINUTES = 10
