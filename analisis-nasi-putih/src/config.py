"""Konfigurasi pipeline Nasi Putih (mandiri, gas-only)."""

from pathlib import Path

BASE = Path(__file__).resolve().parent.parent          # analisis-nasi-putih/
TRAIN_DIR = BASE / "training"
VAL_DIR = BASE / "validation"
RESULTS_DIR = BASE / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RANDOM_STATE = 42
WARMUP_MINUTES = 5                    # buang 5 menit awal tiap trial (warm-up sensor)
FRESH_THRESHOLD_HOURS = 6             # Nasi Putih: fresh < 6 jam, sisanya spoiled
N_SPLITS = 3
RF_N_ITER = 40                        # RandomizedSearchCV untuk Random Forest

# Seleksi fitur = GAS ONLY (single-food -> food_type konstan; humidity/tempC tak dipakai)
FEATURES = ["mq2", "mq135", "mq4"]
LABEL_COLUMN = "label"
GROUP_COLUMN = "run_id"               # 1 file = 1 trial = 1 grup
LABEL_MAP = {"fresh": 0, "spoiled": 1}   # positif = spoiled = 1
POSITIVE_LABEL = 1

# Refit F1 (bukan recall) supaya tidak memilih classifier trivial semua-spoiled.
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
REFIT_METRIC = "f1"
