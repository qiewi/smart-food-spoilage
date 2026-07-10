"""Batch CSV evaluation — score a chosen (split, model) on uploaded labelled CSVs.

Separate from single-reading /api/predict: here the user uploads one or more relabeled
CSVs (columns mq2, mq135, mq4, food_type, label; optional elapsed for warm-up trim) and
gets the model's performance on that data (macro metrics + confusion matrix), overall and
per file. Preprocessing reuses the exact training pipeline (src.preprocess) so it is
consistent with how the models were trained. GAS-ONLY features.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

# make the repo-root `src` package importable from dashboard/backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config
config.NUMERIC_FEATURES = ["mq2", "mq135", "mq4"]                 # GAS ONLY
config.FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
from src.load_data import elapsed_to_seconds, normalize_food
from src.preprocess import build_xy, encode_labels, trim_warmup

import inference

REQUIRED = ["mq2", "mq135", "mq4", "food_type", "label"]


def _metrics(y, yp, proba):
    two_classes = len(np.unique(y)) == 2
    return {
        "accuracy": float(accuracy_score(y, yp)),
        "precision_macro": float(precision_score(y, yp, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, yp, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, yp, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if two_classes else None,
    }


def _prepare(name, content):
    """Parse one uploaded CSV -> (X, y) ready for the pipeline. Raises on bad input."""
    df = pd.read_csv(io.StringIO(content))
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"kolom hilang: {missing}")
    df["run_id"] = name.replace(".csv", "")
    df["food_type"] = df["food_type"].map(normalize_food)
    if "elapsed" in df.columns:                        # warm-up trim only if time is present
        df["elapsed_sec"] = df["elapsed"].map(elapsed_to_seconds)
        df = trim_warmup(df)
    # label may be text (fresh/spoiled) or already 0/1
    if not pd.api.types.is_numeric_dtype(df["label"]):
        df = encode_labels(df)
    else:
        df["label"] = df["label"].astype(int)
    X, y, _ = build_xy(df)
    return X, y.to_numpy()


def evaluate(split, model, files):
    if split not in inference.SPLIT_SLUG:
        raise ValueError(f"split tidak dikenal: {split}")
    if model not in inference.MODEL_SLUG:
        raise ValueError(f"model tidak dikenal: {model}")
    pipe = inference._load(split, model)
    pos = list(pipe.classes_).index(1)

    per_file, warnings = [], []
    ys, yps, probas = [], [], []
    for f in files:
        name = f["name"]
        try:
            X, y = _prepare(name, f["content"])
            if len(y) == 0:
                warnings.append(f"{name}: tidak ada baris tersisa (semua warm-up) — dilewati")
                continue
            yp = pipe.predict(X)
            proba = pipe.predict_proba(X)[:, pos]
        except Exception as e:  # noqa: BLE001
            warnings.append(f"{name}: {e} — dilewati")
            continue
        cm = confusion_matrix(y, yp, labels=[0, 1]).tolist()
        per_file.append({
            "filename": name, "n_rows": int(len(y)),
            "n_fresh": int((y == 0).sum()), "n_spoiled": int((y == 1).sum()),
            "confusion": cm, **_metrics(y, yp, proba),
        })
        ys.append(y); yps.append(yp); probas.append(proba)

    if not per_file:
        raise ValueError("Tidak ada file valid untuk dievaluasi. Pastikan kolom "
                         "mq2, mq135, mq4, food_type, label tersedia.")

    y = np.concatenate(ys); yp = np.concatenate(yps); proba = np.concatenate(probas)
    overall = {
        "n_rows": int(len(y)), "n_fresh": int((y == 0).sum()), "n_spoiled": int((y == 1).sum()),
        "confusion": confusion_matrix(y, yp, labels=[0, 1]).tolist(), **_metrics(y, yp, proba),
    }
    return {
        "split": split, "model": model, "n_files": len(per_file),
        "overall": overall, "per_file": per_file, "warnings": warnings,
        "reference_metrics": inference._META["metrics"][split][model],
    }
