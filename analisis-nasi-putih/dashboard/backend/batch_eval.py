"""Evaluasi batch CSV — skor (split, model) GAS-ONLY Nasi Putih pada CSV berlabel yang diunggah.

Terpisah dari /api/predict satu-baris: di sini user mengunggah satu/lebih CSV (kolom wajib
mq2, mq135, mq4, label; `food_type` opsional & diabaikan; `elapsed` opsional untuk warm-up
trim + koreksi label monoton) dan mendapat performa model (metrik makro + confusion matrix),
keseluruhan & per file. Preprocessing memakai pipeline analisis-nasi-putih/src (gas-only) agar
konsisten dengan cara model dilatih.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

# jadikan paket `src` analisis-nasi-putih importable dari dashboard/backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config, data

import inference

REQUIRED = ["mq2", "mq135", "mq4", "label"]


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
    """Parse satu CSV -> (X, y) siap pipeline gas-only. Raise bila input buruk."""
    df = pd.read_csv(io.StringIO(content))
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"kolom hilang: {missing}")
    df[config.GROUP_COLUMN] = name.replace(".csv", "")
    if "elapsed" in df.columns:                        # ada waktu -> cleaning penuh
        df["elapsed_sec"] = df["elapsed"].map(data.elapsed_to_seconds)
        df = data.clean(df)                            # cummax + re-derive label + trim 5 mnt
        df = data.encode_labels(df)
    elif not pd.api.types.is_numeric_dtype(df["label"]):
        df = data.encode_labels(df)                    # label teks fresh/spoiled -> 0/1
    else:
        df["label"] = df["label"].astype(int)          # label sudah 0/1
    X, y, _ = data.build_xy(df)
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
                         "mq2, mq135, mq4, label tersedia.")

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
