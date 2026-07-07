"""Load the saved (split x model) pipelines and run predictions — GAS-ONLY.

Each .joblib is a full sklearn Pipeline (preprocessing + classifier), so a single
.predict() works on the raw input (mq2, mq135, mq4 + food_type).
"""

import json
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).parent / "models"
_META = json.loads((MODELS_DIR / "metadata.json").read_text())

MODEL_SLUG = {"Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree",
              "KNN": "knn", "Random Forest": "random_forest"}
SPLIT_SLUG = {"Grouped": "grouped", "Random Split": "stratifiedkfold"}
INV_LABEL = {v: k for k, v in _META["label_map"].items()}   # 0->fresh, 1->spoiled

_cache = {}


def _load(split, model):
    key = (split, model)
    if key not in _cache:
        fn = MODELS_DIR / f"{SPLIT_SLUG[split]}__{MODEL_SLUG[model]}.joblib"
        if not fn.exists():
            raise FileNotFoundError(f"Model belum tersedia: {fn.name}")
        _cache[key] = joblib.load(fn)
    return _cache[key]


def metadata():
    return _META


def predict(split, model, food_type, mq2, mq135, mq4):
    if split not in SPLIT_SLUG:
        raise ValueError(f"split tidak dikenal: {split}")
    if model not in MODEL_SLUG:
        raise ValueError(f"model tidak dikenal: {model}")
    pipe = _load(split, model)
    row = pd.DataFrame([{"mq2": mq2, "mq135": mq135, "mq4": mq4, "food_type": food_type}])
    pred = int(pipe.predict(row)[0])
    proba = pipe.predict_proba(row)[0]
    classes = list(pipe.classes_)
    return {
        "label": INV_LABEL[pred],
        "prob_spoiled": float(proba[classes.index(1)]),
        "prob_fresh": float(proba[classes.index(0)]),
        "model_metrics": _META["metrics"][split][model],
    }
