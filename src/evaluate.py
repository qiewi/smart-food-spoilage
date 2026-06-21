"""Leakage-free evaluation: out-of-fold metrics, plots, model selection."""

import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict

from . import config
from .train import cv_score_summary, format_cv_summary, make_cv

CLASS_NAMES = ["fresh", "spoiled"]


def out_of_fold_predictions(estimator, X, y, groups):
    """Each sample predicted only while its run is the held-out fold."""
    cv = make_cv()
    y_pred = cross_val_predict(clone(estimator), X, y, groups=groups, cv=cv, n_jobs=-1)
    y_proba = cross_val_predict(
        clone(estimator), X, y, groups=groups, cv=cv, n_jobs=-1,
        method="predict_proba",
    )[:, config.POSITIVE_LABEL]
    return y_pred, y_proba


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Binary-classification metric suite for the comparison table.

    Recall is reported for the positive class `spoiled` (the priority metric);
    Precision/F1 use macro-average (treats both classes equally); MCC is a
    balanced single-number score robust to class imbalance.
    """
    pos = config.POSITIVE_LABEL
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_spoiled": recall_score(y_true, y_pred, pos_label=pos, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def plot_confusion_matrix(y_true, y_pred, name, out_dir=config.FIGURES_DIR):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]}", ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    # Highlight the dangerous cell: spoiled predicted as fresh (false negative).
    ax.text(0, 1, f"\n\nFN={cm[1, 0]}", ha="center", va="center", color="red")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = out_dir / f"confusion_{name.replace(' ', '_').lower()}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def per_food_mq_importance(fitted_estimator, X, y, scoring="f1", n_repeats=10):
    """Permutation importance of the MQ sensors, computed per food type."""
    foods = X[config.CATEGORICAL_FEATURES[0]]
    rows = {}
    for food in sorted(foods.unique()):
        mask = (foods == food).values
        result = permutation_importance(
            fitted_estimator, X[mask], y[mask], scoring=scoring,
            n_repeats=n_repeats, random_state=config.RANDOM_STATE, n_jobs=1,
        )
        imp = pd.Series(result.importances_mean, index=X.columns)
        rows[food] = imp[config.MQ_SENSORS]
    return pd.DataFrame(rows).T


def plot_per_food_mq_importance(table, model_label="", filename="importance_per_food_mq.png",
                                scoring="f1", out_dir=config.FIGURES_DIR):
    fig, ax = plt.subplots(figsize=(8, 5))
    table.plot.bar(ax=ax)
    title = "Per-food MQ sensor importance (permutation)"
    if model_label:
        title += f" - {model_label}"
    ax.set_title(title)
    ax.set_ylabel(f"Mean drop in {scoring} when shuffled")
    ax.set_xlabel("Food type")
    ax.tick_params(axis="x", rotation=15)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(title="Sensor")
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def measure_inference_latency(fitted_estimator, X, n=200) -> float:
    """Average seconds per single-sample predict (NFR02 < 2s/sample)."""
    sample = X.iloc[[0]]
    start = time.perf_counter()
    for _ in range(n):
        fitted_estimator.predict(sample)
    return (time.perf_counter() - start) / n


def select_best(metrics_table: pd.DataFrame, cv_summaries: dict) -> str:
    """Highest recall(spoiled) -> F1 tiebreak -> smallest CV std."""
    df = metrics_table.copy()
    df["f1_cv_std"] = [cv_summaries[m]["f1_std"] for m in df.index]
    df = df.sort_values(
        by=["recall_spoiled", "f1_macro", "f1_cv_std"],
        ascending=[False, False, True],
    )
    return df.index[0]


def evaluate_all(searches, X, y, groups):
    """Run full evaluation; write tables, figures, best model. Returns report str."""
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics_rows = {}
    cv_summaries = {}
    report_chunks = ["=" * 70, "CROSS-VALIDATION (StratifiedGroupKFold, mean +/- std)", "=" * 70]

    for name, search in searches.items():
        cv_summaries[name] = cv_score_summary(search)
        report_chunks.append(format_cv_summary(name, cv_summaries[name]))

        y_pred, y_proba = out_of_fold_predictions(search.best_estimator_, X, y, groups)
        metrics_rows[name] = compute_metrics(y, y_pred, y_proba)
        # Inference time per sample (best_estimator_ is refit on all data).
        metrics_rows[name]["inference_s"] = measure_inference_latency(search.best_estimator_, X)

        report_chunks.append(f"\nOut-of-fold classification report - {name}")
        report_chunks.append(
            classification_report(y, y_pred, target_names=CLASS_NAMES, digits=3)
        )
        plot_confusion_matrix(y, y_pred, name)

    metrics_table = pd.DataFrame(metrics_rows).T
    metrics_table = metrics_table[
        ["accuracy", "precision_macro", "recall_spoiled", "f1_macro", "mcc", "roc_auc", "inference_s"]
    ]
    metrics_table.to_csv(config.RESULTS_DIR / "comparison_table.csv")

    report_chunks += ["", "=" * 70, "MODEL COMPARISON (out-of-fold)", "=" * 70]
    report_chunks.append(metrics_table.round(3).to_string())

    best_name = select_best(metrics_table, cv_summaries)
    best_estimator = searches[best_name].best_estimator_
    joblib.dump(best_estimator, config.RESULTS_DIR / "best_model.joblib")

    # Per-food MQ-sensor importance ("which gas sensor matters per food").
    def _emit_importance(estimator, label, csv_name, fig_name):
        tbl = per_food_mq_importance(estimator, X, y)
        tbl.to_csv(config.RESULTS_DIR / csv_name)
        plot_per_food_mq_importance(tbl, model_label=label, filename=fig_name)
        report_chunks.extend([
            "",
            "=" * 70,
            f"PER-FOOD MQ IMPORTANCE (permutation, model: {label})",
            "=" * 70,
            tbl.round(4).to_string(),
        ])
        return tbl

    _emit_importance(
        best_estimator, best_name,
        "feature_importance_per_food.csv", "importance_per_food_mq.png",
    )
    # Random Forest gives the cleaner, standard tree-based interpretation.
    if "Random Forest" in searches and best_name != "Random Forest":
        _emit_importance(
            searches["Random Forest"].best_estimator_, "Random Forest",
            "feature_importance_per_food_rf.csv", "importance_per_food_mq_rf.png",
        )

    latency = measure_inference_latency(best_estimator, X)
    report_chunks += [
        "",
        "=" * 70,
        f"BEST MODEL: {best_name}",
        f"  recall(spoiled) = {metrics_table.loc[best_name, 'recall_spoiled']:.3f}",
        f"  accuracy        = {metrics_table.loc[best_name, 'accuracy']:.3f}  (NFR01 target > 0.85)",
        f"  inference       = {latency * 1000:.3f} ms/sample  (NFR02 target < 2000 ms)",
        "Saved -> results/best_model.joblib",
        "=" * 70,
    ]

    report = "\n".join(report_chunks)
    (config.RESULTS_DIR / "classification_reports.txt").write_text(report, encoding="utf-8")
    return report, metrics_table, cv_summaries, best_name
