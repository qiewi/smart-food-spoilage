"""Cross-validation simetris: Grouped=StratifiedGroupKFold, Random=StratifiedKFold.
Rotasi penuh (3 fold) -> mean ± std per model per split, untuk internal test & validasi.
Figur: OOF-pooled (internal) + rata-rata fold (validasi)."""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from . import config
from .models import model_specs, tune

warnings.filterwarnings("ignore")
CLASSES = ["fresh", "spoiled"]
MODEL_COLOR = {"Logistic Regression": "#10b981", "Decision Tree": "#0ea5e9",
               "KNN": "#8b5cf6", "Random Forest": "#f59e0b"}
SPLIT_ORDER = ["Grouped", "Random Split"]
METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"]
MODELS = [s[0] for s in model_specs()]


def metrics(yt, yp, pr):
    two = len(np.unique(yt)) == 2
    return {"accuracy": accuracy_score(yt, yp),
            "precision_macro": precision_score(yt, yp, average="macro", zero_division=0),
            "recall_macro": recall_score(yt, yp, average="macro", zero_division=0),
            "f1_macro": f1_score(yt, yp, average="macro", zero_division=0),
            "roc_auc": roc_auc_score(yt, pr) if two else float("nan")}


def rebalance_5050(X, y):
    """Undersample kelas mayoritas -> fresh:spoiled 50:50."""
    y = np.asarray(y)
    rng = np.random.default_rng(config.RANDOM_STATE)
    fi, si = np.where(y == 0)[0], np.where(y == 1)[0]
    m = min(len(fi), len(si))
    keep = np.sort(np.concatenate([rng.choice(fi, m, replace=False),
                                   rng.choice(si, m, replace=False)]))
    return X.iloc[keep], y[keep]


def _agg(fold_dicts):
    out = {}
    for m in METRICS:
        vals = [d[m] for d in fold_dicts]
        out[f"{m}_mean"] = float(np.nanmean(vals))
        out[f"{m}_std"] = float(np.nanstd(vals))
    return out


def cv_evaluate(X, y, groups, Xv, yv):
    """CV rotasi penuh untuk 2 skema split. Return (internal_df, validation_df) dgn mean±std."""
    yn = y.to_numpy()
    schemes = [
        ("Grouped", StratifiedGroupKFold(config.N_SPLITS, shuffle=True,
                                         random_state=config.RANDOM_STATE), True),
        ("Random Split", StratifiedKFold(config.N_SPLITS, shuffle=True,
                                         random_state=config.RANDOM_STATE), False),
    ]
    internal_rows, validation_rows = [], []
    oof, valens = {}, {}   # figur: (split,model)->(ytrue,proba) OOF ; (split,model)->proba rata2 fold
    for sname, cv, grouped in schemes:
        folds = list(cv.split(X, y, groups)) if grouped else list(cv.split(X, y))
        per = {m: {"int": [], "val": [], "oy": [], "op": [], "vp": []} for m in MODELS}
        for tri, tei in folds:
            yte = yn[tei]
            for name, est, grid, kind in model_specs():
                best, _ = tune(est, grid, kind, X.iloc[tri], y.iloc[tri])
                pos = list(best.classes_).index(1)
                pte = best.predict_proba(X.iloc[tei])[:, pos]
                pv = best.predict_proba(Xv)[:, pos]
                per[name]["int"].append(metrics(yte, best.predict(X.iloc[tei]), pte))
                per[name]["val"].append(metrics(yv, best.predict(Xv), pv))
                per[name]["oy"].append(yte); per[name]["op"].append(pte); per[name]["vp"].append(pv)
        print(f"  [selesai CV {len(folds)} fold] {sname}")
        for name in MODELS:
            internal_rows.append({"split": sname, "model": name, **_agg(per[name]["int"])})
            validation_rows.append({"split": sname, "model": name, **_agg(per[name]["val"])})
            oof[(sname, name)] = (np.concatenate(per[name]["oy"]), np.concatenate(per[name]["op"]))
            valens[(sname, name)] = np.mean(per[name]["vp"], axis=0)

    val_entries = {(s, m): (yv, valens[(s, m)]) for s in SPLIT_ORDER for m in MODELS}
    # Internal test dari prediksi OOF 3-fold DIGABUNG (lintas-trial, jujur) -> satu angka per
    # metrik, PERSIS sumber yang dipakai confusion_matrices_internal.png & roc_curves_internal.png.
    internal_pooled = [
        {"split": s, "model": m,
         **metrics(oof[(s, m)][0], (oof[(s, m)][1] >= 0.5).astype(int), oof[(s, m)][1])}
        for s in SPLIT_ORDER for m in MODELS]
    # Validation dari probabilitas 3-fold DIRATA-RATA per baris (soft-ensemble) -> satu angka per
    # metrik, PERSIS sumber confusion_matrices_validation.png & roc_curves_validation.png.
    validation_pooled = [
        {"split": s, "model": m,
         **metrics(val_entries[(s, m)][0], (val_entries[(s, m)][1] >= 0.5).astype(int),
                   val_entries[(s, m)][1])}
        for s in SPLIT_ORDER for m in MODELS]
    _plot_cm(oof, config.FIGURES_DIR / "confusion_matrices_internal.png",
             "Confusion Matrix — internal test OOF (Nasi Putih) — 2 split × 4 model")
    _plot_cm(val_entries, config.FIGURES_DIR / "confusion_matrices_validation.png",
             "Confusion Matrix — validation 50/50 (Nasi Putih) — 2 split × 4 model")
    _plot_roc(oof, config.FIGURES_DIR / "roc_curves_internal.png",
              "Kurva ROC — internal test OOF (Nasi Putih)")
    _plot_roc(val_entries, config.FIGURES_DIR / "roc_curves_validation.png",
              "Kurva ROC — validation 50/50 (Nasi Putih)")

    cols = ["split", "model"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    pooled_cols = ["split", "model"] + METRICS
    return (pd.DataFrame(internal_rows)[cols], pd.DataFrame(validation_rows)[cols],
            pd.DataFrame(internal_pooled)[pooled_cols],
            pd.DataFrame(validation_pooled)[pooled_cols])


def final_params(X, y):
    """Param terpilih model final (tune di SELURUH data training)."""
    rows = []
    for name, est, grid, kind in model_specs():
        _, bp = tune(est, grid, kind, X, y)
        rows.append({"model": name, **bp})
    return pd.DataFrame(rows)


def _plot_cm(entries, path, title):
    fig, axes = plt.subplots(len(SPLIT_ORDER), len(MODELS), figsize=(14, 7))
    for i, s in enumerate(SPLIT_ORDER):
        for j, m in enumerate(MODELS):
            yt, pr = entries[(s, m)]
            yp = (pr >= 0.5).astype(int)
            cm = confusion_matrix(yt, yp, labels=[0, 1])
            acc = accuracy_score(yt, yp)
            rec = recall_score(yt, yp, average="macro", zero_division=0)
            ax = axes[i, j]
            ax.imshow(cm, cmap="Greens", vmin=0, vmax=cm.max())
            for (r, c), v in np.ndenumerate(cm):
                ax.text(c, r, str(v), ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if v > cm.max() * 0.6 else "#111")
            ax.set_xticks([0, 1]); ax.set_xticklabels(CLASSES, fontsize=8)
            ax.set_yticks([0, 1]); ax.set_yticklabels(CLASSES, fontsize=8)
            ax.set_xlabel(f"Prediksi\nacc={acc:.2f}  rec-macro={rec:.2f}", fontsize=8)
            if i == 0:
                ax.set_title(m, fontsize=11, fontweight="bold", pad=10)
            if j == 0:
                ax.set_ylabel(f"{s}\n\nAktual", fontsize=9, fontweight="bold")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_roc(entries, path, title):
    fig, axes = plt.subplots(1, len(SPLIT_ORDER), figsize=(10.5, 5.2))
    for j, s in enumerate(SPLIT_ORDER):
        ax = axes[j]
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#999", label="Acak (AUC 0.50)")
        for m, col in MODEL_COLOR.items():
            yt, pr = entries[(s, m)]
            fpr, tpr, _ = roc_curve(yt, pr)
            ax.plot(fpr, tpr, lw=2, color=col, label=f"{m} (AUC {roc_auc_score(yt, pr):.3f})")
        ax.set_title(s, fontsize=12, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        if j == 0:
            ax.set_ylabel("True Positive Rate")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
        ax.grid(True, ls=":", lw=0.6, color="#ddd")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
