"""Orkestrator pipeline Nasi Putih (gas-only), CV simetris.

Alur: load -> cleaning (elapsed monoton + warm-up 5 mnt) -> label encoding ->
seleksi fitur gas-only -> standardization (dalam pipeline) -> cross-validation:
  Random Split = StratifiedKFold(3), Grouped = StratifiedGroupKFold(3),
rotasi penuh -> mean ± std (internal test & validation 50/50). Tulis CSV + figur.

Jalankan:  cd analisis-nasi-putih && python run_analysis.py
"""

import warnings

import pandas as pd

from src import config, data, evaluate

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)


def _compo(df, name):
    g = df["label"].value_counts()
    f, s = int(g.get("fresh", g.get(0, 0))), int(g.get("spoiled", g.get(1, 0)))
    print(f"{name}: {len(df)} baris | fresh {f} / spoiled {s}")


def _fmt_plain(df):
    """Tabel satu-angka (tanpa std), indeks model."""
    return df.drop(columns="split").set_index("model").round(3)


_METRIC_LABEL = {"accuracy": "Accuracy", "precision_macro": "Precision (macro)",
                 "recall_macro": "Recall (macro)", "f1_macro": "F1 (macro)", "roc_auc": "AUC"}


def _write_table_md(df, path, title, note_lines):
    """Tulis tabel metrik (tanpa std) ke Markdown untuk dokumen tesis."""
    head = "| Model | " + " | ".join(_METRIC_LABEL[m] for m in evaluate.METRICS) + " |"
    sep = "|" + "---|" * (len(evaluate.METRICS) + 1)
    lines = [f"# {title}", ""] + note_lines + [""]
    for s in evaluate.SPLIT_ORDER:
        lines += [f"## {s}", "", head, sep]
        for _, r in df[df.split == s].iterrows():
            vals = " | ".join(f"{r[m]:.3f}" for m in evaluate.METRICS)
            lines.append(f"| {r['model']} | {vals} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- TRAINING ----
    tr = data.clean(data.load(config.TRAIN_DIR))
    _compo(tr, "TRAINING (setelah cleaning)")
    print("Trial:", sorted(tr["run_id"].unique()))
    tr = data.encode_labels(tr)
    Xtr, ytr, groups = data.build_xy(tr)
    elapsed_tr = tr["elapsed_sec"].to_numpy()      # metadata utk trailing-trim 70:30 (bukan fitur)
    print(f"Fitur (gas-only): {list(Xtr.columns)}\n")

    # ---- VALIDATION -> 50/50 ----
    val = data.clean(data.load(config.VAL_DIR))
    _compo(val, "VALIDATION (setelah cleaning)")
    val = data.encode_labels(val)
    Xval, yval, _ = data.build_xy(val)
    Xv, yv = evaluate.rebalance_5050(Xval, yval.to_numpy())
    print(f"VALIDATION 50/50: {len(yv)} baris (fresh {(yv==0).sum()} / spoiled {(yv==1).sum()})\n")

    # ---- Cross-validation (Grouped=StratifiedGroupKFold, Random=StratifiedKFold) ----
    print("Cross-validation 3 fold, internal test dipangkas ekor -> 70:30 (4 model x 2 split)…")
    (internal_df, validation_df,
     internal_pooled_df, validation_pooled_df) = evaluate.cv_evaluate(Xtr, ytr, groups, Xv, yv,
                                                                      elapsed_tr)
    params_df = evaluate.final_params(Xtr, ytr)

    # Internal & validation = pooled (tanpa std, konsisten dgn figur masing-masing).
    internal_pooled_df.round(4).to_csv(config.RESULTS_DIR / "metrics_internal.csv", index=False)
    validation_pooled_df.round(4).to_csv(config.RESULTS_DIR / "metrics_validation.csv", index=False)
    _write_table_md(internal_pooled_df, config.RESULTS_DIR / "tabel_internal_test.md",
                    "Evaluasi Internal Test — Nasi Putih (gas-only)",
                    ["Rotasi 3-fold; data uji tiap fold **dipangkas ekor (trailing-trim) → train:test = 70:30**",
                     "(ekor tiap trial dibuang urut waktu, agar uji tetap kontigu temporal & memuat 2 kelas).",
                     "Metrik dari prediksi **out-of-fold 3-fold digabung** (lintas-trial), sumber sama dengan",
                     "`confusion_matrices_internal.png` & `roc_curves_internal.png` → angka tabel = angka gambar.",
                     "Positif = spoiled; threshold 0.5."])
    _write_table_md(validation_pooled_df, config.RESULTS_DIR / "tabel_validation_test.md",
                    "Evaluasi Validation Test 50/50 — Nasi Putih (gas-only)",
                    ["Metrik dari probabilitas **3-fold dirata-rata per baris** (soft-ensemble),",
                     "sumber sama dengan `confusion_matrices_validation.png` & `roc_curves_validation.png`",
                     "→ angka tabel = angka gambar. Validasi = trial 0407, 50/50. Positif = spoiled; threshold 0.5."])
    params_df.to_csv(config.RESULTS_DIR / "best_params.csv", index=False)

    for title, df in [("INTERNAL TEST — OOF-pooled (rata-rata, tanpa std) — konsisten dgn figur", internal_pooled_df),
                      ("VALIDATION 50/50 — rata-rata (tanpa std) — konsisten dgn figur", validation_pooled_df)]:
        print("\n" + "=" * 90 + f"\n### {title}")
        for s in evaluate.SPLIT_ORDER:
            print(f"\n-- {s} --")
            print(_fmt_plain(df[df.split == s]).to_string())

    print("\nOutput: results/metrics_internal.csv, tabel_internal_test.md,")
    print("        results/metrics_validation.csv, tabel_validation_test.md, best_params.csv")
    print("  results/figures/confusion_matrices_{internal,validation}.png, roc_curves_{internal,validation}.png")


if __name__ == "__main__":
    main()
