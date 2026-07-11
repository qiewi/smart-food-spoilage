"""Data & pra-pemrosesan Nasi Putih: load -> cleaning -> label encoding -> seleksi fitur
-> standardization (di dalam pipeline). Semua leakage-safe (statistik di-fit per-fold)."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def elapsed_to_seconds(value) -> float:
    """'HH:MM:SS' -> detik (jam bisa > 24; diparse manual)."""
    s = str(value).strip()
    if ":" in s:
        h, m, sec = (int(x) for x in s.split(":"))
        return h * 3600 + m * 60 + sec
    return int(s) / 1000


def load(data_dir: Path) -> pd.DataFrame:
    """Gabungkan semua CSV di folder; tiap file = 1 trial (run_id)."""
    paths = sorted(Path(data_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"Tidak ada CSV di {data_dir}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["run_id"] = p.stem.replace("_relabeled", "")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["elapsed_sec"] = out["elapsed"].map(elapsed_to_seconds)
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """STEP 1 — Cleaning sensor data.

    (a) Perbaiki elapsed korup: paksa monoton naik per trial (cummax) — waktu asli tidak
        mungkin mundur karena logger merekam sekuensial. Memperbaiki glitch (mis. Nasi 0413).
    (b) Label konsistensi: turunkan ulang label dari elapsed monoton vs ambang 6 jam
        ("sekali spoiled tetap spoiled"). Menjamin label benar apa pun kondisi CSV.
    (c) Buang 5 menit awal tiap trial (warm-up sensor), relatif ke awal rekaman masing-masing.
    Catatan: elapsed/elapsed_sec HANYA metadata (untuk trim & urutan), tidak pernah jadi fitur.
    """
    df = df.copy()
    # (a) elapsed monoton per trial
    df["elapsed_sec"] = df.groupby(config.GROUP_COLUMN)["elapsed_sec"].cummax()
    # (b) label dari elapsed terkoreksi
    thr = config.FRESH_THRESHOLD_HOURS * 3600
    df[config.LABEL_COLUMN] = np.where(df["elapsed_sec"] >= thr, "spoiled", "fresh")
    # (c) warm-up trim: buang baris dalam WARMUP_MINUTES pertama tiap trial
    start = df.groupby(config.GROUP_COLUMN)["elapsed_sec"].transform("min")
    cutoff = config.WARMUP_MINUTES * 60
    return df[df["elapsed_sec"] >= start + cutoff].reset_index(drop=True)


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """STEP 2 — Label encoding: fresh->0, spoiled->1."""
    df = df.copy()
    df[config.LABEL_COLUMN] = df[config.LABEL_COLUMN].map(config.LABEL_MAP)
    if df[config.LABEL_COLUMN].isna().any():
        raise ValueError("Ada label di luar {fresh, spoiled}")
    return df


def build_xy(df: pd.DataFrame):
    """STEP 3 — Seleksi fitur (gas-only): X=[mq2,mq135,mq4], y=label, groups=run_id."""
    X = df[config.FEATURES].copy()
    y = df[config.LABEL_COLUMN].astype(int)
    groups = df[config.GROUP_COLUMN]
    return X, y, groups


def make_preprocessor() -> ColumnTransformer:
    """STEP 4 — Standardization: impute median + StandardScaler pada 3 sinyal gas.
    Dibungkus dalam Pipeline/ColumnTransformer -> di-fit per-fold (tak bocor)."""
    numeric = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(transformers=[("num", numeric, config.FEATURES)])


def prepare(df: pd.DataFrame):
    """Pipeline pra-pemrosesan penuh -> (X, y, groups)."""
    return build_xy(encode_labels(clean(df)))
