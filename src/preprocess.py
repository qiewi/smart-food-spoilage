"""Prepare features/target/groups and build the leakage-safe preprocessor."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def apply_baseline_correction(df: pd.DataFrame) -> pd.DataFrame:
    """Replace each MQ sensor with R/R0, normalized per run.

    R0 is each run's own baseline (mean over the first BASELINE_MINUTES
    after warm-up). Because it uses only that run's early readings, it is
    leakage-free across CV folds and reproducible at deployment (warm up
    the sensor with the fresh food, record the baseline, then ratio).
    """
    df = df.copy()
    start = config.WARMUP_MINUTES * 60
    end = (config.WARMUP_MINUTES + config.BASELINE_MINUTES) * 60
    in_window = (df["elapsed_sec"] >= start) & (df["elapsed_sec"] < end)

    for sensor in config.MQ_SENSORS:
        r0 = df[in_window].groupby(config.GROUP_COLUMN)[sensor].mean()
        r0_per_row = df[config.GROUP_COLUMN].map(r0)
        # Fallback for any run with no rows in the window: its overall mean.
        fallback = df.groupby(config.GROUP_COLUMN)[sensor].transform("mean")
        r0_per_row = r0_per_row.fillna(fallback)
        df[sensor] = df[sensor] / r0_per_row
    return df


def trim_warmup(df: pd.DataFrame, minutes: int = config.WARMUP_MINUTES) -> pd.DataFrame:
    """Drop the first `minutes` of EACH run (sensor warm-up).

    Measured from each run's own first reading: recording starts the moment the food
    is placed and the sensor powers on, so the opening `minutes` of every run are
    warm-up regardless of the absolute elapsed value. Row removal only, applied before
    splitting — no fitted statistics, so it cannot leak test information into train.
    """
    cutoff = minutes * 60
    start = df.groupby(config.GROUP_COLUMN)["elapsed_sec"].transform("min")
    return df[df["elapsed_sec"] >= start + cutoff].reset_index(drop=True)


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[config.LABEL_COLUMN] = df[config.LABEL_COLUMN].map(config.LABEL_MAP)
    if df[config.LABEL_COLUMN].isna().any():
        bad = df[config.LABEL_COLUMN].isna().sum()
        raise ValueError(f"{bad} rows have a label outside {set(config.LABEL_MAP)}")
    return df


def build_xy(df: pd.DataFrame):
    """Return (X, y, groups). X holds only the model features."""
    X = df[config.FEATURES].copy()
    y = df[config.LABEL_COLUMN].astype(int)
    groups = df[config.GROUP_COLUMN]
    return X, y, groups


def make_preprocessor() -> ColumnTransformer:
    """Impute + scale numerics and one-hot the food type.

    Wrapped in a ColumnTransformer so every fitted statistic (median,
    mean/std) is learned per CV fold, never on the held-out runs.
    """
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, config.NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                config.CATEGORICAL_FEATURES,
            ),
        ]
    )


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean + encode, returning a df that still carries run_id/food_type.

    baseline correction (optional) -> warm-up trim -> label encode.
    Keeps grouping/strata columns so the data can be split by run afterwards.
    """
    if config.BASELINE_CORRECTION:
        df = apply_baseline_correction(df)
    df = trim_warmup(df)
    df = encode_labels(df)
    return df


def prepare(df: pd.DataFrame):
    """Full preparation: returns (X, y, groups) for grouped cross-validation."""
    return build_xy(prepare_df(df))
