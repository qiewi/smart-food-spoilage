"""Load and combine the relabeled CSVs into a single tidy DataFrame."""

from pathlib import Path

import pandas as pd

from . import config

# Map slightly different food_type spellings to a canonical label so the
# one-hot encoding stays clean (mirrors the aliases used in relabel.py).
FOOD_ALIASES = {
    "ikan goreng nila": "Ikan Goreng",
    "ikan goreng": "Ikan Goreng",
    "nasi putih": "Nasi Putih",
    "nasi": "Nasi Putih",
    "ayam goreng": "Ayam Goreng",
    "telur rebus": "Telur Rebus",
}


def normalize_food(name: str) -> str:
    key = str(name).strip().lower()
    return FOOD_ALIASES.get(key, str(name).strip())


def elapsed_to_seconds(value: str) -> float:
    """Convert 'HH:MM:SS' to seconds.

    Hours can exceed 24 (e.g. '35:59:27'), so this is parsed manually
    rather than via pd.to_timedelta.
    """
    s = str(value).strip()
    if ":" in s:
        h, m, sec = (int(x) for x in s.split(":"))
        return h * 3600 + m * 60 + sec
    # Fallback for older datasets that stored raw milliseconds.
    return int(s) / 1000


def load_relabeled(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """Combine every relabeled CSV, tagging each file as one run/group."""
    csv_paths = sorted(Path(data_dir).glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        # 1 file = 1 trial = 1 run (used as CV group, never as a feature).
        df["run_id"] = path.stem.replace("_relabeled", "")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["food_type"] = combined["food_type"].map(normalize_food)
    combined["elapsed_sec"] = combined["elapsed"].map(elapsed_to_seconds)
    return combined


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """fresh:spoiled counts per food type (for EDA)."""
    return (
        df.groupby(["food_type", config.LABEL_COLUMN])
        .size()
        .unstack(fill_value=0)
        .assign(total=lambda d: d.sum(axis=1))
    )


def run_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Rows per run plus its food type (confirms group structure)."""
    return (
        df.groupby(["food_type", config.GROUP_COLUMN])
        .size()
        .rename("n_rows")
        .reset_index()
    )


def missing_report(df: pd.DataFrame) -> pd.Series:
    """NaN counts (DHT22 humidity/tempC occasionally drop out)."""
    return df[config.NUMERIC_FEATURES].isna().sum()
