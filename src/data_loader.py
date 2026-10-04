"""Data loading and validation for the housing dataset."""
from pathlib import Path

import pandas as pd

import config
from config import TARGET_COL


class DataLoadError(Exception):
    """Raised when the dataset cannot be loaded or validated."""


def load_data(path: str | Path = "data/housing.csv") -> pd.DataFrame:
    """Load the housing CSV and validate the target column exists.

    Args:
        path: Path to the CSV file.

    Returns:
        The loaded DataFrame.

    Raises:
        DataLoadError: If the file is missing, empty, unreadable, or the
            target column is absent.
    """
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"Dataset not found at {path}.")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - surface as friendly error
        raise DataLoadError(f"Could not read {path}: {exc}") from exc
    if df.empty:
        raise DataLoadError(f"Dataset at {path} is empty.")
    if TARGET_COL not in df.columns:
        raise DataLoadError(
            f"Target column '{TARGET_COL}' not found. "
            f"Columns present: {list(df.columns)}"
        )
    return df


def basic_checks(df: pd.DataFrame) -> dict:
    """Return a small dict of sanity checks for logging / display."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_total": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "target_mean": float(df[config.TARGET_COL].mean()),
        "target_std": float(df[config.TARGET_COL].std()),
    }
