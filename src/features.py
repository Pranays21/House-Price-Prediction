"""Feature engineering and correlation analysis helpers."""
import pandas as pd

import config


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features used by the training pipeline.

    - total_rooms: bedrooms + bathrooms
    - area_per_room: area divided by (total_rooms + 1) to avoid div-by-zero

    Args:
        df: Input DataFrame with the raw schema.

    Returns:
        A copy with the two new columns appended.
    """
    df = df.copy()
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["area_per_room"] = df["area"] / (df["total_rooms"] + 1)
    return df


ENGINEERED_NUMERIC = ["total_rooms", "area_per_room"]


def correlation_with_target(df: pd.DataFrame, target: str | None = None) -> pd.Series:
    """Pearson correlation of every numeric column with the target.

    Args:
        df: DataFrame (raw schema is fine).
        target: Target column name, defaults to config.TARGET_COL.

    Returns:
        Series of correlations sorted descending, excluding the target itself.
    """
    target = target or config.TARGET_COL
    corr = df.select_dtypes(include="number").corr(numeric_only=True)[target]
    return corr.drop(labels=[target]).sort_values(ascending=False)
