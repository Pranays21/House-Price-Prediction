"""Tests for src/preprocessing.py."""
import numpy as np
import pandas as pd

from src.preprocessing import build_preprocessor, cap_outliers_iqr


def test_cap_outliers_iqr_caps_extreme_values():
    df = pd.DataFrame({"area": [1000, 2000, 3000, 4000, 5000, 1_000_000]})
    capped = cap_outliers_iqr(df, ["area"])
    q1, q3 = df["area"].quantile(0.25), df["area"].quantile(0.75)
    upper = q3 + 1.5 * (q3 - q1)
    assert capped["area"].max() <= upper
    assert len(capped) == len(df)


def test_build_preprocessor_transforms_mixed_frame():
    df = pd.DataFrame({
        "area": [1000, 2000, np.nan],
        "bedrooms": [2, 3, 4],
        "mainroad": ["yes", np.nan, "no"],
        "furnishingstatus": ["furnished", "unfurnished", "semi-furnished"],
    })
    pre = build_preprocessor(
        numeric_features=["area", "bedrooms"],
        categorical_features=["mainroad", "furnishingstatus"],
    )
    out = pre.fit_transform(df)
    assert out.shape[0] == 3
    assert not np.isnan(out).any()
    # 2 numeric + 2 mainroad cats + 3 furnishing cats = 7
    assert out.shape[1] == 7
