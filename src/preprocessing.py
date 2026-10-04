"""Preprocessing: outlier capping and the sklearn ColumnTransformer.

Numeric columns: median imputation + StandardScaler.
Categorical columns: most-frequent imputation + OneHotEncoder.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


def cap_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Cap outliers in numeric columns using the IQR rule.

    Values below Q1 - k*IQR are raised to the lower bound and values above
    Q3 + k*IQR are lowered to the upper bound (k from config).

    Args:
        df: Input DataFrame.
        columns: Numeric columns to cap.

    Returns:
        A copy of df with outliers capped.
    """
    df = df.copy()
    k = config.OUTLIER_IQR_MULTIPLIER
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        df[col] = df[col].clip(lower, upper)
    return df


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Build the ColumnTransformer used by every model pipeline."""
    numeric_features = numeric_features or config.NUMERIC_FEATURES
    categorical_features = categorical_features or config.CATEGORICAL_FEATURES

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return output feature names after fitting the preprocessor."""
    names: list[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(list(cols))
        elif name == "cat":
            ohe = transformer.named_steps["onehot"]
            names.extend(list(ohe.get_feature_names_out(cols)))
    return names
