"""Single-prediction inference using the saved best model."""
from pathlib import Path

import joblib
import pandas as pd

import config
from src.features import add_engineered_features


def load_artifacts(
    model_path: str | Path = config.BEST_MODEL_FILE,
) -> tuple[object, object]:
    """Load the saved pipeline and return (pipeline, preprocessor).

    The saved artifact is a full sklearn Pipeline (preprocessing + estimator),
    so the preprocessor is embedded; the second return value is the fitted
    ColumnTransformer extracted from the pipeline for inspection.
    """
    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps["preprocess"]
    return pipeline, preprocessor


def predict_single(features: dict) -> float:
    """Predict the price for one house described by a feature dict.

    Args:
        features: Dict with the 12 raw input columns (everything in
            data/housing.csv except 'price').

    Returns:
        Predicted price as a float.
    """
    pipeline, _ = load_artifacts()
    df = pd.DataFrame([features])
    df = add_engineered_features(df)
    return float(pipeline.predict(df)[0])
