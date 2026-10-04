"""Evaluation metrics and model-comparison helpers."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    return float(r2_score(y_true, y_pred))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE and R2 in one call."""
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred), "r2": r2(y_true, y_pred)}


def leaderboard_table(results: list[dict]) -> pd.DataFrame:
    """Build a sorted leaderboard DataFrame from per-model result dicts.

    Each dict must have keys: model, rmse, mae, r2 (and optionally cv_rmse).
    Sorted by R2 descending.
    """
    df = pd.DataFrame(results)
    return df.sort_values("r2", ascending=False).reset_index(drop=True)


def residual_data(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Return a DataFrame of predicted values and residuals for plotting."""
    return pd.DataFrame(
        {"predicted": np.asarray(y_pred), "residual": np.asarray(y_true) - np.asarray(y_pred)}
    )
