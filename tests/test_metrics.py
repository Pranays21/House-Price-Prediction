"""Tests for src/evaluate.py metrics."""
import numpy as np
import pandas as pd

from src.evaluate import all_metrics, leaderboard_table, mae, r2, residual_data, rmse


def test_metrics_on_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0
    assert r2(y, y) == 1.0


def test_all_metrics_keys():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.5, 2.5, 2.5, 3.5])
    m = all_metrics(y, p)
    assert set(m) == {"rmse", "mae", "r2"}
    assert m["rmse"] > 0 and m["mae"] > 0 and m["r2"] < 1.0


def test_leaderboard_sorted_by_r2():
    rows = [
        {"model": "a", "rmse": 5.0, "mae": 4.0, "r2": 0.5},
        {"model": "b", "rmse": 3.0, "mae": 2.0, "r2": 0.8},
    ]
    lb = leaderboard_table(rows)
    assert list(lb["model"]) == ["b", "a"]


def test_residual_data_shape():
    y = np.array([1.0, 2.0])
    p = np.array([1.2, 1.8])
    df = residual_data(y, p)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["predicted", "residual"]
    assert len(df) == 2
