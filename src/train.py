"""Train and compare regression models for house price prediction.

Run from the project root::

    python -m src.train

Prints a leaderboard (RMSE / MAE / R2), tunes the top-2 models with
GridSearchCV, and saves the best pipeline plus the leaderboard to models/.
"""
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

# Lasso on million-scale targets triggers benign duality-gap warnings for tiny
# alphas; the models still converge to essentially OLS solutions.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from config import MODEL_DISPLAY_NAMES
from src import data_loader, evaluate, features, preprocessing
from src.features import ENGINEERED_NUMERIC

BASE_ESTIMATORS: dict[str, object] = {
    "linear": LinearRegression(),
    "ridge": Ridge(random_state=config.RANDOM_STATE),
    "lasso": Lasso(random_state=config.RANDOM_STATE, max_iter=5000),
    "random_forest": RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1),
    "gradient_boosting": GradientBoostingRegressor(random_state=config.RANDOM_STATE),
}


def make_pipeline(estimator: object) -> Pipeline:
    """Wrap the shared preprocessor and an estimator in a Pipeline."""
    numeric = config.NUMERIC_FEATURES + ENGINEERED_NUMERIC
    pre = preprocessing.build_preprocessor(
        numeric_features=numeric,
        categorical_features=config.CATEGORICAL_FEATURES,
    )
    return Pipeline([("preprocess", pre), ("model", estimator)])


def main() -> None:
    print("Loading data...")
    df = data_loader.load_data("data/housing.csv")
    print("Checks:", data_loader.basic_checks(df))

    df = preprocessing.cap_outliers_iqr(df, config.NUMERIC_FEATURES)
    df = features.add_engineered_features(df)

    print("\nTop correlations with price:")
    print(features.correlation_with_target(df).round(3).to_string())

    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    results: list[dict] = []
    fitted: dict[str, Pipeline] = {}
    for key, estimator in BASE_ESTIMATORS.items():
        pipe = make_pipeline(estimator)
        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=config.CV_FOLDS,
            scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        pipe.fit(X_train, y_train)
        metrics = evaluate.all_metrics(y_test.values, pipe.predict(X_test))
        results.append({
            "model": MODEL_DISPLAY_NAMES[key],
            "key": key,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "cv_rmse": float(-cv_scores.mean()),
        })
        fitted[key] = pipe
        print(f"  {MODEL_DISPLAY_NAMES[key]:20s} R2={metrics['r2']:.4f} "
              f"RMSE={metrics['rmse']:,.0f} CV_RMSE={-cv_scores.mean():,.0f}")

    # Tune the top-2 models by R2
    board = evaluate.leaderboard_table(results)
    top2 = board.head(2)["key"].tolist()
    print("\nTuning top-2 models with GridSearchCV...")
    for key in top2:
        if key not in config.MODEL_PARAMS:
            continue
        grid = GridSearchCV(
            make_pipeline(BASE_ESTIMATORS[key]),
            param_grid={f"model__{k}": v for k, v in config.MODEL_PARAMS[key].items()},
            cv=config.CV_FOLDS,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        metrics = evaluate.all_metrics(y_test.values, grid.predict(X_test))
        name = MODEL_DISPLAY_NAMES[key] + " (tuned)"
        results.append({
            "model": name, "key": key + "_tuned",
            "rmse": metrics["rmse"], "mae": metrics["mae"], "r2": metrics["r2"],
            "cv_rmse": float(-grid.best_score_),
        })
        fitted[key + "_tuned"] = grid.best_estimator_
        print(f"  {name:28s} R2={metrics['r2']:.4f} RMSE={metrics['rmse']:,.0f} "
              f"best_params={grid.best_params_}")

    board = evaluate.leaderboard_table(results)
    print("\n================ LEADERBOARD ================")
    print(board[["model", "rmse", "mae", "r2", "cv_rmse"]].to_string(index=False))
    print("===========================================")

    # Save artifacts
    Path(config.ARTIFACT_DIR).mkdir(exist_ok=True)
    best_key = board.iloc[0]["key"]
    best_pipe = fitted[best_key]
    joblib.dump(best_pipe, config.BEST_MODEL_FILE)
    joblib.dump(best_pipe.named_steps["preprocess"], config.PREPROCESSOR_FILE)
    board.to_csv(config.LEADERBOARD_FILE, index=False)
    print(f"\nBest model: {board.iloc[0]['model']} "
          f"(R2={board.iloc[0]['r2']:.4f}) saved to {config.BEST_MODEL_FILE}")


if __name__ == "__main__":
    main()
