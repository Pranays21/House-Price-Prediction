"""Central configuration for the house price prediction project.

All magic numbers live here so training, evaluation and the demo app stay
consistent.
"""

TARGET_COL = "price"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
OUTLIER_IQR_MULTIPLIER = 1.5

# Columns by role (must match data/housing.csv schema)
NUMERIC_FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking"]
CATEGORICAL_FEATURES = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]

# Model registry: name -> (estimator, param_grid for tuning)
MODEL_PARAMS = {
    "ridge": {"alpha": [0.1, 1.0, 10.0, 100.0]},
    "lasso": {"alpha": [1e3, 1e4, 1e5, 1e6]},
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
    },
    "gradient_boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    },
}

MODEL_DISPLAY_NAMES = {
    "linear": "Linear Regression",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}

ARTIFACT_DIR = "models"
BEST_MODEL_FILE = "models/best_model.pkl"
PREPROCESSOR_FILE = "models/preprocessor.pkl"
LEADERBOARD_FILE = "models/leaderboard.csv"
