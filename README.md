# 🏠 House Price Prediction using Regression Models

Predict residential property prices from structured housing data. The pipeline
covers preprocessing, feature scaling, correlation analysis and outlier handling,
then trains and compares multiple regression algorithms using RMSE, MAE and R².

## Dataset

`data/housing.csv` — 545 rows × 13 columns, schema-compatible with the Kaggle
**"Housing Prices Dataset"** (`yasserh/housing-prices-dataset`). The bundled file
is synthetically generated (see `data/README.md`); replace it with the real
Kaggle CSV to train on real data — no code changes needed.

Target: `price`. Features: `area`, `bedrooms`, `bathrooms`, `stories`,
`parking`, `mainroad`, `guestroom`, `basement`, `hotwaterheating`,
`airconditioning`, `prefarea`, `furnishingstatus`.

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python -m src.train
```

This prints the correlation summary and a leaderboard (RMSE / MAE / R², plus
5-fold CV RMSE), tunes the top-2 models with GridSearchCV, and saves
`models/best_model.pkl`, `models/preprocessor.pkl` and `models/leaderboard.csv`.

### Results (synthetic data, seed 42)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| *Run `python -m src.train` to fill in* | | | |

Expected: tree/regularized models reach R² ≈ 0.6–0.7 on the synthetic data.

## Demo app

```bash
streamlit run app.py
```

Enter property features in the sidebar form and get an instant price estimate,
plus the training leaderboard.

## Project structure

```
├── app.py                 # Streamlit price-prediction demo
├── config.py              # All hyper-parameters & paths (no magic numbers)
├── data/
│   ├── housing.csv        # Dataset (synthetic, Kaggle-compatible schema)
│   └── README.md
├── src/
│   ├── data_loader.py     # Load + validate + sanity checks
│   ├── preprocessing.py   # IQR capping, ColumnTransformer (impute/scale/encode)
│   ├── features.py        # total_rooms, area_per_room, correlation helper
│   ├── train.py           # Train/compare/tune, print leaderboard, save artifacts
│   ├── evaluate.py        # RMSE, MAE, R², leaderboard & residual helpers
│   └── predict.py         # predict_single(dict) -> float
├── models/                # best_model.pkl, preprocessor.pkl, leaderboard.csv
├── tests/                 # pytest suite
├── Dockerfile
└── requirements.txt
```

## Tests

```bash
pytest tests/ -v
```
