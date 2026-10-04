# Dataset

## housing.csv (synthetic)

`housing.csv` is a **synthetically generated** dataset with 545 rows and 13 columns,
matching the schema of the Kaggle **"Housing Prices Dataset"** (`yasserh/housing-prices-dataset`)
so the project runs out of the box without Kaggle API credentials.

To use the real data, download it from Kaggle and replace this file — the
pipeline accepts any CSV with the same columns and a `price` target column.

### Schema

| Column | Type | Description |
|---|---|---|
| `price` | int | Target — house sale price |
| `area` | int | Property area in square feet |
| `bedrooms` | int | Number of bedrooms (1-6) |
| `bathrooms` | int | Number of bathrooms (1-4) |
| `stories` | int | Number of floors (1-4) |
| `mainroad` | yes/no | Main road access |
| `guestroom` | yes/no | Guest room availability |
| `basement` | yes/no | Basement availability |
| `hotwaterheating` | yes/no | Hot water heating |
| `airconditioning` | yes/no | Air conditioning |
| `parking` | int | Number of parking spaces (0-3) |
| `prefarea` | yes/no | Located in preferred area |
| `furnishingstatus` | furnished / semi-furnished / unfurnished | Furnishing level |

### Generation notes

Price was generated as a function of `area`, `bathrooms`, `stories`,
`airconditioning`, `prefarea` and `parking` plus Gaussian noise, so trained
regressors reach R² ≈ 0.6–0.7. Seed = 42 for reproducibility.
