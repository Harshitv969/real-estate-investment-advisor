# Real Estate Investment Advisor: Predicting Property Profitability & Future Value

Two ML models on `india_housing_prices.csv` (250,000 properties, 23 columns):
1. **Classification**: is a property a *Good Investment*?
2. **Regression**: what will its price be *after 5 years*?

A Streamlit app serves both; MLflow tracks all experiments and registers the best models.

**Live app:** https://rea-invest-advisor-guvi.streamlit.app/
**Full EDA report (all 20 questions, charted):** [EDA.md](EDA.md)

## How to run
```bash
pip install -r requirements.txt
python 01_preprocess.py      # raw csv  -> cleaned_housing.csv (features + targets)
python 02_eda.py             # 20 EDA charts -> eda_plots/, numbers -> eda_findings.txt
python 03_train.py           # trains 5 classifiers + 6 regressors, logs to MLflow, saves models/
streamlit run app.py         # the app
mlflow ui --backend-store-uri sqlite:///mlflow.db   # experiment tracking UI + model registry
```

## Methodology
**Preprocessing**: no missing values/duplicates were found (checks still coded, with median/mode imputation).
`Total_Floors` forced >= `Floor_No`; `Price_per_SqFt` recomputed in Rs/sqft (original was in Lakhs, rounded to 2 dp).
Outliers flagged with the IQR rule (7.9% of price/sqft; none in size). Numeric features standardised, categoricals one-hot encoded (inside the sklearn Pipeline).

**Engineered features**: Amenity_Density_Score, Transport_Score, Infrastructure_Score (schools + hospitals + 3 x transport), Floor_Ratio, Rooms_per_1000SqFt, Is_Ready, Has_Parking, Has_Security, Age_of_Property.

**Target 1, Good_Investment**: multi-factor score, 1 point each for: price/sqft <= city median, price <= city median, BHK >= 3, ready-to-move, infrastructure score >= median. Score >= 3 => good (54% of properties).

**Target 2, Future_Price_5Y** = `Price x (1 + r)^5`, where r = 8% base +/- city adjustment (city price rank, -1.5%..+1.5%) + property-type adjustment (villa +1%, house +0.5%) + transport (+/-0.5%) + under-construction (+1%). r ranges 6.1%-12%.

**Models**: Classifiers: Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting, XGBoost. Regressors: Linear, Ridge, Decision Tree, Random Forest, HistGradientBoosting, XGBoost. 80/20 split, seed 42. Best classifier chosen by F1, best regressor by RMSE.

## Results (20% hold-out test set)
| Classifier | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.889 | 0.898 | 0.963 |
| Decision Tree | 0.996 | 0.996 | 0.999 |
| Random Forest | 0.996 | 0.996 | 1.000 |
| **HistGradientBoosting** | **0.999** | **0.999** | **1.000** |
| XGBoost | 0.998 | 0.999 | 1.000 |

| Regressor | RMSE (Lakhs) | MAE | R2 |
|---|---|---|---|
| Linear / Ridge | 11.49 | 8.13 | 0.997 |
| Decision Tree | 17.18 | 12.62 | 0.994 |
| Random Forest | 15.50 | 11.97 | 0.995 |
| HistGradientBoosting | 1.82 | 1.40 | 0.9999 |
| **XGBoost** | **1.69** | **1.29** | **0.9999** |

Full metrics (precision, recall, confusion matrices, MSE): `models/metrics_summary.json`.

## Key findings (EDA) and honest caveats
* **The dataset looks synthetic.** Price barely correlates with size, schools, hospitals, furnishing, parking, facing or amenities (all |r| < 0.01), and prices/sizes are near-uniform. Avg price per sq ft is ~Rs 13,000 in every state/type. City-level differences are tiny (e.g. avg price Rs 251-259 L).
* Public transport is the one visible relationship: good-investment share is 46% (Low), 55% (Medium), 62% (High), because it is part of our infrastructure score.
* **Near-perfect scores are expected, not magic.** Both targets are *derived from* dataset columns by rules we defined, so the models are re-learning those rules (permutation importance confirms this: price, BHK, ready-to-move, infrastructure for the classifier; current price for the regressor). Scores would be lower on a real-world "good investment" label from actual sale outcomes.
* To limit leakage, the score itself, the growth rate and the city medians used to build the labels are *not* model inputs.
* The brief mentions crime rate; the dataset has no such column. Infrastructure score is used as the closest proxy (`eda_plots/extra_infra_vs_good.png`).
* No sale dates exist, so "price trends" (Q10) use Year_Built as a proxy time axis.

## Deliverables map
| Deliverable | File |
|---|---|
| Cleaned dataset | `cleaned_housing.csv` |
| EDA / training scripts | `01_preprocess.py`, `02_eda.py`, `03_train.py` |
| MLflow logs + registry | `mlflow.db`, `mlruns/` (models `RealEstate_GoodInvestment_Classifier`, `RealEstate_FuturePrice_Regressor`, alias `production`) |
| Streamlit app | `app.py` |
| Documentation | this file |

## Use cases
Investors screening listings for long-term return; buyers comparing localities; real-estate platforms auto-scoring listings with a confidence score and 5-year price forecast.

## Deployment (Streamlit Community Cloud)
Push the repo to GitHub (raw CSV, `mlruns/` and `mlflow.db` are git-ignored; the app only needs `cleaned_housing.csv.gz` and `models/`). On share.streamlit.io choose the repo and `app.py` as the main file. Regenerate everything locally with the scripts above if needed.
