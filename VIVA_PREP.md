# Viva / Live Evaluation Prep

## 60-second pitch
"I built an advisor for property investors. Using 250,000 Indian listings I trained one model that says whether a property is a Good Investment and another that predicts its price after 5 years. I cleaned the data, engineered features, compared 5 classifiers and 6 regressors, tracked everything in MLflow, and served the best models in a Streamlit app with filters, charts, confidence scores and feature importance."

## Likely questions and answers
**Where did the labels come from?** The dataset has none, so I created them from domain rules as the brief suggests. Good_Investment = a 5-point score (price/sqft and price below city median, BHK >= 3, ready-to-move, infrastructure above median); 3+ points = good. Future price = Price x (1+r)^5 with r about 8%, adjusted by city, property type, transport access and construction status.

**Why is accuracy ~99.9%? Isn't that overfitting?** No. It is measured on a 20% test set the model never saw. It is high because the labels are rule-based functions of the columns, so the model re-learns my rules. Real labels (actual resale outcomes) would give lower scores. I left the score, growth rate and city medians out of the inputs to limit leakage.

**What is data leakage?** Giving the model information that would not be available at prediction time, or that directly encodes the target.

**Why did Logistic Regression score lower (0.889)?** It is linear, but the rule uses thresholds and comparisons against medians, which trees capture better.

**Why HistGradientBoosting / XGBoost won?** Boosted trees fit thresholds and interactions well. Regression winner: XGBoost, RMSE 1.69 Lakhs, R2 0.9999. Linear models had RMSE 11.5 because growth is multiplicative and varies by city/type.

**Which metric did you pick the best model with?** F1 for classification (balances precision and recall), RMSE for regression (same units as price, penalises big errors).

**Explain precision vs recall.** Precision: of properties I call good, how many really are. Recall: of all truly good properties, how many I found.

**Why scale and one-hot encode?** Linear models need similar scales; models need numbers, so categories like City become 0/1 columns. Both are inside a Pipeline so the app applies identical transforms.

**How did you handle outliers?** IQR rule; 7.9% of price/sqft flagged, none in size. I flagged rather than deleted them.

**What does MLflow do here?** Logs parameters, metrics and the model of every run, lets me compare runs in the UI, and the Model Registry holds the best models under the alias `production`.

**What did EDA show?** Price hardly depends on size, schools, furnishing, parking or facing (correlations about 0), which suggests synthetic data. Public transport is the visible driver of good-investment share (46% Low, 55% Medium, 62% High), because it feeds the infrastructure score.

**Limitations?** Synthetic-looking data; rule-based labels; no crime-rate or sale-date columns (used infrastructure score and Year_Built as proxies); constant growth assumption.

**What would you do next?** Use real historical prices for the target, add crime/demand data, tune hyperparameters, add SHAP explanations.

## Demo order (3 minutes)
1. Predict tab: enter a property, show verdict, confidence, 5-year price. 2. Explore tab: filter by city/BHK/price. 3. Insights: heatmap. 4. Model performance: comparison table + importance. 5. MLflow UI: runs and registry.
