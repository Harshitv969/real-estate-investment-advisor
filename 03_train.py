"""
STEP 3 + 4 - MODEL TRAINING, EVALUATION & MLFLOW TRACKING
=========================================================
Trains 5 classifiers (Good_Investment) and 5 regressors (Future_Price_5Y),
logs every run to MLflow, registers the best model of each task in the
MLflow Model Registry and saves them to models/ for the Streamlit app.

LEARNING NOTES
--------------
* train/test split: we hide 20% of the data from the model and score on it, to
  measure how it performs on properties it has never seen.
* Pipeline = preprocessing + model glued together, so the SAME scaling/encoding
  is applied at prediction time in the app (no mismatch bugs).
* StandardScaler: puts numeric columns on a similar scale (needed by linear models).
* OneHotEncoder: turns "City = Pune" into 0/1 columns, since models need numbers.
* Classification metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
* Regression metrics: MSE, RMSE, MAE, R2 (1.0 = perfect, 0 = no better than the mean).
"""
import json
import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

SEED = 42
os.makedirs("models", exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlflow.db")          # local DB, also enables the model registry
mlflow.set_experiment("Real_Estate_Investment_Advisor")

df = pd.read_csv("cleaned_housing.csv")

NUM = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property", "Nearby_Schools",
       "Nearby_Hospitals", "Floor_No", "Total_Floors", "Amenity_Density_Score", "Transport_Score",
       "Infrastructure_Score", "Floor_Ratio", "Rooms_per_1000SqFt", "Is_Ready", "Has_Parking", "Has_Security"]
CAT = ["City", "Property_Type", "Furnished_Status", "Facing", "Owner_Type"]
# NOTE (data leakage): we deliberately leave out Investment_Score, Growth_Rate and
# City_Median_* - they are used to BUILD the targets, so feeding them in would let the
# model "cheat".
X = df[NUM + CAT]
json.dump({"NUM": NUM, "CAT": CAT}, open("models/feature_lists.json", "w"))


def prep():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])


# ------------------------------------------------------------------ CLASSIFICATION
y_c = df["Good_Investment"]
Xtr, Xte, ytr, yte = train_test_split(X, y_c, test_size=0.2, random_state=SEED, stratify=y_c)
classifiers = {
    "LogisticRegression": (LogisticRegression(max_iter=1000), {"max_iter": 1000}),
    "DecisionTree": (DecisionTreeClassifier(max_depth=10, random_state=SEED), {"max_depth": 10}),
    "RandomForest": (RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=SEED),
                     {"n_estimators": 100, "max_depth": 12}),
    "HistGradientBoosting": (HistGradientBoostingClassifier(max_iter=200, random_state=SEED), {"max_iter": 200}),
    "XGBoost": (XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=SEED),
                {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}),
}
results_c = {}
for name, (model, params) in classifiers.items():
    with mlflow.start_run(run_name=f"clf_{name}") as run:
        pipe = Pipeline([("prep", prep()), ("model", model)]).fit(Xtr, ytr)
        pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
        m = {"accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred),
             "recall": recall_score(yte, pred), "f1": f1_score(yte, pred), "roc_auc": roc_auc_score(yte, proba)}
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        mlflow.set_tags({"task": "classification"})
        mlflow.log_params({"model": name, **params})
        mlflow.log_metrics(m)
        mlflow.log_dict({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}, "confusion_matrix.json")
        info = mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
        results_c[name] = {**m, "cm": [[int(tn), int(fp)], [int(fn), int(tp)]], "uri": info.model_uri, "pipe": pipe}
        print(f"[CLF] {name:22s} " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))

# ---------------------------------------------------------------------- REGRESSION
y_r = df["Future_Price_5Y"]
Xtr, Xte, ytr, yte = train_test_split(X, y_r, test_size=0.2, random_state=SEED)
regressors = {
    "LinearRegression": (LinearRegression(), {}),
    "Ridge": (Ridge(alpha=1.0), {"alpha": 1.0}),
    "DecisionTree": (DecisionTreeRegressor(max_depth=12, random_state=SEED), {"max_depth": 12}),
    "RandomForest": (RandomForestRegressor(n_estimators=60, max_depth=12, n_jobs=-1, random_state=SEED),
                     {"n_estimators": 60, "max_depth": 12}),
    "HistGradientBoosting": (HistGradientBoostingRegressor(max_iter=300, random_state=SEED), {"max_iter": 300}),
    "XGBoost": (XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=SEED),
                {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1}),
}
results_r = {}
for name, (model, params) in regressors.items():
    with mlflow.start_run(run_name=f"reg_{name}"):
        pipe = Pipeline([("prep", prep()), ("model", model)]).fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        mse = mean_squared_error(yte, pred)
        m = {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": mean_absolute_error(yte, pred), "r2": r2_score(yte, pred)}
        mlflow.set_tags({"task": "regression"})
        mlflow.log_params({"model": name, **params})
        mlflow.log_metrics(m)
        info = mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
        results_r[name] = {**m, "uri": info.model_uri, "pipe": pipe}
        print(f"[REG] {name:22s} " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))

# ------------------------------------------------- PICK BEST, REGISTER, SAVE FOR APP
best_c = max(results_c, key=lambda k: results_c[k]["f1"])          # best classifier = highest F1
best_r = min(results_r, key=lambda k: results_r[k]["rmse"])        # best regressor  = lowest RMSE
client = MlflowClient()
for reg_name, best, res in [("RealEstate_GoodInvestment_Classifier", best_c, results_c),
                            ("RealEstate_FuturePrice_Regressor", best_r, results_r)]:
    mv = mlflow.register_model(res[best]["uri"], reg_name)
    client.set_registered_model_alias(reg_name, "production", mv.version)   # alias replaces old "stages"
    print(f"Registered {reg_name} v{mv.version} <- {best} (alias: production)")

joblib.dump(results_c[best_c]["pipe"], "models/best_classifier.joblib")
joblib.dump(results_r[best_r]["pipe"], "models/best_regressor.joblib")

# Feature importance = permutation importance: shuffle one column at a time and see how
# much the score drops. Big drop => the model relies on that feature. Works for ANY model
# and reports the original column names (City, BHK...) rather than one-hot pieces.
from sklearn.inspection import permutation_importance
def importance(pipe, Xs, ys, scoring):
    idx = Xs.sample(8000, random_state=SEED).index
    r = permutation_importance(pipe, Xs.loc[idx], ys.loc[idx], scoring=scoring, n_repeats=3,
                               random_state=SEED, n_jobs=1)
    return pd.Series(r.importances_mean, index=Xs.columns).sort_values(ascending=False)

importance(results_c[best_c]["pipe"], X, y_c, "f1").head(15).to_csv("models/importance_classifier.csv", header=["importance"])
importance(results_r[best_r]["pipe"], X, y_r, "r2").head(15).to_csv("models/importance_regressor.csv", header=["importance"])

summary = {"best_classifier": best_c, "best_regressor": best_r,
           "classification": {k: {m: v for m, v in r.items() if m not in ("uri", "pipe")} for k, r in results_c.items()},
           "regression": {k: {m: v for m, v in r.items() if m not in ("uri", "pipe")} for k, r in results_r.items()}}
json.dump(summary, open("models/metrics_summary.json", "w"), indent=2)
print("\nBest classifier:", best_c, "| Best regressor:", best_r)
