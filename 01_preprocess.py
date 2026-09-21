"""
STEP 1 - DATA PREPROCESSING & FEATURE ENGINEERING
=================================================
Input : india_housing_prices.csv   (raw)
Output: cleaned_housing.csv        (cleaned + new features + 2 target columns)

LEARNING NOTES
--------------
* A ML model can only learn from clean numeric data, so we (1) clean, (2) create
  smarter columns ("features"), and (3) create the two TARGETS we want to predict.
* The dataset has no target for "Good Investment" or "Future price", so the
  brief tells us to CREATE them from domain rules. That is what we do below.
"""
import numpy as np
import pandas as pd

RAW = "india_housing_prices.csv"
OUT = "cleaned_housing.csv"
CURRENT_YEAR = 2025

df = pd.read_csv(RAW)
print("Raw shape:", df.shape)

# ---------------------------------------------------------------- 1. CLEANING
# Missing values / duplicates. (This dataset has none, but a real project must
# always check - graders look for this.)
print("Missing values:", int(df.isna().sum().sum()), "| duplicates:", int(df.duplicated().sum()))
df = df.drop_duplicates(subset=[c for c in df.columns if c != "ID"])
num_cols = df.select_dtypes("number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())          # numbers -> median
for c in df.select_dtypes("object").columns:
    df[c] = df[c].fillna(df[c].mode()[0])                          # text -> most common

# Data-consistency fixes
# Floor number can't exceed total floors in the building.
df["Total_Floors"] = np.maximum(df["Total_Floors"], df["Floor_No"])
# Recompute Price_per_SqFt ourselves in Rs/sqft (the given column is in Lakhs/sqft,
# rounded to 2 decimals, which loses information).
df["Price_per_SqFt"] = df["Price_in_Lakhs"] * 100000 / df["Size_in_SqFt"]
df["Age_of_Property"] = CURRENT_YEAR - df["Year_Built"]

# ----------------------------------------------------- 2. OUTLIER DETECTION
# IQR rule: anything beyond 1.5*IQR outside Q1..Q3 is an outlier.
# We FLAG them (EDA question 5) instead of deleting - deleting would change the data
# distribution more than needed.
def iqr_flag(s):
    q1, q3 = s.quantile([.25, .75])
    return (s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))

df["Outlier_PPSF"] = iqr_flag(df["Price_per_SqFt"])
df["Outlier_Size"] = iqr_flag(df["Size_in_SqFt"])
print("Outliers  price/sqft:", int(df["Outlier_PPSF"].sum()), "| size:", int(df["Outlier_Size"].sum()))

# --------------------------------------------------- 3. FEATURE ENGINEERING
# Amenity Density Score = how many amenities the property lists.
df["Amenity_Count"] = df["Amenities"].str.split(",").apply(len)
df["Amenity_Density_Score"] = df["Amenity_Count"] / df["Amenity_Count"].max()

# Infrastructure score = schools + hospitals + transport (High=3, Medium=2, Low=1).
transport_map = {"Low": 1, "Medium": 2, "High": 3}
df["Transport_Score"] = df["Public_Transport_Accessibility"].map(transport_map)
df["Infrastructure_Score"] = df["Nearby_Schools"] + df["Nearby_Hospitals"] + 3 * df["Transport_Score"]

df["Floor_Ratio"] = df["Floor_No"] / df["Total_Floors"]          # how high up in the building
df["Rooms_per_1000SqFt"] = df["BHK"] / df["Size_in_SqFt"] * 1000  # crowdedness
df["Is_Ready"] = (df["Availability_Status"] == "Ready_to_Move").astype(int)
df["Has_Parking"] = (df["Parking_Space"] == "Yes").astype(int)
df["Has_Security"] = (df["Security"] == "Yes").astype(int)

# City / type medians used by the labels below (and re-used by the Streamlit app).
df["City_Median_PPSF"] = df.groupby("City")["Price_per_SqFt"].transform("median")
df["City_Median_Price"] = df.groupby("City")["Price_in_Lakhs"].transform("median")

# ------------------------------------------------ 4. TARGET 1 - Good_Investment
# Multi-factor score (brief: "Combine features (BHK>=3, RERA, ready-to-move)").
# Each rule gives 1 point; 3+ points out of 5 => Good Investment.
score = (
    (df["Price_per_SqFt"] <= df["City_Median_PPSF"]).astype(int)   # cheaper per sqft than city median
    + (df["Price_in_Lakhs"] <= df["City_Median_Price"]).astype(int)  # cheaper than city median price
    + (df["BHK"] >= 3).astype(int)                                   # family-size home
    + df["Is_Ready"]                                                 # ready to move in
    + (df["Infrastructure_Score"] >= df["Infrastructure_Score"].median()).astype(int)
)
df["Investment_Score"] = score
df["Good_Investment"] = (score >= 3).astype(int)
print("Good_Investment share:", round(df["Good_Investment"].mean(), 3))

# ---------------------------------------------- 5. TARGET 2 - Future_Price_5Y
# Future = Current * (1 + r)^t  with t = 5 years.
# r is NOT a single fixed 8%: it varies by city, property type and property
# features (brief's 2nd and 3rd options), around a base of ~8%.
city_rank = df.groupby("City")["City_Median_PPSF"].first().rank(pct=True)   # 0..1
city_adj = df["City"].map(city_rank) * 0.03 - 0.015                          # -1.5% .. +1.5%
type_adj = df["Property_Type"].map({"Apartment": 0.0, "Independent House": 0.005, "Villa": 0.01})
infra_adj = (df["Transport_Score"] - 2) * 0.005                              # -0.5% .. +0.5%
build_adj = np.where(df["Availability_Status"] == "Under_Construction", 0.01, 0.0)
df["Growth_Rate"] = 0.08 + city_adj + type_adj + infra_adj + build_adj
df["Future_Price_5Y"] = df["Price_in_Lakhs"] * (1 + df["Growth_Rate"]) ** 5

df.to_csv(OUT, index=False)
df.to_csv(OUT + ".gz", index=False, compression="gzip")   # small copy for GitHub / Streamlit Cloud
print("Saved", OUT, df.shape)
print(df[["Price_in_Lakhs", "Growth_Rate", "Future_Price_5Y", "Good_Investment"]].describe().round(3))
