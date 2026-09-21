"""
STEP 2 - EXPLORATORY DATA ANALYSIS (EDA)
========================================
Answers the 20 questions in the brief. One PNG per question -> eda_plots/
Also writes eda_findings.txt with the numbers behind each chart (use in your docs/viva).

LEARNING NOTES
--------------
EDA = "look at the data before modelling". It tells you which features matter,
whether there are outliers, and whether your assumptions hold.
Typical chart choices: histogram (one numeric column), boxplot (numeric by category),
bar (average by category), scatter (two numerics), heatmap (correlations).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

os.makedirs("eda_plots", exist_ok=True)
df = pd.read_csv("cleaned_housing.csv")
sample = df.sample(5000, random_state=42)          # scatter plots of 250k points are unreadable
sns.set_theme(style="whitegrid")
notes = []

def save(name, title, note=None):
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"eda_plots/{name}.png", dpi=110)
    plt.close()
    if note:
        notes.append(f"{title}\n{note}\n")

# ---- 1-5 Price & size ------------------------------------------------------
sns.histplot(df["Price_in_Lakhs"], bins=50)
save("q01_price_dist", "Q1 Distribution of property prices (Lakhs)",
     f"mean={df.Price_in_Lakhs.mean():.1f}, median={df.Price_in_Lakhs.median():.1f}, range {df.Price_in_Lakhs.min()}-{df.Price_in_Lakhs.max()}")

sns.histplot(df["Size_in_SqFt"], bins=50)
save("q02_size_dist", "Q2 Distribution of property sizes (sq ft)",
     f"mean={df.Size_in_SqFt.mean():.0f}, range {df.Size_in_SqFt.min()}-{df.Size_in_SqFt.max()}")

sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt", showfliers=False)
save("q03_ppsf_by_type", "Q3 Price per sq ft (Rs) by property type",
     df.groupby("Property_Type")["Price_per_SqFt"].mean().round(0).to_string())

sns.scatterplot(data=sample, x="Size_in_SqFt", y="Price_in_Lakhs", alpha=.3)
save("q04_size_vs_price", "Q4 Size vs price",
     f"correlation = {df.Size_in_SqFt.corr(df.Price_in_Lakhs):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(y=df["Price_per_SqFt"], ax=ax[0]); ax[0].set_title("Price per sq ft")
sns.boxplot(y=df["Size_in_SqFt"], ax=ax[1]); ax[1].set_title("Size in sq ft")
save("q05_outliers", "Q5 Outliers (IQR rule)",
     f"price/sqft outliers = {int(df.Outlier_PPSF.sum())} ({df.Outlier_PPSF.mean():.1%}); size outliers = {int(df.Outlier_Size.sum())}")

# ---- 6-10 Location ---------------------------------------------------------
s = df.groupby("State")["Price_per_SqFt"].mean().sort_values()
s.plot.barh(figsize=(8, 7)); save("q06_ppsf_by_state", "Q6 Avg price per sq ft (Rs) by state",
    f"highest: {s.index[-1]} ({s.iloc[-1]:.0f}); lowest: {s.index[0]} ({s.iloc[0]:.0f})")

c = df.groupby("City")["Price_in_Lakhs"].mean().sort_values()
c.plot.barh(figsize=(8, 10)); save("q07_price_by_city", "Q7 Average price (Lakhs) by city",
    f"highest: {c.index[-1]} ({c.iloc[-1]:.1f}); lowest: {c.index[0]} ({c.iloc[0]:.1f})")

a = df.groupby("Locality")["Age_of_Property"].median().sort_values()
a.head(40).plot.barh(figsize=(8, 9)); save("q08_median_age_locality", "Q8 Median age of properties (40 youngest of 500 localities)",
    f"overall median of locality medians = {a.median():.1f} yrs; range {a.min():.1f}-{a.max():.1f}")

pd.crosstab(df["City"], df["BHK"]).plot.bar(stacked=True, figsize=(13, 5))
save("q09_bhk_by_city", "Q9 BHK distribution across cities",
     "BHK is spread almost uniformly (about 20% each of 1-5 BHK) in every city.")

top = df.groupby("Locality")["Price_per_SqFt"].mean().nlargest(5).index
t = df[df.Locality.isin(top)].groupby(["Year_Built", "Locality"])["Price_per_SqFt"].mean().unstack()
t.plot(figsize=(10, 5), marker="o", ms=3)
save("q10_top5_locality_trend", "Q10 Price/sq ft trend by year built, top-5 priciest localities",
     "Top 5: " + ", ".join(top) + ". The dataset has no sale dates, so Year_Built is the only time axis (proxy).")

# ---- 11-15 Relationships ---------------------------------------------------
num = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property", "Nearby_Schools",
       "Nearby_Hospitals", "Floor_No", "Total_Floors", "Infrastructure_Score", "Amenity_Density_Score"]
cm = df[num].corr()
plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", center=0)
save("q11_correlation", "Q11 Correlation of numeric features",
     "Strongest with Price_in_Lakhs:\n" + cm["Price_in_Lakhs"].drop("Price_in_Lakhs").abs().sort_values(ascending=False).head(4).round(3).to_string())

sns.barplot(data=df, x="Nearby_Schools", y="Price_per_SqFt")
save("q12_schools_vs_ppsf", "Q12 Nearby schools vs price per sq ft",
     f"correlation = {df.Nearby_Schools.corr(df.Price_per_SqFt):.3f}")
sns.barplot(data=df, x="Nearby_Hospitals", y="Price_per_SqFt")
save("q13_hospitals_vs_ppsf", "Q13 Nearby hospitals vs price per sq ft",
     f"correlation = {df.Nearby_Hospitals.corr(df.Price_per_SqFt):.3f}")

sns.boxplot(data=df, x="Furnished_Status", y="Price_in_Lakhs")
save("q14_furnished_vs_price", "Q14 Price by furnished status",
     df.groupby("Furnished_Status")["Price_in_Lakhs"].mean().round(1).to_string())
sns.barplot(data=df, x="Facing", y="Price_per_SqFt")
save("q15_facing_vs_ppsf", "Q15 Price per sq ft by facing direction",
     df.groupby("Facing")["Price_per_SqFt"].mean().round(0).to_string())

# ---- 16-20 Ownership / amenities ------------------------------------------
df["Owner_Type"].value_counts().plot.bar()
save("q16_owner_type", "Q16 Properties by owner type", df["Owner_Type"].value_counts().to_string())
df["Availability_Status"].value_counts().plot.bar()
save("q17_availability", "Q17 Properties by availability status", df["Availability_Status"].value_counts().to_string())
sns.boxplot(data=df, x="Parking_Space", y="Price_in_Lakhs")
save("q18_parking_vs_price", "Q18 Parking space vs price",
     df.groupby("Parking_Space")["Price_in_Lakhs"].mean().round(1).to_string())
sns.scatterplot(data=sample, x="Amenity_Count", y="Price_per_SqFt", alpha=.3)
save("q19_amenities_vs_ppsf", "Q19 Amenity count vs price per sq ft",
     f"correlation = {df.Amenity_Count.corr(df.Price_per_SqFt):.3f}")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(data=df, x="Public_Transport_Accessibility", y="Price_per_SqFt", order=["Low", "Medium", "High"], ax=ax[0])
sns.barplot(data=df, x="Public_Transport_Accessibility", y="Good_Investment", order=["Low", "Medium", "High"], ax=ax[1])
ax[1].set_ylabel("Share Good Investment")
save("q20_transport", "Q20 Public transport vs price/sqft and investment potential",
     df.groupby("Public_Transport_Accessibility")[["Price_per_SqFt", "Good_Investment"]].mean().round(3).to_string())

# ---- Extra: link back to the classification target (brief mentions crime rate, which
# this dataset does NOT have - we use infrastructure as the closest proxy) ----------
sns.barplot(data=df, x="Infrastructure_Score", y="Good_Investment")
save("extra_infra_vs_good", "Extra: infrastructure score vs share of Good Investments")

open("eda_findings.txt", "w").write("\n".join(notes))
print("Saved", len(os.listdir("eda_plots")), "plots and eda_findings.txt")
