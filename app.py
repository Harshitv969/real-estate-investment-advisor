"""
STEP 5 - STREAMLIT APP   (run:  streamlit run app.py)

LEARNING NOTES
--------------
Streamlit re-runs this whole script top-to-bottom every time the user touches a widget.
So: load heavy things (data, models) once with @st.cache_*, then just draw widgets.
The app loads the SAME pipelines trained in 03_train.py, so the user's inputs go through
identical scaling/encoding as the training data.
"""
import json
import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

TRANSPORT = {"Low": 1, "Medium": 2, "High": 3}
AMENITIES = ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]


@st.cache_resource
def load_models():
    return joblib.load("models/best_classifier.joblib"), joblib.load("models/best_regressor.joblib")


@st.cache_data
def load_data():
    return pd.read_csv("cleaned_housing.csv.gz")


@st.cache_data
def load_metrics():
    return json.load(open("models/metrics_summary.json"))


clf, reg = load_models()
df = load_data()
metrics = load_metrics()
MAX_AMENITIES = int(df["Amenity_Count"].max())

st.title("🏠 Real Estate Investment Advisor")
st.caption("Is a property a good investment, and what will it be worth in 5 years?")
tab_pred, tab_explore, tab_insight, tab_model = st.tabs(
    ["🔮 Predict", "🔎 Explore properties", "📊 Market insights", "🤖 Model performance"])

# ============================================================== TAB 1: PREDICT
with tab_pred:
    st.subheader("Enter property details")
    c1, c2, c3 = st.columns(3)
    with c1:
        state = st.selectbox("State", sorted(df["State"].unique()))
        city = st.selectbox("City", sorted(df[df["State"] == state]["City"].unique()))
        ptype = st.selectbox("Property type", sorted(df["Property_Type"].unique()))
        bhk = st.slider("BHK", 1, 5, 3)
        size = st.number_input("Size (sq ft)", 500, 5000, 1500, step=50)
        price = st.number_input("Asking price (Lakhs)", 10.0, 500.0, 120.0, step=5.0)
    with c2:
        year = st.slider("Year built", 1990, 2023, 2010)
        furnished = st.selectbox("Furnishing", sorted(df["Furnished_Status"].unique()))
        total_floors = st.slider("Total floors in building", 1, 30, 10)
        floor_no = st.slider("Floor number", 0, 30, 3)
        facing = st.selectbox("Facing", sorted(df["Facing"].unique()))
        owner = st.selectbox("Owner type", sorted(df["Owner_Type"].unique()))
    with c3:
        status = st.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])
        transport = st.selectbox("Public transport access", ["Low", "Medium", "High"], index=1)
        schools = st.slider("Nearby schools", 1, 10, 5)
        hospitals = st.slider("Nearby hospitals", 1, 10, 5)
        parking = st.checkbox("Parking space", True)
        security = st.checkbox("Security", True)
        amen = st.multiselect("Amenities", AMENITIES, default=["Gym", "Garden"])

    if st.button("Analyse property", type="primary"):
        floor_no = min(floor_no, total_floors)
        t_score = TRANSPORT[transport]
        row = pd.DataFrame([{
            "BHK": bhk, "Size_in_SqFt": size, "Price_in_Lakhs": price,
            "Price_per_SqFt": price * 100000 / size, "Age_of_Property": 2025 - year,
            "Nearby_Schools": schools, "Nearby_Hospitals": hospitals, "Floor_No": floor_no,
            "Total_Floors": total_floors, "Amenity_Density_Score": len(amen) / MAX_AMENITIES,
            "Transport_Score": t_score, "Infrastructure_Score": schools + hospitals + 3 * t_score,
            "Floor_Ratio": floor_no / total_floors, "Rooms_per_1000SqFt": bhk / size * 1000,
            "Is_Ready": int(status == "Ready_to_Move"), "Has_Parking": int(parking), "Has_Security": int(security),
            "City": city, "Property_Type": ptype, "Furnished_Status": furnished, "Facing": facing, "Owner_Type": owner,
        }])
        proba = float(clf.predict_proba(row)[0, 1])
        future = float(reg.predict(row)[0])
        cagr = (future / price) ** (1 / 5) - 1

        r1, r2 = st.columns(2)
        with r1:
            if proba >= 0.5:
                st.success(f"✅ Good Investment  (confidence {proba:.1%})")
            else:
                st.error(f"❌ Not a Good Investment  (confidence {1 - proba:.1%})")
            st.progress(proba, text=f"Probability of being a good investment: {proba:.1%}")
        with r2:
            st.metric("Estimated price after 5 years", f"₹ {future:,.1f} Lakhs",
                      f"{future - price:+,.1f} Lakhs  ({cagr:.1%} per year)")

        years = np.arange(0, 6)
        curve = pd.DataFrame({"Year": years, "Price (Lakhs)": price * (future / price) ** (years / 5)})
        st.altair_chart(alt.Chart(curve).mark_line(point=True).encode(
            x="Year:O", y=alt.Y("Price (Lakhs):Q", scale=alt.Scale(zero=False))).properties(
            title="Projected price path (constant growth between today and year 5)", height=280),
            use_container_width=True)

        city_med = df[df["City"] == city]["Price_per_SqFt"].median()
        st.info(f"Price per sq ft: ₹{price * 100000 / size:,.0f} vs {city} median ₹{city_med:,.0f}")

# ============================================================ TAB 2: EXPLORE
with tab_explore:
    st.subheader("Filter properties")
    f1, f2, f3, f4 = st.columns(4)
    states = f1.multiselect("State", sorted(df["State"].unique()))
    cities = f2.multiselect("City", sorted(df[df["State"].isin(states)]["City"].unique() if states else df["City"].unique()))
    bhks = f3.multiselect("BHK", [1, 2, 3, 4, 5])
    types = f4.multiselect("Property type", sorted(df["Property_Type"].unique()))
    p_lo, p_hi = st.slider("Price range (Lakhs)", 10, 500, (10, 500))
    a_lo, a_hi = st.slider("Area range (sq ft)", 500, 5000, (500, 5000))
    only_good = st.checkbox("Only show good investments")

    v = df[df["Price_in_Lakhs"].between(p_lo, p_hi) & df["Size_in_SqFt"].between(a_lo, a_hi)]
    if states: v = v[v["State"].isin(states)]
    if cities: v = v[v["City"].isin(cities)]
    if bhks: v = v[v["BHK"].isin(bhks)]
    if types: v = v[v["Property_Type"].isin(types)]
    if only_good: v = v[v["Good_Investment"] == 1]
    st.write(f"**{len(v):,}** matching properties")
    cols = ["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
            "Price_per_SqFt", "Future_Price_5Y", "Good_Investment"]
    st.dataframe(v[cols].head(500), use_container_width=True)

# ============================================================ TAB 3: INSIGHTS
with tab_insight:
    st.subheader("Market insights")
    st.markdown("**Average price (Lakhs): city × BHK heatmap**")
    top_cities = df["City"].value_counts().head(20).index
    hm = df[df["City"].isin(top_cities)].groupby(["City", "BHK"])["Price_in_Lakhs"].mean().reset_index()
    st.altair_chart(alt.Chart(hm).mark_rect().encode(
        x="BHK:O", y=alt.Y("City:N", sort="-x"),
        color=alt.Color("Price_in_Lakhs:Q", scale=alt.Scale(scheme="viridis")),
        tooltip=["City", "BHK", alt.Tooltip("Price_in_Lakhs:Q", format=".1f")]).properties(height=520),
        use_container_width=True)

    i1, i2 = st.columns(2)
    with i1:
        st.markdown("**Today vs projected 5-year price by city (top 15 by growth)**")
        g = df.groupby("City")[["Price_in_Lakhs", "Future_Price_5Y"]].mean()
        g["growth_%"] = (g["Future_Price_5Y"] / g["Price_in_Lakhs"] - 1) * 100
        st.bar_chart(g.sort_values("growth_%", ascending=False).head(15)["growth_%"])
    with i2:
        st.markdown("**Share of good investments by city (top 15)**")
        st.bar_chart(df.groupby("City")["Good_Investment"].mean().sort_values(ascending=False).head(15))

    st.markdown("**Average price per sq ft (₹) by year built**")
    st.line_chart(df.groupby("Year_Built")["Price_per_SqFt"].mean())

# ======================================================= TAB 4: MODEL PERFORMANCE
with tab_model:
    st.subheader("Model comparison (20% held-out test set)")
    st.markdown(f"Best classifier: **{metrics['best_classifier']}** · Best regressor: **{metrics['best_regressor']}**")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Classification: Good Investment**")
        st.dataframe(pd.DataFrame(metrics["classification"]).T.drop(columns="cm").round(4))
        cm = np.array(metrics["classification"][metrics["best_classifier"]]["cm"])
        st.markdown("Confusion matrix (best model): rows = actual, columns = predicted")
        st.dataframe(pd.DataFrame(cm, index=["Actual: No", "Actual: Yes"], columns=["Pred: No", "Pred: Yes"]))
    with m2:
        st.markdown("**Regression: Price after 5 years**")
        st.dataframe(pd.DataFrame(metrics["regression"]).T.round(4))
    st.subheader("Feature importance (permutation importance)")
    fi1, fi2 = st.columns(2)
    fi1.markdown("Classifier")
    fi1.bar_chart(pd.read_csv("models/importance_classifier.csv", index_col=0).clip(lower=0))
    fi2.markdown("Regressor")
    fi2.bar_chart(pd.read_csv("models/importance_regressor.csv", index_col=0).clip(lower=0))
    st.caption("Run `mlflow ui --backend-store-uri sqlite:///mlflow.db` to browse all experiment runs and the model registry.")
