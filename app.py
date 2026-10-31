import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="wide")

st.markdown("""
<style>
.main {background-color: #f7faf5;}
h1, h2, h3 {color: #245b38;}
div[data-testid="stMetric"] {
    background: white; padding: 16px; border-radius: 12px;
    border: 1px solid #e0eadf;
}
</style>
""", unsafe_allow_html=True)

st.title("🌾 Crop Yield Prediction")
st.write("A machine-learning dashboard to estimate crop yield from rainfall, temperature and soil type.")
st.caption("Model: Random Forest Regressor • Evaluation: RMSE and R²")

def demo_data(n=700, seed=42):
    rng = np.random.default_rng(seed)
    rain = rng.uniform(200, 1500, n)
    temp = rng.uniform(10, 40, n)
    soils = rng.choice(["Loamy", "Clay", "Sandy", "Black", "Alluvial"], n)
    soil_effect = {"Loamy": 2.0, "Clay": 1.5, "Sandy": 0.5, "Black": 2.5, "Alluvial": 3.0}
    y = 1.5 + rain * 0.004 - (temp - 25) ** 2 * 0.015
    y += np.array([soil_effect[s] for s in soils])
    y += rng.normal(0, 0.6, n)
    return pd.DataFrame({"Rainfall": rain, "Temperature": temp, "Soil_Type": soils, "Yield": np.maximum(y, 0.1)})

def find_col(columns, terms):
    for col in columns:
        normalized = str(col).lower().replace("_", " ").replace("-", " ").strip()
        if any(term in normalized for term in terms):
            return col
    return None

def clean_data(df):
    cols = list(df.columns)
    rain_col = find_col(cols, ["rainfall", "precipitation", "rain"])
    temp_col = find_col(cols, ["temperature", "temp"])
    soil_col = find_col(cols, ["soil type", "soil"])
    yield_col = find_col(cols, ["yield", "production"])
    if not all([rain_col, temp_col, soil_col, yield_col]):
        raise ValueError(
            "Could not detect all required columns. Please include columns for rainfall, "
            "temperature, soil type, and crop yield. Rename them to Rainfall, Temperature, Soil_Type, Yield if needed."
        )
    out = pd.DataFrame({
        "Rainfall": pd.to_numeric(df[rain_col], errors="coerce"),
        "Temperature": pd.to_numeric(df[temp_col], errors="coerce"),
        "Soil_Type": df[soil_col].astype("string").str.strip(),
        "Yield": pd.to_numeric(df[yield_col], errors="coerce")
    })
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out[(out.Rainfall >= 0) & (out.Temperature >= -20) &
              (out.Temperature <= 60) & (out.Yield >= 0) & (out.Soil_Type != "")]
    out["Soil_Type"] = out["Soil_Type"].astype(str)
    if len(out) < 10:
        raise ValueError("At least 10 valid records are required to train and test the model.")
    return out

with st.sidebar:
    st.header("📂 Dataset")
    upload = st.file_uploader("Upload a CSV dataset", type=["csv"])
    st.info("Expected columns: Rainfall, Temperature, Soil_Type, Yield.")
    use_demo = st.checkbox("Use synthetic demo data if no CSV is loaded", value=True)

try:
    if upload is not None:
        data = clean_data(pd.read_csv(upload))
        source = "Uploaded dataset"
        demo = False
    elif os.path.exists("crop_yield.csv"):
        try:
            data = clean_data(pd.read_csv("crop_yield.csv"))
            source = "crop_yield.csv"
            demo = False
        except Exception:
            if not use_demo:
                raise
            data = demo_data()
            source = "Synthetic demonstration data (local CSV could not be read)"
            demo = True
    elif use_demo:
        data = demo_data()
        source = "Synthetic demonstration data"
        demo = True
    else:
        st.warning("Upload a CSV file or enable synthetic demo data in the sidebar.")
        st.stop()
except Exception as exc:
    st.error(f"Dataset error: {exc}")
    st.stop()

if demo:
    st.warning("DEMO MODE: This is synthetic data. Its predictions and metrics are for software demonstration only, not real agricultural advice.")
else:
    st.success(f"Dataset loaded successfully: {source}")

X = data[["Rainfall", "Temperature", "Soil_Type"]]
y = data["Yield"]
if y.nunique() < 2:
    st.error("The yield target must contain at least two different values.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
preprocess = ColumnTransformer([
    ("soil", OneHotEncoder(handle_unknown="ignore"), ["Soil_Type"]),
    ("numbers", "passthrough", ["Rainfall", "Temperature"])
])
model = Pipeline([
    ("preprocess", preprocess),
    ("regressor", RandomForestRegressor(
        n_estimators=150, random_state=42, min_samples_leaf=2, n_jobs=-1
    ))
])
model.fit(X_train, y_train)
test_pred = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
r2 = float(r2_score(y_test, test_pred)) if len(y_test) >= 2 else float("nan")

m1, m2, m3 = st.columns(3)
m1.metric("Dataset records", f"{len(data):,}")
m2.metric("RMSE", f"{rmse:.3f}")
m3.metric("R² score", f"{r2:.3f}")
st.caption("RMSE is in the same units as the dataset's yield target. Metrics are calculated on a held-out test split.")

st.divider()
st.header("🌱 Predict crop yield")
left, mid, right = st.columns(3)
with left:
    rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=10000.0, value=float(np.clip(data.Rainfall.median(), 0, 10000)), step=10.0)
with mid:
    temp = st.number_input("Temperature (°C)", min_value=-20.0, max_value=60.0, value=float(np.clip(data.Temperature.median(), -20, 60)), step=1.0)
with right:
    soils = sorted(data.Soil_Type.unique().tolist())
    soil = st.selectbox("Soil type", soils)

if st.button("🔍 Predict yield", type="primary", use_container_width=True):
    pred = float(model.predict(pd.DataFrame({
        "Rainfall": [rain], "Temperature": [temp], "Soil_Type": [soil]
    }))[0])
    st.subheader("Prediction result")
    st.metric("Estimated crop yield", f"{max(0, pred):.2f} yield units")
    st.info("The unit is inherited from the dataset's yield column (for example, tonnes per hectare if that is how your data is measured).")
    if rain < data.Rainfall.min() or rain > data.Rainfall.max() or temp < data.Temperature.min() or temp > data.Temperature.max():
        st.warning("At least one numeric input is outside the training-data range; this prediction may be less reliable.")

with st.expander("📊 Explore dataset"):
    st.dataframe(data, use_container_width=True)
    st.write("Numeric summary")
    st.dataframe(data[["Rainfall", "Temperature", "Yield"]].describe(), use_container_width=True)

with st.expander("Compare actual vs predicted values"):
    comparison = pd.DataFrame({"Actual yield": y_test.to_numpy(), "Predicted yield": test_pred})
    st.scatter_chart(comparison, x="Actual yield", y="Predicted yield")
    st.dataframe(comparison.head(30), use_container_width=True)
    st.download_button("Download test predictions (CSV)", comparison.to_csv(index=False), "yield_test_predictions.csv", "text/csv")

st.divider()
st.caption("Crop Yield Prediction • Python • Streamlit • Scikit-learn")
