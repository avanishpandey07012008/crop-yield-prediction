
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Crop Yield Prediction")
st.write("Predict agricultural crop yield using Machine Learning.")

# Generate demo dataset
np.random.seed(42)

n = 600

rainfall = np.random.uniform(200, 1500, n)
temperature = np.random.uniform(10, 40, n)

soil = np.random.choice(
    ["Loamy", "Clay", "Sandy", "Black", "Alluvial"], n
)

effects = {
    "Loamy": 2,
    "Clay": 1.5,
    "Sandy": 0.5,
    "Black": 2.5,
    "Alluvial": 3
}

yield_data = (
    1.5
    + rainfall * 0.004
    - (temperature - 25)**2 * 0.015
    + np.array([effects[s] for s in soil])
    + np.random.normal(0, 0.6, n)
)

data = pd.DataFrame({
    "Rainfall": rainfall,
    "Temperature": temperature,
    "Soil_Type": soil,
    "Yield": np.maximum(yield_data, 0.1)
})

st.warning("Demo mode: synthetic data is being used.")

# Model training
X = pd.get_dummies(
    data[["Rainfall", "Temperature", "Soil_Type"]]
)

y = data["Yield"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

# Dashboard
st.header("Model Performance")

c1, c2, c3 = st.columns(3)

c1.metric("Dataset Records", len(data))
c2.metric("RMSE", round(rmse, 3))
c3.metric("R² Score", round(r2, 3))

st.divider()

# User input
st.header("🌱 Predict Crop Yield")

rain = st.number_input(
    "Rainfall (mm)", 0.0, 10000.0, 800.0
)

temp = st.number_input(
    "Temperature (°C)", -20.0, 60.0, 25.0
)

soil_type = st.selectbox(
    "Soil Type",
    ["Loamy", "Clay", "Sandy", "Black", "Alluvial"]
)

if st.button("Predict Crop Yield"):

    new_data = pd.DataFrame({
        "Rainfall": [rain],
        "Temperature": [temp],
        "Soil_Type": [soil_type]
    })

    new_data = pd.get_dummies(new_data)

    new_data = new_data.reindex(
        columns=X.columns,
        fill_value=0
    )

    result = model.predict(new_data)[0]

    st.success(
        f"Estimated Crop Yield: {result:.2f} yield units"
    )

st.subheader("Dataset Preview")
st.dataframe(data.head(20))

st.subheader("Actual vs Predicted Yield")

chart = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

st.scatter_chart(chart, x="Actual", y="Predicted")
