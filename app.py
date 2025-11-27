
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px

st.set_page_config(page_title="Anmol Soil Health Monitoring (IoT + ML)", page_icon="🌱", layout="wide")

st.title("🌱 Anmol Soil Health Monitoring Dashboard")
st.write("Upload sensor data or use sample data to predict soil moisture and visualize trends.")

@st.cache_data
def load_sample():
    return pd.read_csv("soil_data.csv")

st.sidebar.header("Data")
use_sample = st.sidebar.checkbox("Use bundled sample dataset", value=True)
uploaded = st.sidebar.file_uploader("Or upload CSV", type=["csv"])

if use_sample:
    df = load_sample()
else:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        st.warning("Upload a CSV or tick 'Use bundled sample dataset'.")
        st.stop()

st.subheader("Data Preview")
st.dataframe(df.head())

features = [c for c in df.columns if c != "moisture_pct"]
target = "moisture_pct" if "moisture_pct" in df.columns else None

st.markdown("### Feature Distributions")
col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(df, x="ph", nbins=30, title="pH")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.histogram(df, x="ec_dSm", nbins=30, title="EC (dS/m)")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Train Model")
if target:
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = RandomForestRegressor(n_estimators=200, random_state=0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    st.write(f"**R²:** {r2_score(y_test, pred):.3f} | **MAE:** {mean_absolute_error(y_test, pred):.2f}")

    st.markdown("### Predict on New Readings")
    cols = st.columns(len(features))
    inputs = []
    for i, f in enumerate(features):
        val = float(df[f].median())
        inputs.append(cols[i].number_input(f, value=val))
    if st.button("Predict Moisture"):
        new = np.array(inputs).reshape(1, -1)
        yhat = model.predict(new)[0]
        st.success(f"Estimated Soil Moisture: **{yhat:.2f}%**")

st.markdown("""---
### Deployment
- **Streamlit Cloud**: push this repo to GitHub, deploy at https://share.streamlit.io (entrypoint `app.py`)
- **Local run**: `pip install -r requirements.txt && streamlit run app.py`
""")
