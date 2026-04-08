import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from prophet import Prophet

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="Sales & Demand Forecasting for Businesses", layout="wide")

st.title("🚀 Sales & Demand Forecasting for Businesses")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df = df.dropna()

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -----------------------
    # AUTO DETECT
    # -----------------------
    target = "sales" if "sales" in df.columns else df.columns[-1]

    date_col = None
    for col in df.columns:
        if "date" in col.lower():
            date_col = col
            break

    # -----------------------
    # DATE FEATURES
    # -----------------------
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])

        df["year"] = df[date_col].dt.year
        df["month"] = df[date_col].dt.month
        df["day"] = df[date_col].dt.day
        df["dayofweek"] = df[date_col].dt.dayofweek

    # -----------------------
    # ML MODEL
    # -----------------------
    features = [col for col in df.columns if col not in [target, date_col]]

    X = df[features]
    y = df[target]

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=150)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # -----------------------
    # KPIs
    # -----------------------
    st.subheader("📊 Business Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", round(mean_absolute_error(y_test, preds), 2))
    col2.metric("R² Score", round(r2_score(y_test, preds), 2))
    col3.metric("Total Sales", int(np.sum(y)))

    # -----------------------
    # ML GRAPH
    # -----------------------
    st.subheader("📈 Actual vs Predicted")

    fig = px.line(pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": preds
    }))

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------
    # PROPHET FORECAST
    # -----------------------
    if date_col:

        st.subheader("🔮 Future Forecast (AI Time-Series)")

        df_prophet = df[[date_col, target]].rename(
            columns={date_col: "ds", target: "y"}
        )

        model_p = Prophet()
        model_p.fit(df_prophet)

        days = st.slider("Select Days to Forecast", 30, 180, 60)

        future = model_p.make_future_dataframe(periods=days)
        forecast = model_p.predict(future)

        fig2 = px.line(forecast, x="ds", y="yhat")
        st.plotly_chart(fig2, use_container_width=True)

        # -----------------------
        # DOWNLOAD
        # -----------------------
        csv = forecast[["ds", "yhat"]].to_csv(index=False).encode()

        st.download_button(
            "📥 Download Forecast",
            csv,
            "forecast.csv",
            "text/csv"
        )

    # -----------------------
    # PREDICTION
    # -----------------------
    st.subheader("🔮 Predict Custom Input")

    input_data = {}

    for col in X.columns:
        input_data[col] = st.number_input(col, value=0.0)

    if st.button("Predict Sales"):

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        st.success(f"💰 Predicted Sales: {round(prediction, 2)}")

else:
    st.info("👈 Upload dataset to start")