import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from prophet import Prophet
import joblib
import os

# -----------------------------
# Load Data Function
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('data/transactions.csv', parse_dates=['date'])
    # Filter hanya store 1 dan tanggal Januari 2013 - Juli 2017
    df = df[(df['store_nbr'] == 1) & (df['date'] >= '2013-01-01') & (df['date'] <= '2017-07-31')]
    df_monthly = df.resample('M', on='date').sum().reset_index()
    df_monthly = df_monthly.rename(columns={'date': 'ds', 'transactions': 'y'})
    return df_monthly

# -----------------------------
# Forecast Function
# -----------------------------
def forecast_prophet(df, periods):
    model_path = 'model/prophet_model.pkl'

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = Prophet()
        model.fit(df)
        joblib.dump(model, model_path)

    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    return forecast, model

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Retail Forecasting", layout="wide")
st.title("🛍️ Monthly Transaction Forecasting")
st.caption("By Myrta Prayogo")
st.markdown("""
This application provides an interactive visualization of monthly retail transaction forecasts using time series modeling. Users can explore historical trends and generate forecasts for upcoming months. 
The model is built using Prophet and is designed to support better business planning and decision-making through data-driven insights.
""")

# Load Data
df_monthly = load_data()

# Tampilkan Data Historikal
st.subheader("📈 Historical Monthly Transactions")
st.line_chart(df_monthly.set_index('ds')['y'])

# Pilih Periode Forecast
forecast_months = st.slider("Select number of months to forecast:", 3, 12, 6)

# Tombol Forecast
if st.button("🔮 Forecast"):
    forecast_result, model = forecast_prophet(df_monthly, forecast_months)

    # Filter hasil forecast saja (tanpa data training)
    forecast_only = forecast_result[forecast_result['ds'] > df_monthly['ds'].max()]

    # Tampilkan Tabel Forecast
    st.subheader("📅 Forecast Result")
    st.dataframe(forecast_only[['ds', 'yhat']].head(forecast_months), use_container_width=True)

    # Visualisasi Forecast
    st.subheader("📊 Forecast Visualization")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_monthly['ds'], y=df_monthly['y'], name='Actual'))
    fig.add_trace(go.Scatter(x=forecast_only['ds'], y=forecast_only['yhat'], name='Forecast', line=dict(dash='dot')))
    fig.update_layout(xaxis_title='Month', yaxis_title='Transactions')
    st.plotly_chart(fig, use_container_width=True)
